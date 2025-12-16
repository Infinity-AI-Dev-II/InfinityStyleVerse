#this is loaded by the docker file and calls the create app function.
#initializes the backend through docker
import os
from flask import Flask, Response, request, send_from_directory, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, get_jwt_identity, verify_jwt_in_request
from flask_login import LoginManager
from flasgger import Swagger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import logging
from dotenv import load_dotenv
from backend.app.middleware.request_id import register_request_id_middleware

# Database and app initialization
from backend.app.database import db, init_db
from backend.app.services.GetHeartBeatPolicy import getLatestWorkerPolicy
from backend.app.services.get_SLA_alert_rules import get_Latest_SLA_rule
from backend.app.services.kafka_service import create_kafka_topics
from backend.app.sse import sse_bp

# Import models for reference
from .models import RequestLog, TokenBlocklist
from .models.user import User 

# Import settings
from .config.settings import settings
import os
#TODO: Build
from flask_caching import Cache
import redis
from flask_sse import sse
# Load environment variables
load_dotenv()

# Initialize extensions
migrate = Migrate()
login_manager = LoginManager()

#worker heartbeat policy cache
policies_cache = {}
alertRule = {}
def create_app(config_name=None):
    # Base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(base_dir, '..', '..', 'Frontend', 'templates')
    static_folder = os.path.join(base_dir, '..', '..', 'Frontend', 'assets')
    # Create Flask app
    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder
    )
    #create kafka topics
    create_kafka_topics()
    #set the global worker policy cache
    global policies_cache
    policies_cache = getLatestWorkerPolicy()
    
    global alertRule
    alertRult = get_Latest_SLA_rule()
    
    #redirect flask logger to guicorn server (to debug in docker)
    gunicorn_logger = logging.getLogger("gunicorn.error")
    if gunicorn_logger.handlers:
       app.logger.handlers = gunicorn_logger.handlers
       app.logger.setLevel(gunicorn_logger.level)
    #configure Redis to use caching
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_HOST'] = os.getenv('REDIS_HOST')
    app.config['CACHE_REDIS_PORT'] = os.getenv('REDIS_PORT')
    app.config['CACHE_REDIS_DB'] = 0
    cache = Cache(app=app)
    cache.init_app(app)
    # Initialize Redis client
    redis_client = redis.Redis(host=app.config['CACHE_REDIS_HOST'], port=app.config['CACHE_REDIS_PORT'], db=app.config['CACHE_REDIS_DB'])
    app.config["REDIS_CLIENT"] = redis_client
    # Load configuration
    app.config.from_object(settings)

    # Initialize DB (calls db.init_app internally)
    init_db(app)

    # Initialize other extensions
    CORS(app)
    migrate.init_app(app, db)
    jwt = JWTManager(app)
    login_manager.init_app(app)

    # -------------------- Prometheus Metrics --------------------
    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    # -------------------- Serve frontend scripts --------------------
    @app.route('/scripts/<path:filename>')
    def serve_scripts(filename):
        scripts_dir = os.path.join(base_dir, '..', '..', 'Frontend', 'scripts')
        return send_from_directory(scripts_dir, filename)

    # -------------------- Request logging --------------------
    @app.before_request
    def log_request():
        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except Exception:
            pass

        log = RequestLog(
            user_id=user_id,
            endpoint=request.path,
             method=request.method
        )
        db.session.add(log)
        db.session.commit()

    # -------------------- Orchestration Logger --------------------
    orch_logger = logging.getLogger("infinitybrain")
    orch_logger.setLevel(app.config.get("ORCHESTRATOR_LOG_LEVEL", "INFO"))

    # -------------------- Flask-Login user loader --------------------
    from .models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return User.query.get(int(user_id))
        except Exception:
            return None

    # -------------------- JWT Callbacks --------------------
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        jti = jwt_payload.get("jti")
        return TokenBlocklist.query.filter_by(jti=jti).first() is not None

    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        return jsonify({"msg": "Token has been revoked"}), 401

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({"msg": "Token has expired"}), 401

    # -------------------- Swagger Configuration --------------------
    app.config['SWAGGER'] = {'title': 'Infinity StyleVerse API', 'uiversion': 3}

    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "InfinityStyleVerse API",
            "description": "Auth, Admin, and InfinityBrain orchestration APIs.\n\n"
                           "**Auth note:** Use the Authorize button and paste `Bearer <access_token>`.",
            "version": "1.0.0"
        },
        "basePath": "/",
        "securityDefinitions": {
            "BearerAuth": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "JWT Authorization header using the Bearer scheme. Example: `Bearer eyJhbGciOi...`"
            }
        },
        "security": [{"BearerAuth": []}],
        "tags": [
            {"name": "Auth", "description": "Authentication & profile"},
            {"name": "Admin - Users", "description": "User management (admin only)"},
            {"name": "Admin - Roles", "description": "Role & Permission management (admin only)"},
            {"name": "InfinityBrain", "description": "Mock AI gateway/services"}
        ],
        "x-rbac": "Routes with admin-only access require role: admin"
    }

    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }

    Swagger(app, template=swagger_template, config=swagger_config)

    # -------------------- Seed command --------------------
    from .seeds.seed_data import run_seed

    @app.cli.command("seed-db")
    def seed_db():
        """Seed database with default roles, permissions, and sample users."""
        run_seed()

    # -------------------- Register Request ID Middleware --------------------
    register_request_id_middleware(app)
    # -------------------- Import and register blueprints --------------------
    from .routes.auth_routes import auth_bp
    from .routes.product_routes import product_bp
    from .routes.main import main_bp
    from .routes.feedback_routes import feedback_bp
    from .routes.recommendation_routes import recommendation_bp
    from .routes.admin_routes import admin_bp
    from .routes.design_routes import design_bp
    from backend.app.routes.persona_mesh import persona_mesh_bp
    from .routes.infinitybrain_routes import ib_bp
    from .routes.workflow_routes import bp as workflow_bp
    from .routes.flow import flow_bp
    from .routes.health_routes import health_bp
    from .routes.echo_routes import echo_bp
    from .StyleSense.StyleAPI.BodyMorphRoutes import bodyMorph_bp
    from backend.app.routes.TaskPulseOS_routes import TaskPulse_bp
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(sse_bp)
    app.register_blueprint(flow_bp, url_prefix="/flow")
    app.register_blueprint(workflow_bp)
    app.register_blueprint(ib_bp)
    app.register_blueprint(design_bp, url_prefix='/designs')
    app.register_blueprint(admin_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(product_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(persona_mesh_bp)
    app.register_blueprint(echo_bp)
    app.register_blueprint(bodyMorph_bp)
    app.register_blueprint(TaskPulse_bp)
    #app.register_blueprint(sse, url_prefix="/pulse/stream")
    return app