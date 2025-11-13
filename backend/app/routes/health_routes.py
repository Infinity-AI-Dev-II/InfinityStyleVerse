from celery import current_app
from flask import Blueprint, jsonify
from backend.app.database import db
import redis
from kafka import KafkaAdminClient


health_bp = Blueprint("health", __name__)

@health_bp.route("/healthz", methods=["GET"])
def healthz():
    health = {"db": False, "redis": False, "kafka": False}

    # ------------------
    # Database check
    # ------------------
    try:
      
        with current_app.app_context():
            db.session.execute("SELECT 1")
            health["db"] = True
    except Exception as e:
        health["db"] = False
        print("DB Health check failed:", e)

    # ------------------
    # Redis check
    try:
        r = redis.Redis(host="redis", port=6379, db=0)
        r.ping()
        health["redis"] = True
    except Exception:
        pass

    # Kafka check
    try:
        admin = KafkaAdminClient(bootstrap_servers="kafka:9092", client_id="healthcheck")
        admin.list_topics()
        health["kafka"] = True
    except Exception:
        pass

    return jsonify(health)

