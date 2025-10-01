import pytest
from backend.app import create_app
from backend.app.database import db
from backend.app.models.roles import Role
from backend.app.models.user import User
from werkzeug.security import generate_password_hash
from flask_jwt_extended import create_access_token, create_refresh_token
from backend.app.database import SessionLocal, Base, engine
# ---------------------------
# App fixture
# ---------------------------
@pytest.fixture(scope="function")
def app_ctx():
    app = create_app("testing")
    app.config["JWT_COOKIE_CSRF_PROTECT"] = False  # disable CSRF for tests
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["TESTING"] = True

    with app.app_context():
        db.drop_all()
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

# ---------------------------
# Clean DB before each test
# ---------------------------
@pytest.fixture(autouse=True)
def clean_db(app_ctx):
    db.session.rollback()
    for tbl in reversed(db.metadata.sorted_tables):
        db.session.execute(tbl.delete())
    db.session.commit()

# ---------------------------
# Test client fixture
# ---------------------------
@pytest.fixture(scope="function")
def client(app_ctx):
    return app_ctx.test_client()

# ---------------------------
# Seed roles and users fixture
# ---------------------------
@pytest.fixture(scope="function")
def seed_roles_and_users(app_ctx):
    """Seed roles and test users into DB"""
    # Clear DB
    User.query.delete()
    Role.query.delete()
    db.session.commit()

    # Create roles
    admin_role = Role(role_name="admin")
    user_role = Role(role_name="user")
    db.session.add_all([admin_role, user_role])
    db.session.commit()

    # Create users
    admin = User(
        name="Admin",
        email="admin@example.com",
        password_hash=generate_password_hash("Admin123!"),
        role_id=admin_role.id,
        status="Active"
    )
    alice = User(
        name="Alice",
        email="alice@example.com",
        password_hash=generate_password_hash("User123!"),
        role_id=user_role.id,
        status="Active"
    )
    db.session.add_all([admin, alice])
    db.session.commit()

    return admin, alice  # return users for convenience

# ---------------------------
# get_user fixture
# ---------------------------
@pytest.fixture(scope="function")
def get_user(app_ctx):
    """Return a function to fetch a user by email"""
    def _get_user(email: str) -> User:
        return User.query.filter_by(email=email).first()
    return _get_user

# ---------------------------
# login_and_get_tokens fixture
# ---------------------------
@pytest.fixture(scope="function")
def login_and_get_tokens(app_ctx, get_user):
    """
    Logs in a user by email and password.
    Returns (access_token, refresh_token)
    """
    def _login(email="alice@example.com", password="User123!"):
        user = get_user(email)
        if not user:
            raise RuntimeError(f"User {email} not found.")
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)
        return access_token, refresh_token
    return _login

@pytest.fixture(scope="function")
def db_session():
    # Create tables
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(bind=engine)

from backend.celery_app import celery_app

@pytest.fixture(scope="function")
def celery_worker():
    # Use solo pool to run tasks in the same process
    from celery.contrib.testing.worker import start_worker
    with start_worker(celery_app, pool="solo", perform_ping_check=False) as worker:
        yield worker

@pytest.fixture
def app():
    return create_app()
