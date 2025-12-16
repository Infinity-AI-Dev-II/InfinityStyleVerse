# backend/app/database.py
#   Centralizes database configuration and session management for both
#   Flask web app and standalone scripts/Celery tasks.
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from .config.settings import settings
from flask_sqlalchemy import SQLAlchemy

# ------------------------------
# Flask-SQLAlchemy for app usage
# ------------------------------
db = SQLAlchemy()  # use db.Model for your models

# ------------------------------
# Standard SQLAlchemy for scripts/tasks
# ------------------------------
DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, expire_on_commit=False))

# Base for declarative models (non-Flask)
Base = declarative_base()
@contextmanager
def get_db_session():
    """Yield a SQLAlchemy session for scripts or Celery tasks."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def init_db(app=None):
    """
    Initialize Flask app with SQLAlchemy and create tables if not exist.
    Can be used standalone if 'app' is provided.
    """
    # if app:
    #     db.init_app(app)
    #     with app.app_context():
    #         db.create_all()
    if app:
        db.init_app(app)
