from backend.app import create_app
from backend.app.database import db

# Import all your models here so SQLAlchemy knows about them
from backend.app.models.user import User
from backend.app.models.design import Design
from backend.app.models.product import Product
from backend.app.models.feedback import Feedback
from backend.app.models.echo_models import RequestLog


def init_db():
    app = create_app()
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")

if __name__ == "__main__":
    init_db()
