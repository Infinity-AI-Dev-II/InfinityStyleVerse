from dotenv import load_dotenv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, inspect, text
from backend.app import create_app
from backend.app.database import db

load_dotenv()

# Creating app and getting the database connection
app = create_app()
with app.app_context():
    engine = db.engine  
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("Available tables:", tables)

    # Verifying a connection and checking a specific table (Eg: run_steps)
    if 'run_steps' in tables:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT COUNT(*) FROM run_steps"))
            row = result.fetchone()
            print(f"Number of rows in run_steps: {row[0]}")
    else:
        print("run_steps table not found.")

    print("Connection successful!")