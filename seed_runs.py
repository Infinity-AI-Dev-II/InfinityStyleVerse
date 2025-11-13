# seed_runs.py
#   Standalone script to seed the database with a test Run entry.
from backend.app import app       # import the Flask app object
from backend.app.database import db, init_db
from backend.app.models import Run, workflow_defs

with app.app_context():
    # Create a sample Workflow if none exists
    if not workflow_defs.query.first():
        workflow = workflow_defs(name="Test Workflow")
        db.session.add(workflow)
        db.session.commit()
    else:
        workflow = workflow_defs.query.first()

    # Create a test Run
    test_run = Run(
        name="Test Run",
        workflow_id=workflow.id,   # link to existing workflow
        status="pending"           # or whatever your model uses
    )

    db.session.add(test_run)
    db.session.commit()
    print(f"Test Run created with ID: {test_run.id}")
