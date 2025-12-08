from datetime import datetime
from ..database import db


"""
use this later in step creation

@app.task(bind=True)
def my_task(self, *args, **kwargs):
    worker_id = self.request.hostname  # Built-in! e.g., "celery@worker1"
    # Use for /pulse/hooks/step
    requests.post("/pulse/hooks/step", json={"worker_id": worker_id, ...})

"""
class RunStep(db.Model):
    __tablename__ = "run_steps"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_id = db.Column(db.Integer, db.ForeignKey("runs.id"), nullable=False)
    step_id = db.Column(db.String(100), nullable=False)
    #TODO: get the worker ID from flower when creating a step
    worker_id = db.Column(db.String(128), nullable=False) #TODO: Test the eligibility
    type = db.Column(db.String(50))
    status = db.Column(db.String(50))
    attempt = db.Column(db.Integer, default=0)
    started_at = db.Column(db.DateTime)
    ended_at = db.Column(db.DateTime)
    input_json = db.Column(db.Text)
    output_json = db.Column(db.Text)
    error_json = db.Column(db.Text)
