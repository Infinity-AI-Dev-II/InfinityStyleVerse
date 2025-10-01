# backend/app/tasks/workflow_tasks.py
from celery import Celery
from ..database import db
from ..models import Run, RunStep
from datetime import datetime

celery = Celery("workflow_tasks", broker="redis://localhost:6379/0")  # Update Redis URL if needed

@celery.task(bind=True)
def execute_step(self, step_id, run_id):
    step = RunStep.query.filter_by(run_id=run_id, step_id=step_id).first()
    if not step or step.status in ["success", "skipped"]:
        return

    step.status = "running"
    step.started_at = datetime.utcnow()  # optional timestamp
    db.session.commit()

    try:
        # Simulate or run actual step logic
        result = f"Executed step {step_id}"  # Replace with real logic
        step.output = {"result": result}
        step.status = "success"
    except Exception as e:
        step.status = "failed"
        step.output = {"error": str(e)}
    finally:
        step.ended_at = datetime.utcnow()  # optional timestamp
        db.session.commit()

    # Update run status if all steps done
    run = Run.query.get(run_id)
    pending = RunStep.query.filter_by(run_id=run_id, status="pending").count()
    running = RunStep.query.filter_by(run_id=run_id, status="running").count()
    if pending + running == 0:
        run.status = "success"
        db.session.commit()
