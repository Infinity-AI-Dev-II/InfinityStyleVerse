# backend/app/workflow_engine.py
from backend.app.database import db
from backend.app.models.run_steps import RunStep
from backend.app.event_publisher import publish_step_event
from datetime import datetime

def update_run_step(run_id, step_id, status):
    step = RunStep.query.filter_by(run_id=run_id, step_id=step_id).first()
    if not step:
        step = RunStep(
            run_id=run_id,
            step_id=step_id,
            status=status,
            started_at=datetime.utcnow(),
            ended_at=datetime.utcnow()
        )
        db.session.add(step)
    else:
        step.status = status
        step.ended_at = datetime.utcnow()

    db.session.commit()

    # Push live update
    publish_step_event(run_id, step_id, status)
