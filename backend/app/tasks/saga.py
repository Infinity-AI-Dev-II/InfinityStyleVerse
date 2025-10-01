import logging
from datetime import datetime
from backend.app.database import db
from backend.app.models.run_steps import RunStep
from backend.app.models.compensations import Compensation
from backend.app.models import Run
from .step_functions import STEP_REGISTRY

logger = logging.getLogger(__name__)


def create_run(run_id):
    """
    Create a Run entry safely inside an app context.
    """
    run = Run(id=run_id)
    db.session.add(run)
    db.session.commit()
    return run


def execute_saga(run_id, steps):
    """
    Execute steps sequentially and handle compensation on failure.
    
    steps: List of dicts with keys:
        - id: step_id
        - type: optional, type of step
        - execute_fn: callable function
    """
    executed_steps = []

    for step in steps:
        # Fetch or create RunStep in DB
        run_step = RunStep.query.filter_by(run_id=run_id, step_id=step["id"]).first()
        if not run_step:
            run_step = RunStep(
                run_id=run_id,
                step_id=step["id"],
                type=step.get("type"),
                status="pending",
                started_at=None,
                ended_at=None
            )
            db.session.add(run_step)
            db.session.commit()

        try:
            run_step.started_at = datetime.utcnow()
            db.session.commit()

            # Execute the main step
            output = step["execute_fn"](run_step.input_json)
            run_step.output_json = output
            run_step.status = "completed"
            run_step.ended_at = datetime.utcnow()
            db.session.commit()

            executed_steps.append(run_step)

        except Exception as e:
            logger.error(f"Step {step['id']} failed: {str(e)}")
            run_step.status = "failed"
            run_step.error_json = str(e)
            run_step.ended_at = datetime.utcnow()
            db.session.commit()

            # Run rollback for executed steps
            rollback_executed_steps(run_id, executed_steps)
            return {"status": "failed", "failed_step": step["id"]}

    return {"status": "success"}


def rollback_executed_steps(run_id, executed_steps):
    """
    Run compensation steps in reverse order of execution
    """
    for run_step in reversed(executed_steps):
        compensation = Compensation.query.filter_by(run_id=run_id, step_id=run_step.step_id).first()
        if compensation and compensation.status != "completed":
            try:
                # Get the compensation function from registry
                step_fn = get_compensation_fn(run_step.step_id)
                if step_fn:
                    step_fn(compensation.payload_json or run_step.output_json)

                compensation.status = "completed"
                db.session.commit()
                logger.info(f"Compensation for {run_step.step_id} completed")

            except Exception as e:
                logger.error(f"Compensation for {run_step.step_id} failed: {str(e)}")
                compensation.status = "failed"
                db.session.commit()


def get_compensation_fn(step_id):
    """
    Retrieve the compensation function for a given step_id
    """
    return STEP_REGISTRY.get(step_id, {}).get("compensate_fn")
