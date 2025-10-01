from datetime import datetime
from backend.app.database import SessionLocal
from backend.app.models.run_steps import RunStep
import logging
import time

logger = logging.getLogger(__name__)

def execute_step(run_id: int, step_id: str, crash_midway: bool = False):
    """
    Dummy step executor.
    Writes a deterministic output to RunStep.output_json.
    If crash_midway=True, simulates a crash before marking completed.
    """
    session = SessionLocal()
    step = (
        session.query(RunStep)
        .filter_by(run_id=run_id, step_id=step_id)
        .with_for_update()
        .first()
    )

    if not step:
        logger.error(f"RunStep {step_id} not found for run {run_id}")
        session.close()
        return

    # If already completed → idempotency guard
    if step.status == "completed":
        logger.info(f"Step {step_id} already completed, skipping.")
        session.close()
        return

    step.status = "in_progress"
    step.attempt = (step.attempt or 0) + 1
    step.started_at = datetime.utcnow()
    session.commit()

    # simulate doing work
    time.sleep(0.2)

    # simulate crash
    if crash_midway:
        logger.warning(f"Simulating crash on step {step_id}")
        session.close()
        raise SystemExit("Crash simulated!")

    # complete the step deterministically
    step.output_json = f"step_{step_id}_done"
    step.status = "completed"
    step.ended_at = datetime.utcnow()
    session.commit()
    session.close()
