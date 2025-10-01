# backend/app/tasks/wait_signal.py
from backend.celery_app import celery_app
from backend.app.models import WaitStepTimer  # <-- import existing model
from backend.app.database import SessionLocal
from datetime import datetime, timedelta
import time
import logging
from backend.app.tasks.saga import execute_saga
from backend.app.tasks.step_functions import STEP_REGISTRY


logger = logging.getLogger(__name__)

# ------------------------
# Celery Task
# ------------------------
@celery_app.task(bind=True, name="backend.app.tasks.wait_signal.wait_step_task")
def wait_step_task(self, timer_id: int):
    db = SessionLocal()
    try:
        timer = db.query(WaitStepTimer).get(timer_id)
        if not timer:
            logger.error(f"Timer {timer_id} not found")
            return
        now = datetime.utcnow()
        if timer.trigger_at > now:
            sleep_seconds = (timer.trigger_at - now).total_seconds()
            time.sleep(sleep_seconds)
        timer.status = "triggered"
        db.commit()
        logger.info(f"Timer {timer_id} triggered for run {timer.run_id} step {timer.step_id}")
    except Exception as e:
        logger.error(f"Error in wait_step_task: {e}")
        db.rollback()
    finally:
        db.close()

# ------------------------
# Helper to schedule a wait step
# ------------------------
def schedule_wait_step(run_id: int, step_id: str, wait_seconds: int) -> int:
    db = SessionLocal()
    try:
        trigger_time = datetime.utcnow() + timedelta(seconds=wait_seconds)
        timer = WaitStepTimer(run_id=run_id, step_id=step_id, trigger_at=trigger_time)
        db.add(timer)
        db.commit()
        db.refresh(timer)
        # Enqueue Celery task
        wait_step_task.apply_async(args=[timer.id], countdown=wait_seconds)
        logger.info(f"Scheduled wait step {step_id} for run {run_id} with timer_id {timer.id}")
        return timer.id
    finally:
        db.close()


@celery_app.task(bind=True, name="backend.app.tasks.wait_signal.wait_step_task")
def wait_step_task(self, run_id, step_ids):
    steps = []
    for step_id in step_ids:
        step_info = STEP_REGISTRY.get(step_id)
        if step_info:
            steps.append({"id": step_id, "type": "main", "execute_fn": step_info["execute_fn"]})
    
    result = execute_saga(run_id, steps)
    return result
