from datetime import datetime, timedelta
from backend.app.models import WaitStepTimer
from backend.app.database import get_db
from backend.app.tasks.wait_signal import wait_step_task

def schedule_wait_step(run_id: int, step_id: str, wait_seconds: int, payload=None):
    db = get_db()
    timer = WaitStepTimer(
        run_id=run_id,
        step_id=step_id,
        trigger_at=datetime.utcnow() + timedelta(seconds=wait_seconds),
        payload_json=payload or {},
    )
    db.add(timer)
    db.commit()
    db.refresh(timer)

    # enqueue celery task
    wait_step_task.apply_async((timer.id,), eta=timer.trigger_at)
    return timer.id
