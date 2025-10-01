# backend/app/api/endpoints/signals.py
from fastapi import APIRouter
from backend.app.database import get_db
from backend.app.models import WaitStepTimer
from sqlalchemy.orm import Session
from backend.app.tasks.wait_signal import wait_step_task

router = APIRouter()

@router.post("/flow/runs/{run_id}/signal/{signal_name}")
def send_signal(run_id: int, signal_name: str):
    db: Session = get_db()
    # Find pending timers waiting for this signal
    timers = db.query(WaitStepTimer).filter(
        WaitStepTimer.run_id == run_id,
        WaitStepTimer.step_id == signal_name,
        WaitStepTimer.status == "pending"
    ).all()
    for timer in timers:
        timer.status = "triggered"
        db.commit()
        # enqueue step immediately
        wait_step_task.delay(timer.id)

    return {"success": True, "triggered_timers": len(timers)}
