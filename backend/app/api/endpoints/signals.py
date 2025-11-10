# backend/app/api/endpoints/signals.py
from flask import Blueprint, jsonify
from backend.app.database import get_db
from backend.app.models import WaitStepTimer
from backend.app.tasks.wait_signal import wait_step_task

signals_bp = Blueprint("signals", __name__)

# -------------------------------------------------------------------------
# Endpoint: POST /flow/runs/<run_id>/signal/<signal_name>
# Purpose:
#   - Acts as a signal handler for workflow runs.
#   - When a signal is received, it finds all timers in the database
#     (WaitStepTimer) that are waiting on this signal and still "pending".
#   - Updates their status to "triggered" and commits to DB.
#   - Immediately enqueues the next step via Celery task (wait_step_task).
#   - Returns how many timers were triggered.
# -------------------------------------------------------------------------


@signals_bp.route("/flow/runs/<int:run_id>/signal/<signal_name>", methods=["POST"])
def send_signal(run_id, signal_name):
    db = get_db()
    # Find pending timers waiting for this signal
    timers = db.query(WaitStepTimer).filter(
        WaitStepTimer.run_id == run_id,
        WaitStepTimer.step_id == signal_name,
        WaitStepTimer.status == "pending"
    ).all()

    triggered_count = 0
    for timer in timers:
        timer.status = "triggered"
        db.commit()
        # enqueue step immediately
        wait_step_task.delay(timer.id)
        triggered_count += 1

    return jsonify({"success": True, "triggered_timers": triggered_count})
