# tests/test_wait_signal.py
def test_wait_signal_timer(db_session, celery_worker):
    from backend.app.tasks.wait_signal import wait_step_task, schedule_wait_step
    from backend.app.models import WaitStepTimer  # <-- import from models

    # Schedule a 1-second wait step
    timer_id = schedule_wait_step(run_id=1, step_id="approval", wait_seconds=1)

    # Wait for Celery task to complete
    wait_step_task.apply(args=[timer_id]).get(timeout=10)

    # Fetch timer from DB and assert status
    timer = db_session.query(WaitStepTimer).get(timer_id)
    assert timer.status == "triggered"
