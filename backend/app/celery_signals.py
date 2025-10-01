# backend/app/celery_signals.py
import logging
from celery.signals import task_prerun, task_success, task_failure
from backend.app.metrics import TASKS_STARTED, TASKS_FAILED

logger = logging.getLogger(__name__)

@task_prerun.connect
def on_task_prerun(sender=None, task_id=None, task=None, **kwargs):
    # log with task_id in extras
    logging.LoggerAdapter(logger, {"task_id": task_id}).info(f"task_prerun: {sender.name if sender else task}")
    TASKS_STARTED.labels(sender.name if sender else "unknown").inc()

@task_success.connect
def on_task_success(sender=None, result=None, **kwargs):
    tid = kwargs.get("task_id")
    logging.LoggerAdapter(logger, {"task_id": tid}).info("task_success")

@task_failure.connect
def on_task_failure(sender=None, exception=None, **kwargs):
    tid = kwargs.get("task_id")
    logging.LoggerAdapter(logger, {"task_id": tid}).error(f"task_failure: {exception}")
    TASKS_FAILED.labels(sender.name if sender else "unknown").inc()
