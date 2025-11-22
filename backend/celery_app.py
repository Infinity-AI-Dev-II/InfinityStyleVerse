# backend/celery_app.py
#Sets up and configures the Celery application for the workflow engine
from celery import Celery
from kombu import Queue
from backend.app.config.settings import settings
from backend.app.logging_conf import configure_logging

# configure logging early so signals and tasks use structured JSON log
#Calls for in the task creations 
configure_logging()
#create the celery application
celery_app = Celery(
    "workflow_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=60,
    task_soft_time_limit=50,
    result_expires=3600,
    task_default_retry_delay=5,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
#specific custom queues
celery_app.conf.task_queues = (
    Queue("default"),
    Queue("io"),
    Queue("cpu"),
    Queue(name='flowGate_queue'),
    Queue(name='image_queue')
)
#send the tasks to the relavent queues
celery_app.conf.task_routes = {
    "backend.app.tasks.http_call.http_call": {"queue": "io"},
    "backend.app.tasks.python_fn.python_fn": {"queue": "cpu"},
    "backend.app.tasks.anomaly_detector_functions.detect_anomalies": {"queue": "flowGate_queue"},
    "backend.app.tasks.GetImageBySignedUrl.load_image_from_signed_url": {"queue": "image_queue"}
}

# register signals so they're active in worker process
import backend.app.celery_signals  # noqa: F401
