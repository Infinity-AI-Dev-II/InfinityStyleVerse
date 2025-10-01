# backend/app/tasks/decorators.py
import time
from functools import wraps
from backend.app.metrics import TASKS_STARTED, TASKS_FAILED, TASKS_TIME
import logging

logger = logging.getLogger(__name__)

def instrument(name: str):
    """
    Decorator to instrument a function/task with Prometheus metrics and basic logs.
    Use above Celery @task to combine async behavior with instrumentation.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            TASKS_STARTED.labels(name).inc()
            start = time.time()
            logger.info(f"instrument start: {name}")
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                TASKS_FAILED.labels(name).inc()
                logger.exception(f"instrument error: {name}")
                raise
            finally:
                TASKS_TIME.labels(name).observe(time.time() - start)
        return wrapper
    return deco
