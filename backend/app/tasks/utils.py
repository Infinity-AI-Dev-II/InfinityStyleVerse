# backend/app/tasks/utils.py
def safe_retry(task, exc):
    """Retry a Celery task safely in tests (prevents NoneType)."""
    if getattr(task.request, "called_directly", False):
        # In tests using .run(), just return structured error
        return {"status": None, "data": None, "error": str(exc)}
    return task.retry(exc=exc)
