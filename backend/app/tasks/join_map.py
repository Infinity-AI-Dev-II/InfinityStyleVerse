# backend/app/tasks/join_map.py
from backend.celery_app import celery_app
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="backend.app.tasks.join_map.join_task")
def join_task(self, *inputs):
    """Aggregate multiple inputs into a list"""
    try:
        result = [i for i in inputs if i is not None]
        return {"success": True, "output": result, "error": None}
    except Exception as e:
        logger.error(f"Join task failed: {e}")
        return {"success": False, "output": None, "error": str(e)}

@celery_app.task(bind=True, name="backend.app.tasks.join_map.map_task")
def map_task(self, func_code: str, items: list):
    """Apply a user-defined function to a list of items"""
    local_scope = {}
    try:
        exec(func_code, {}, local_scope)
        user_func = local_scope.get("user_func")
        if not user_func:
            raise ValueError("No function named 'user_func' found")
        result = [user_func(item) for item in items]
        return {"success": True, "output": result, "error": None}
    except Exception as e:
        logger.error(f"Map task failed: {e}")
        return {"success": False, "output": None, "error": str(e)}
