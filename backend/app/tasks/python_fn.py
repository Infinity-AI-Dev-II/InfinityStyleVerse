# tasks/python_fn.py
#   Provides a Celery task to safely execute user-defined Python functions.
#   - Uses a restricted set of built-in functions (`SAFE_BUILTINS`) for security.
#   - Supports function arguments, keyword arguments, and step context injection.
#   - Captures printed output and handles timeouts or exceptions.
#   - Returns structured results with success status, output text, result, and error info.
#   - Designed for use in workflow steps that require dynamic Python function execution.

from backend.celery_app import celery_app
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import io
import sys

logger = logging.getLogger(__name__)

SAFE_BUILTINS = {
    "range": range,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "print": print,  
    "__import__": __import__, 
}


def safe_exec(user_func, args=None, kwargs=None, timeout=5):
    args = args or []
    kwargs = kwargs or {}

    # Capture stdout (print)
    stdout_capture = io.StringIO()
    with ThreadPoolExecutor(max_workers=1) as executor:
        original_stdout = sys.stdout
        sys.stdout = stdout_capture
        try:
            future = executor.submit(user_func, *args, **kwargs)
            result = future.result(timeout=timeout)
            output_text = stdout_capture.getvalue()
            return {"success": True, "result": result, "output": output_text, "error": None}
        except TimeoutError:
            logger.error("Function execution timed out")
            return {"success": False, "result": None, "output": stdout_capture.getvalue(), "error": "Timeout"}
        except Exception:
            error_msg = traceback.format_exc()
            logger.error(f"Function execution error: {error_msg}")
            return {"success": False, "result": None, "output": stdout_capture.getvalue(), "error": error_msg}
        finally:
            sys.stdout = original_stdout


@celery_app.task(
    bind=True,
    name="backend.app.tasks.python_fn.python_fn",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=2,
    time_limit=30,
    soft_time_limit=25,
    queue="cpu",
)
def python_fn(self, func_code: str, func_args=None, func_kwargs=None, step_context=None):
    func_args = func_args or []
    func_kwargs = func_kwargs or {}
    local_scope = {}

    if step_context:
        func_kwargs["context"] = step_context

    try:
        exec(func_code, {"__builtins__": SAFE_BUILTINS}, local_scope)
        user_func = local_scope.get("user_func")
        if not user_func:
            return {"success": False, "result": None, "output": None, "error": "No function named 'user_func' found"}

        return safe_exec(user_func, func_args, func_kwargs, timeout=5)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"Error parsing function: {error_msg}")
        return {"success": False, "result": None, "output": None, "error": error_msg}
