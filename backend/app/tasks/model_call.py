# backend/app/tasks/model_call.py
#   Provides Celery tasks for basic fan-in/fan-out workflow patterns.
#   - `join_task`: aggregates multiple inputs into a single list, handling None values.
#   - `map_task`: executes a user-defined function (`user_func`) on each item in a list.
#   - Supports parallel workflow execution and combining results, with error handling and logging.

from backend.celery_app import celery_app
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from backend.app.tasks.utils import safe_retry
from typing import Optional

logger = logging.getLogger(__name__)

@celery_app.task(
    bind=True,
    name="backend.app.tasks.model_call.model_call_task",
    autoretry_for=(requests.RequestException,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
    time_limit=30,
    soft_time_limit=25,
    queue="io",
)
def model_call_task(self, model_url: str, payload: dict, headers: Optional[dict] = None):
    """
    Celery task to call an external ML model endpoint with retries.
    Always returns a consistent dict with 'success', 'output', and 'error'.
    Validates output is a dictionary.
    """
    headers = headers or {}

    try:
        with requests.Session() as session:
            retries = Retry(
                total=3,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            logger.info(f"Sending request to {model_url} with payload: {payload}")
            response = session.post(model_url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()

            try:
                data = response.json()
            except ValueError:
                logger.error(f"Response is not JSON: {response.text}")
                data = response.text

            # Validate output is a dictionary
            if not isinstance(data, dict):
                error_msg = f"Invalid model output, expected dict but got {type(data).__name__}"
                logger.error(error_msg)
                return {"success": False, "output": None, "error": error_msg}

            return {"success": True, "output": data, "error": None}

    except requests.RequestException as e:
        logger.error(f"Model call failed: {e}")
        # Optionally retry via Celery
        # safe_retry(self, e)

        return {"success": False, "output": None, "error": str(e)}

    except Exception as e:
        # Catch-all to ensure workflow doesn't crash on unexpected errors
        logger.error(f"Unexpected error during model call: {e}", exc_info=True)
        return {"success": False, "output": None, "error": str(e)}
