# backend/app/tasks/fan_in_out.py

from backend.celery_app import celery_app
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

@celery_app.task(
    bind=True,
    name="backend.app.tasks.fan_in_out.map_runner",
    time_limit=60,
    soft_time_limit=50,
    queue="cpu",
)
def map_runner(self, user_func, items, context=None):
    results = []
    context = context or {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(user_func, item, **context): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                output = future.result()
                results.append({"item": item, "status": "completed", "output": output})
            except Exception as e:
                logger.error(f"Map runner failed for item {item}: {e}")
                results.append({"item": item, "status": "failed", "output": None, "error": str(e)})

    return results

@celery_app.task(
    bind=True,
    name="backend.app.tasks.fan_in_out.join_runner",
    time_limit=60,
    soft_time_limit=50,
    queue="cpu",
)
def join_runner(self, step_outputs):
    combined_output = []
    final_status = "completed"

    for step in step_outputs:
        if step.get("status") == "failed":
            final_status = "failed"
        if "output" in step and step["output"] is not None:
            combined_output.append(step["output"])

    return {"status": final_status, "output": combined_output}
