# workflow.py  (project root)
#   Script to run a workflow defined in a YAML file using Celery tasks.
import yaml
from backend.app.tasks import http_call, python_fn
from celery.exceptions import SoftTimeLimitExceeded

def run_workflow_from_yaml(yaml_file):
    with open(yaml_file) as f:
        workflow = yaml.safe_load(f)

    results = {}
    for step in workflow.get("steps", []):
        task_type = step["type"]
        task = {"http_call": http_call, "python_fn": python_fn}.get(task_type)
        if not task:
            results[step["id"]] = {"error": f"Task type '{task_type}' not found"}
            continue

        args = step.get("args", [])
        kwargs = step.get("kwargs", {})
        if "fn" in step:
            kwargs["fn"] = step["fn"]

        print(f"Queuing {step['id']} ({task_type})...")
        async_result = task.delay(*args, **kwargs)

        try:
            task_result = async_result.get(timeout=120)
        except SoftTimeLimitExceeded:
            task_result = {"error": "Task exceeded soft time limit"}
        except Exception as e:
            task_result = {"error": str(e)}

        print(f"Result of {step['id']}: {task_result}")
        results[step["id"]] = task_result

    return results

if __name__ == "__main__":
    print("Running workflow...")
    out = run_workflow_from_yaml("design_to_shop.yaml")
    print("Finished:", out)
