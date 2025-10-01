# backend/app/tasks/executor.py
from backend.celery_app import celery_app
from backend.app.persistence import create_run, create_run_steps, update_step_start, update_step_finish, get_run_state
from backend.app.database import db as db_module
from backend.app import compiler
from backend.app.tasks.http_call import http_call_task
from backend.app.tasks.model_call import model_call_task
from backend.app.tasks.python_fn import python_fn
from backend.app.metrics_tracing_step import track_step, track_retry, track_compensation

STEP_TASK_MAP = {
    "http_call": http_call_task,
    "model_call": model_call_task,
    "python_fn": python_fn
}

def submit_step(step_def: dict, run_id: int):
    task_type = step_def.get("type")
    task = STEP_TASK_MAP.get(task_type)
    if not task:
        raise ValueError(f"No task registered for {task_type}")
    return task.apply_async(kwargs={**step_def.get("args", {}), "run_id": run_id, "step_id": step_def["id"]})

def start_run_from_dsl(dsl_text: str, workflow_id: int, version: str, input_json: str | None = None):
    db = db_module.session()
    try:
        dag = compiler.compile_workflow_from_yaml(dsl_text)
        step_defs = dag["nodes"]
        run = create_run(db, workflow_id, version, inputs_json=input_json)
        create_run_steps(db, run.id, step_defs)
        # enqueue entry steps (no incoming edges)
        entry_nodes = [n for n in step_defs if not n.get("incoming")]
        for n in entry_nodes:
            submit_step(n, run.id)
        return {"run_id": run.id, "dag": dag}
    finally:
        db.close()

def resume_run(run_id: int):
    db = db_module.session()
    try:
        state = get_run_state(db, run_id)
        if not state:
            return {"resumed": 0}
        pending_steps = [s for s in state["steps"] if s.status in ["pending", "running"]]
        for s in pending_steps:
            step_def = compiler.lookup_step(s.step_id)
            submit_step(step_def, run_id)
        return {"resumed": len(pending_steps)}
    finally:
        db.close()


def execute_step(step, workflow_name, run_id):
    step_id = step.id

    def step_logic():
        if step.payload.get("should_fail"):
            raise Exception("Step failure simulation")
        return "ok"

    # Execute the step
    status = "pending"
    try:
        result = track_step(workflow_name, run_id, step_id, step_logic)
        status = "completed"
    except Exception:
        status = "failed"

    # Create RunStep entry
    from backend.app.models import RunStep
    from backend.app.database import db

    run_step = RunStep(
        run_id=run_id,
        step_id=step_id,
        status="completed"
        )
    db.session.add(run_step)
    db.session.commit()   
    return result
