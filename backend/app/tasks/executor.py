# backend/app/tasks/executor.py
#   Implements the asynchronous workflow executor using Celery tasks.
#   - Maps workflow step types ("http_call", "model_call", "python_fn") to Celery tasks.
#   - Provides functions to:
#       • submit individual steps to Celery (`submit_step`)
#       • start a new workflow run from a DSL definition (`start_run_from_dsl`)
#       • resume a partially completed run (`resume_run`)
#       • execute a step synchronously with tracking (`execute_step`)
#   - Integrates with persistence, metrics/tracing, and compiler modules
#     to manage workflow state, step execution, and monitoring.
from backend.celery_app import celery_app
from backend.app.persistence import create_run, create_run_steps, update_step_start, update_step_finish, get_run_state
from backend.app.database import db as db_module
from backend.app import compiler
from backend.app.tasks.http_call import http_call_task
from backend.app.tasks.model_call import model_call_task
from backend.app.tasks.python_fn import python_fn
from backend.app.metrics_tracing_step import track_step, track_retry, track_compensation
from tasks.anomaly_detector_functions import detect_anomalies

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
        dag = compiler.compile_workflow_from_yaml(dsl_text)#from compiler.py
        step_defs = dag["nodes"]
        run = create_run(db, workflow_id, version, inputs_json=input_json)
        create_run_steps(db, run.id, step_defs)#from persistance.py
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
        fallbackStat = False
    except Exception:
        status = "failed"
        fallbackStat = True

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
# Compute duration
    if step.started_at and step.ended_at:
        duration = step.ended_at - step.started_at
        duration_ms = int(duration.total_seconds() * 1000)  # milliseconds
    else:
        duration_ms = None
        # task_payload = {
        #         "id": None,                    
        #         "run_id": run_id,                
        #         "step_id": step.id,               
        #         "type": step.type,    
        #         "status": step.status,           
        #         "attempt": step.attempt,                 
        #         "started_at": step.started_at,            
        #         "ended_at": step.ended_at,              
        #         "input_json": step.input_json,  
        #         "output_json": step.output_json,           
        #         "error_json": step.error_json,            
        #         "duration_ms": duration_ms,           
        #         "fallback_triggered": fallbackStat,   
        #         "worker_heartbeat": '2025-09-18 05:29:18.881673+05:30', #FIXME: For now     
        #         "error_code": 'S000', #FIXME: For now           
        #         }   
        # #flowGate OS
        # detect_anomalies.delay(task_payload)
    return result
