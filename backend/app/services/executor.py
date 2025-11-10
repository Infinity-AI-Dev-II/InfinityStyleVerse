# services/executor.py
# Purpose:
#   Provides the core workflow execution engine.
import json
from ..database import db
from ..models import Run, RunStep
from ..services import compensator
from flask import current_app

def start_run_from_dag(dag_steps, workflow_id=None, version="1.0", created_by=None):
    if workflow_id is None:
        raise ValueError("workflow_id must be provided")

    # Ensure steps_list is always a list of step dicts
    if isinstance(dag_steps, dict) and "steps" in dag_steps:
        steps_list = dag_steps["steps"]
    elif isinstance(dag_steps, list):
        steps_list = dag_steps
    else:
        raise ValueError(f"Invalid dag_steps format: {dag_steps}")

    # Build DAG
    nodes = {step["step_id"]: {"action": step.get("action")} for step in steps_list}
    edges = [
        (dep, step["step_id"]) 
        for step in steps_list 
        for dep in step.get("depends_on", [])
    ]


    dag = {"nodes": nodes, "edges": edges}

    run = Run(
        workflow_id=workflow_id,
        version=version,
        status="pending",
        caller=created_by,
        inputs_json=json.dumps(dag)
    )
    db.session.add(run)
    db.session.commit()

    # create RunStep objects
    for step_id in nodes:
        step = RunStep(
            run_id=run.id,
            step_id=step_id,
            status="pending"
        )
        db.session.add(step)
    db.session.commit()

    # refresh the run object so relationships are loaded
    db.session.refresh(run)

    return run


def run_workflow(run_id):
    run = Run.query.get(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")

    # If already terminal or running, just log and continue
    if run.status in ("completed", "failed", "cancelled", "running"):
        current_app.logger.info("Run already in terminal or running state: %s", run.status)

    run.status = "running"
    db.session.commit()
    db.session.refresh(run)  # ensure relationships and latest data are loaded

    dag = json.loads(run.inputs_json or "{}")
    nodes = dag.get("nodes", {})
    edges = dag.get("edges", [])

    # Build predecessor map
    preds = {n: set() for n in nodes}
    for a, b in edges:
        preds[b].add(a)

    def get_step_model(step_id):
        return RunStep.query.filter_by(run_id=run.id, step_id=step_id).first()

    try:
        while True:
            pending_steps = RunStep.query.filter_by(run_id=run.id, status="pending").all()
            if not pending_steps:
                break

            progressed = False
            for step in pending_steps:
                sid = step.step_id
                step_meta = nodes.get(sid, {})
                pred_done = all(get_step_model(p).status == "done" for p in preds.get(sid, []))

                if not pred_done:
                    continue

                progressed = True
                step.status = "running"
                db.session.commit()
                db.session.refresh(step)

                try:
                    ok = execute_action(sid, step_meta, run, step)
                    if ok:
                        step.status = "done"
                        db.session.commit()
                        db.session.refresh(step)
                except Exception:
                    current_app.logger.exception("Step execution failed for %s", sid)
                    step.status = "failed"
                    db.session.commit()
                    run.status = "failed"
                    db.session.commit()
                    db.session.refresh(run)
                    compensator.handle_failure(run, step, dag)
                    return run

            if not progressed:
                # No steps could progress → deadlock
                current_app.logger.error("Deadlock detected for run %s - marking as failed", run.id)
                run.status = "failed"
                db.session.commit()
                db.session.refresh(run)
                return run

        # If any steps are waiting for signal, mark run accordingly
        waiting_steps = RunStep.query.filter_by(run_id=run.id, status="waiting_for_signal").count()
        if waiting_steps > 0:
            run.status = "waiting_for_signal"
            db.session.commit()
            db.session.refresh(run)
            return run

        # All steps done
        run.status = "completed"
        db.session.commit()
        db.session.refresh(run)
        return run

    except Exception:
        current_app.logger.exception("Unexpected executor error")
        run.status = "failed"
        db.session.commit()
        db.session.refresh(run)
        return run



def execute_action(step_id, step_meta, run, step_model):
    action = step_meta.get("action")
    current_app.logger.info("Executing step %s action=%s", step_id, action)

    if action == "fail":
        raise RuntimeError("simulated failure for testing")
    if action == "wait_for_signal":
        step_model.status = "waiting_for_signal"
        db.session.commit()          # <-- commit here!
        db.session.refresh(step_model)  # ensure latest data
        run.status = "waiting_for_signal"
        db.session.commit()
        db.session.refresh(run)
        return False

    # default success
    return True