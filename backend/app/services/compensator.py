# services/compensator.py
#   Implements compensation logic for failed workflow steps.
#   In workflow orchestration, "compensation" is the undo or rollback
#   action that should be executed when a step fails.
import json
from ..database import db
from ..models import Compensation, RunStep, Run
from flask import current_app

def handle_failure(run, failed_step, dag):
    """
    When a step fails:
      1. Look up compensation defined in the step metadata (from dag['nodes'][step_id]['compensation'])
      2. Persist a Compensation row (status=created) and attempt to execute it synchronously
      3. Mark compensation status and update Run status accordingly
    """
    sid = failed_step.step_id
    nodes = dag.get("nodes", {})
    meta = nodes.get(sid, {}) if nodes else {}
    comp_action = meta.get("compensation")

    comp = Compensation(run_id=run.id, step_id=sid, status="created", payload_json=json.dumps({"compensation": comp_action}))
    db.session.add(comp)
    db.session.commit()

    if not comp_action:
        current_app.logger.warning("no compensation action defined for step %s (run %s)", sid, run.id)
        return comp

    current_app.logger.info("running compensation for step %s (action=%s)", sid, comp_action)
    try:
        ok = execute_compensation(comp_action, run, failed_step)
        comp.status = "done" if ok else "failed"
        db.session.commit()
        return comp
    except Exception:
        current_app.logger.exception("compensation execution failed")
        comp.status = "failed"
        db.session.commit()
        return comp


def execute_compensation(action_name, run, failed_step):
    """
    Placeholder: execute the compensation action.
    Return True on success, False/raise on error.

    Common compensation examples:
      - refund_card
      - cancel_reservation
      - release_inventory
    """
    current_app.logger.info("executing compensation %s for run %s step %s", action_name, run.id, failed_step.step_id)
    # Simple simulation: if action == "fail_comp" then throw to test behavior.
    if action_name == "fail_comp":
        raise RuntimeError("simulated compensation failure")

    # Real implementation would call services, APIs, or queue background tasks.
    # For now, return success.
    return True
