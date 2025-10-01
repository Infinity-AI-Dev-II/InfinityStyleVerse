# services/signals.py
import json
from ..database import db
from ..models import Run, RunStep
from flask import current_app

def handle_signal(run_id, signal_name, payload=None):
    """
    Find steps in the given run that are waiting for this signal and resume them.
    Returns True if a waiting step was found and resumed, otherwise False.

    This implementation:
    - searches for RunSteps with status == "waiting_for_signal" and payload_json contains an entry "signal": "<name>"
      OR
    - if no explicit 'signal' found, resumes the first waiting step found.
    """
    run = Run.query.get(run_id)
    if not run:
        raise ValueError("run not found")

    waiting_steps = RunStep.query.filter_by(run_id=run.id, status="waiting_for_signal").all()
    if not waiting_steps:
        current_app.logger.info("no waiting_for_signal steps for run %s", run_id)
        return False

    # prefer step which explicitly declared the signal name in its payload
    target = None
    for s in waiting_steps:
        meta = json.loads(s.payload_json or "{}")
        expected_signal = meta.get("signal")
        if expected_signal == signal_name:
            target = s
            break

    if not target:
        # fallback: pick first waiting step
        target = waiting_steps[0]

    # attach payload to the step or run var
    try:
        # update step payload_json with signal payload
        meta = json.loads(target.payload_json or "{}")
        meta.setdefault("signals", []).append({"name": signal_name, "payload": payload})
        target.payload_json = json.dumps(meta)
        # set to pending so executor can pick it up
        target.status = "pending"
        db.session.commit()
        current_app.logger.info("resumed step %s for run %s by signal %s", target.step_id, run_id, signal_name)
        return True
    except Exception:
        current_app.logger.exception("failed to handle signal")
        db.session.rollback()
        raise
