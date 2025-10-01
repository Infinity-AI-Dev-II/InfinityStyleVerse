# routes/flow.py
from flask import Blueprint, json, request, jsonify, current_app
from backend.app.models import Run, RunStep, Compensation
from ..database import db
from ..services import planner, executor, signals

flow_bp = Blueprint("flow", __name__)

@flow_bp.route("/plan", methods=["POST"])
def plan_flow():
    """
    Dry-run: compile DSL -> DAG, do not persist Run or RunStep
    Expects JSON:
      {
        "dsl": "<string or object>"   # YAML/JSON string or parsed object
      }
    Returns DAG structure.
    """
    payload = request.get_json(force=True, silent=True) or {}
    dsl = payload.get("dsl")
    if dsl is None:
        return jsonify({"error": "missing 'dsl' in body"}), 400

    try:
        dag = planner.parse_dsl(dsl)
        return jsonify({"status": "planned", "dag": dag}), 200
    except Exception as e:
        current_app.logger.exception("planner.parse_dsl failed")
        return jsonify({"error": "failed to parse DSL", "detail": str(e)}), 400

@flow_bp.route("/run/start", methods=["POST"])
def start_run():
    """
    Helper endpoint: create Run and start executing synchronously.
    Expects:
      {
        "workflow_id": "<required>",
        "dsl": "<dsl>",
        "created_by": "<optional user id/email>",
        "version": "<optional version>"
      }
    Returns Run id and status.
    """
    payload = request.get_json(force=True, silent=True) or {}
    dsl = payload.get("dsl")
    created_by = payload.get("created_by")
    workflow_id = payload.get("workflow_id")
    version = payload.get("version", "1.0")

    # --- Validation ---
    if not workflow_id:
        return jsonify({"error": "missing 'workflow_id' in body"}), 400
    if dsl is None:
        return jsonify({"error": "missing 'dsl' in body"}), 400

    try:
        dag = planner.parse_dsl(dsl)
    except Exception as e:
        current_app.logger.exception("planner.parse_dsl failed")
        return jsonify({"error": "failed to parse DSL", "detail": str(e)}), 400

    try:
        from backend.app.services.executor import start_run_from_dag, run_workflow

        run = start_run_from_dag(
            dag,
            workflow_id=workflow_id,
            version=version,
            created_by=created_by
        )

        run_workflow(run.id)

        return jsonify({"run_id": run.id, "status": run.status}), 200

    except Exception as e:
        current_app.logger.exception("start_run failed")
        return jsonify({
            "error": "failed to start run",
            "detail": str(e)
        }), 500

@flow_bp.route("/run/<int:run_id>/cancel", methods=["POST"])
def cancel_run(run_id):
    run = Run.query.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404

    if run.status in ("completed", "failed", "cancelled"):
        return jsonify({"run_id": run.id, "status": run.status}), 200

    run.status = "cancelled"
    RunStep.query.filter(
        RunStep.run_id == run.id,
        RunStep.status.in_(["pending", "running", "waiting_for_signal"])
    ).update({"status": "cancelled"}, synchronize_session="fetch")
    db.session.commit()

    return jsonify({"run_id": run.id, "status": run.status}), 200


    
@flow_bp.route("/run/signal/<name>", methods=["POST"])
def signal_run(name):
    """
    Inject signal `name` into a run. Body must contain run_id and optional payload.
      POST /flow/run/signal/approve
      { "run_id": 123, "payload": {"approved_by": "alice"} }
    """
    payload = request.get_json(force=True, silent=True) or {}
    run_id = payload.get("run_id")
    signal_payload = payload.get("payload", {})

    if not run_id:
        return jsonify({"error": "missing run_id"}), 400

    run = Run.query.get(run_id)
    if not run:
        return jsonify({"error": "run not found"}), 404

    try:
        handled = signals.handle_signal(run_id=run.id, signal_name=name, payload=signal_payload)
        if handled:
            return jsonify({"run_id": run.id, "signal": name, "handled": True}), 200
        else:
            return jsonify({"run_id": run.id, "signal": name, "handled": False, "detail": "no waiting step for signal"}), 200
    except Exception as e:
        current_app.logger.exception("signals.handle_signal failed")
        return jsonify({"error": "signal handling failed", "detail": str(e)}), 500


@flow_bp.route("/flow/run/<int:run_id>", methods=["GET"])
def get_run(run_id):
    run = Run.query.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404

    # Always return a list, even if no steps
    steps = RunStep.query.filter_by(run_id=run.id).all() or []

    step_list = [
        {
            "id": getattr(s, "step_id", None),
            "status": getattr(s, "status", "pending"),
            "output": getattr(s, "output_json", None),
        }
        for s in steps
    ]

    return jsonify({
        "run_id": run.id,
        "status": run.status or "pending",
        "steps": step_list
    })


@flow_bp.route("/workflow_defs", methods=["GET"])
def list_workflow_defs():
    from backend.app.models import WorkflowDef
    workflows = WorkflowDef.query.all()
    return jsonify([wf.to_dict() for wf in workflows]), 200
