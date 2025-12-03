from urllib.parse import urlparse
from flask import Blueprint, json, request, jsonify
from flask_jwt_extended import jwt_required
from backend.app.Decorators.ScopesRequirements import require_scopes
from backend.app.services.idempotency_service import compute_request_hash, read_idempotency, write_idempotency
from backend.app.utils.rate_limit import allow

TaskPulse_bp = Blueprint("TaskPulse_bp", __name__)

# --------------------------------
# POST Endpoints
# --------------------------------

@TaskPulse_bp.route("/pulse/hooks/run", methods=["POST"])
def pulse_hook_run():
    payload = request.json
    return jsonify({
        "status": "ok",
        "event": "run",
        "received": payload
    }), 200


@TaskPulse_bp.route("/pulse/hooks/step", methods=["POST"])
def pulse_hook_step():
    payload = request.json
    return jsonify({
        "status": "ok",
        "event": "step",
        "received": payload
    }), 200


@TaskPulse_bp.route("/pulse/hooks/heartbeat", methods=["POST"])
def pulse_hook_heartbeat():
    payload = request.json
    return jsonify({
        "status": "ok",
        "event": "heartbeat",
        "received": payload
    }), 200


# --------------------------------
# GET Endpoints
# --------------------------------

@TaskPulse_bp.route("/pulse/runs", methods=["GET"])
@jwt_required()
@require_scopes(["runs:read"]) #scope validation
def get_runs():
    #rate limitting
    #TODO: Check
    url = "http://localhost:8000//pulse/runs"
    parsed = urlparse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return {"status": None, "data": None, "error": f"Invalid URL, host missing: {url}"}

    if not allow(host):
        return {"status": None, "data": None, "error": f"Rate limit exceeded for host: {host}"}
    #obtaining and validating value parameters
    tenant = request.args.get("tenant")
    status = request.args.get("status")
    q = request.args.get("q")
    page = request.args.get("page")
    if tenant is None or status is None or q is None or page is None:
        return jsonify({"error" : "Values are required"}), 400 
    #Idempotency-Key checking
    idem_key = request.headers.get("Idempotency-Key")
    if not idem_key:
        return jsonify({"error" : "Idempotency key is required"})
    else:
        status = read_idempotency(idem_key)
        if status:
            try:
                response = json.loads(status.response_json)
            except:
                response = {"Error" : "Error occured"}
            return response    
    return jsonify({"runs": []}), 200


@TaskPulse_bp.route("/pulse/runs/<string:run_id>", methods=["GET"])
@jwt_required()
def get_run(run_id):
    return jsonify({
        "run_id": run_id,
        "data": {}
    }), 200


@TaskPulse_bp.route("/pulse/runs/<string:run_id>/steps", methods=["GET"])
@jwt_required()
def get_run_steps(run_id):
    return jsonify({
        "run_id": run_id,
        "steps": []
    }), 200


@TaskPulse_bp.route("/pulse/alerts", methods=["GET"])
@jwt_required()
def get_alerts():
    return jsonify({"alerts": []}), 200


@TaskPulse_bp.route("/pulse/workers", methods=["GET"])
@jwt_required()
def get_workers():
    return jsonify({"workers": []}), 200

