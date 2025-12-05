from urllib.parse import urlparse
from flask import Blueprint, json, request, jsonify
from flask_jwt_extended import jwt_required
from backend.app.Decorators.ScopesRequirements import require_scopes
from backend.app.TaskPulseOS.CommandHandlers.HandleUnhealthySteps import HandleUnhealthySteps
from backend.app.TaskPulseOS.CommandHandlers.SaveHeartbeatCmdHnd import SaveHeartbeatCmdHnd
from backend.app.TaskPulseOS.Commands.HandleUnhealthyStepsCmd import HandleUnhealthyStepsCmd
from backend.app.TaskPulseOS.Commands.SaveHeartbeatCmd import SaveHeartbeatCmd
from backend.app.TaskPulseOS.QuerryHandlers.CheckWorkerHealth import CheckWorkerHealth
from backend.app.TaskPulseOS.QuerryHandlers.SelectLatestHB import SelectLatestHB
from backend.app.services.idempotency_service import compute_request_hash, read_idempotency, write_idempotency
from backend.app.utils.rate_limit import allow
from flask import current_app as app
from backend.app import policies_cache

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
@jwt_required()
@require_scopes(["write:hooks"]) #scope validation
def pulse_hook_heartbeat():
    """
    Auth header:
    Authorization: Bearer <your-jwt-token>
    Idempotency-Key: 8b4d1d0e-52f2-4b4c-8132-18e453d9e8fc
    Content-Type: application/json
    
    Request Body:
    {
    "worker_id": "worker-123",
    "kind": "scheduler",
    "last_seen_at": "2025-02-12T15:30:45Z",
    "meta_json": {
        "version": "1.4.2",
        "ip": "192.168.10.12",
        "status": "healthy"
    }
    }
    """
    #rate limitting
    #TODO: Check
    url = "http://localhost:8000/pulse/hooks/heartbeat"
    parsed = urlparse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return {"status": None, "data": None, "error": f"Invalid URL, host missing: {url}"}

    if not allow(host):
        return {"status": None, "data": None, "error": f"Rate limit exceeded for host: {host}"}
    #data validation
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400
    required_sections = ["worker_id", "kind", "last_seen_at", "meta_json"]
    missing = [f for f in required_sections if f not in data]
    if missing:
        return jsonify({"error": "Missing fields","Missing" : missing}),400
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
    #get the latest heartbeat
    latest_db = SelectLatestHB()
    if latest_db["Error"]:
        return jsonify({"error": latest_db["Message"]}),500
    #store the heartbeat
    command = SaveHeartbeatCmd(worker_id=data["worker_id"],
                               kind=data["kind"],
                               last_seen_at=data["last_seen_at"],
                               meta_json=data["meta_json"])#command creation
    result = SaveHeartbeatCmdHnd(command)
    if result["Error"]:
        return jsonify({"error": result["Message"]}),500
    #get the heartbeat policy
    config_json = policies_cache.config_json
    latest = latest_db["Message"]
    created = result["Message"]
    heatlthStatus = CheckWorkerHealth(config_json, latest, created)
    if heatlthStatus["Status"] == "Healthy":
        return jsonify({"Status": "Success", "Message": "Heartbeat saved and worker is healthy"}),200
    elif heatlthStatus["Status"] == "Unhealthy":
        #TODO: Handle unhealthy worker steps
        unhealthyCmd = HandleUnhealthyStepsCmd(worker_id=created.worker_id)#command creation
        statusUnhealthySteps = HandleUnhealthySteps(unhealthyCmd)
        return jsonify({"Status": "Warning", "Message": "Heartbeat saved but worker is unhealthy"}),200
    else:
        return jsonify({"error": heatlthStatus["Message"]}),500         

# --------------------------------
# GET Endpoints
# --------------------------------

@TaskPulse_bp.route("/pulse/runs", methods=["GET"])
@jwt_required()
@require_scopes(["runs:read"]) #scope validation
def get_runs():
    #rate limitting
    #TODO: Check
    url = "http://localhost:8000/pulse/runs"
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

