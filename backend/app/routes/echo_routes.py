from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required
from flasgger import swag_from
from ..models.echo_models import Decision, Event, Reward, Policy, Experiment
from ..services.kafka_service import publish_kafka
from ..services.idempotency_service import read_idempotency, write_idempotency, compute_request_hash
from ..utils.request_log import record_request_log
from ..database import db
from datetime import datetime
import time, uuid, json

echo_bp = Blueprint("echo", __name__, url_prefix="/echo")

@echo_bp.before_app_request
def _set_start_time_and_requestid():
    g.start_time = time.time()
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

# -------------------------------
# /echo/hello endpoint
# -------------------------------
@echo_bp.route("/hello", methods=["GET"])
@swag_from({
    "tags": ["Echo"],
    "responses": {
        200: {
            "description": "Returns a hello message",
            "content": {
                "application/json": {
                    "example": {"message": "Hello from InfinityStyleVerse!"}
                }
            }
        }
    }
})
def hello():
    return jsonify({"message": "Hello from InfinityStyleVerse!"})

# -------------------------------
# /echo/decision endpoint
# -------------------------------
@echo_bp.route("/decision", methods=["POST"])
@jwt_required(optional=True)
@swag_from({
    "tags": ["Echo"],
    "parameters": [
        {
            "name": "Idempotency-Key",
            "in": "header",
            "type": "string",
            "required": False,
            "description": "Optional key to ensure idempotent requests"
        },
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "policy_id": {"type": "string"},
                    "task": {"type": "string"},
                    "candidates": {"type": "object"},
                    "chosen": {"type": "object"},
                    "context": {"type": "object"},
                    "ab_bucket": {"type": "string"},
                    "scores": {"type": "object"}
                },
                "required": ["request_id", "policy_id", "task", "candidates", "chosen"]
            }
        }
    ],
    "responses": {
        201: {"description": "Decision stored successfully"},
        400: {"description": "Missing required fields"},
        500: {"description": "Server error"}
    }
})
def post_decision():
    idem_key = request.headers.get("Idempotency-Key")
    body = request.get_json(force=True, silent=True) or {}

    required = ["request_id", "policy_id", "task", "candidates", "chosen"]
    if not all(k in body for k in required):
        return jsonify({"ok": False, "error": "missing required fields"}), 400

    if idem_key:
        existing = read_idempotency(idem_key)
        if existing:
            try:
                resp = json.loads(existing.response_json)
            except:
                resp = {"ok": True, "note":"idempotent replay"}
            return jsonify(resp), 200

    try:
        d = Decision(
            request_id=body["request_id"],
            policy_id=body["policy_id"],
            task=body["task"],
            tenant=body.get("tenant"),
            ab_bucket=body.get("ab_bucket"),
            context_json=body.get("context"),
            candidates_json=body.get("candidates"),
            chosen_json=body.get("chosen"),
            scores_json=body.get("scores")
        )
        db.session.add(d)
        db.session.commit()
        publish_kafka("echo.decisions", {"request_id": d.request_id, "task": d.task})
        resp = {"ok": True, "request_id": d.request_id}
        if idem_key:
            write_idempotency(idem_key, compute_request_hash(body), resp)
        record_request_log(201)
        return jsonify(resp), 201
    except Exception:
        db.session.rollback()
        record_request_log(500)
        return jsonify({"ok": False, "error": "server error"}), 500
