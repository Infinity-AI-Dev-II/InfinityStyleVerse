import io
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import redis
import requests
from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import jwt_required
from typing import Any, Dict, Optional, Tuple
from flasgger import swag_from

from backend.app.AWS_configuration import AWSConfig
from backend.app.StyleSense.BodyMorph.engine import BodyMorphEngine
from backend.app.StyleSense.BodyMorph.exceptions import (
    BodyMorphError,
    ExtremePerspectiveError,
    HeavyOcclusionError,
    LowQualityError,
    MultiSubjectError,
    UnknownSilhouetteError,
)
from backend.app.Decorators.requestSizeValidator import validate_request_size
from backend.app.services.idempotency_service import compute_request_hash, read_idempotency, write_idempotency
from backend.app.services.redisKeyGenerate import generate_image_cache_key
from backend.app.tasks.GetImageBySignedUrl import load_image_from_signed_url

# ------------------------------------------------------------------------------
# Blueprint
# ------------------------------------------------------------------------------
bodyMorph_bp = Blueprint('bodyMorph_bp', __name__, url_prefix="/stylesense")

# ------------------------------------------------------------------------------
# Constants / Config
# ------------------------------------------------------------------------------

# Maximum file size (for frontend validation reference)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Allow a lightweight placeholder image when fetch fails (useful for dev/test).
# If disabled, fetch failures become client errors (400) or internal errors (500).
ALLOW_PLACEHOLDER_IMAGE = os.getenv("BODYMORPH_ALLOW_PLACEHOLDER", "true").lower() == "true"
PLACEHOLDER_IMAGE_BYTES = b"placeholder-bodymorph-image"

# Validation constants to keep inputs/output predictable
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_FORCE_FAILURES = {"MULTI_SUBJECT", "LOW_QUALITY", "HEAVY_OCCLUSION", "EXTREME_PERSPECTIVE", "UNKNOWN"}
MAX_IMAGE_URI_LENGTH = 4096
MAX_FOLDER_PATH_LENGTH = 128
FOLDER_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9/_-]+$")

# Hints constraints
MIN_HEIGHT_CM = 80
MAX_HEIGHT_CM = 260
MAX_SUBJECTS = 5

# Camera constraints
MIN_CAMERA_DISTANCE_M = 0.2
MAX_CAMERA_DISTANCE_M = 20.0
MIN_CAMERA_FOV = 1.0
MAX_CAMERA_FOV = 180.0
MAX_TILT_DEG = 90.0
try:
    OPTIONAL_VALIDATION_CONFIDENCE_PENALTY = float(os.getenv("BODYMORPH_OPTIONAL_PENALTY", "0.1"))
except ValueError:
    OPTIONAL_VALIDATION_CONFIDENCE_PENALTY = 0.1

# Initialize Boto3 S3 client
s3_client = AWSConfig.get_s3_client()
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Async / orchestration
BODYMORPH_ASYNC_ENABLED = os.getenv("BODYMORPH_ASYNC_ENABLED", "true").lower() == "true"
BODYMORPH_ASYNC_WORKERS = int(os.getenv("BODYMORPH_ASYNC_WORKERS", "4"))

# TaskPulseOS wiring (disabled by default; enable with BODYMORPH_TASKPULSE_ENABLED=true)
TASKPULSE_ENABLED = os.getenv("BODYMORPH_TASKPULSE_ENABLED", "false").lower() == "true"
TASKPULSE_WORKFLOW_NAME = os.getenv("BODYMORPH_TASKPULSE_WORKFLOW", "BodyMorph.v1")
TASKPULSE_STEP_ID = "body_profile"
TASKPULSE_STEP_IDS = {
    "workflow": "workflow",
    "validate_input": "validate_input",
    "load_image_reference": "load_image_reference",
    "detect_person": "detect_person",
    "segment_body": "segment_body",
    "estimate_pose": "estimate_pose",
    "postprocess_landmarks": "postprocess_landmarks",
    "normalize_scale": "normalize_scale",
    "estimate_proportions": "estimate_proportions",
    "classify_silhouette": "classify_silhouette",
    "posture_heuristics": "posture_heuristics",
    "assemble_response": "assemble_response",
    "cache_write": "cache_write",
    "telemetry_emit": "telemetry_emit",
}

# Map engine trace stages to TaskPulse step ids for telemetry
TRACE_STAGE_TO_STEP = {
    "validate_input": TASKPULSE_STEP_IDS["validate_input"],
    "load_image_reference": TASKPULSE_STEP_IDS["load_image_reference"],
    "detect_person": TASKPULSE_STEP_IDS["detect_person"],
    "segment_body": TASKPULSE_STEP_IDS["segment_body"],
    "pose_estimate": TASKPULSE_STEP_IDS["estimate_pose"],
    "postprocess_landmarks": TASKPULSE_STEP_IDS["postprocess_landmarks"],
    "normalize_scale": TASKPULSE_STEP_IDS["normalize_scale"],
    "estimate_proportions": TASKPULSE_STEP_IDS["estimate_proportions"],
    "classify_silhouette": TASKPULSE_STEP_IDS["classify_silhouette"],
    "posture_heuristics": TASKPULSE_STEP_IDS["posture_heuristics"],
    "assemble_response": TASKPULSE_STEP_IDS["assemble_response"],
    "cache_write": TASKPULSE_STEP_IDS["cache_write"],
    "telemetry_emit": TASKPULSE_STEP_IDS["telemetry_emit"],
}

# Fallback in-memory cache if Redis is unavailable (process-local; not shared across workers)
_LOCAL_CACHE = {}
_RUN_LEDGER = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=BODYMORPH_ASYNC_WORKERS) if BODYMORPH_ASYNC_ENABLED else None
RUN_LEDGER_PREFIX = "bodymorph:run:"
RUN_LEDGER_TTL = 3600


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _run_ledger_key(run_id: str) -> str:
    return f"{RUN_LEDGER_PREFIX}{run_id}"


def _write_run_state(run_id: str, state: dict):
    """
    Persist run state in Redis with in-memory fallback.
    """
    try:
        client = _get_redis_client()
        client.set(_run_ledger_key(run_id), json.dumps(state), ex=RUN_LEDGER_TTL)
        return
    except Exception as exc:
        try:
            current_app.logger.debug({
                "component": "BodyMorph",
                "event": "run_state_write_failed",
                "run_id": run_id,
                "error": str(exc),
            })
        except Exception:
            pass
        _RUN_LEDGER[run_id] = state


def _read_run_state(run_id: str) -> Optional[dict]:
    """
    Fetch run state from Redis or fallback cache.
    """
    try:
        client = _get_redis_client()
        raw = client.get(_run_ledger_key(run_id))
        if raw:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)
    except Exception:
        pass
    return _RUN_LEDGER.get(run_id)


def _init_run_state(run_id: str, mode: str, status: str = "queued", idem_key: Optional[str] = None, request_hash: Optional[str] = None) -> dict:
    now = _now_iso()
    state = {
        "run_id": run_id,
        "workflow": TASKPULSE_WORKFLOW_NAME,
        "mode": mode,
        "status": status,
        "idem_key": idem_key,
        "request_hash": request_hash,
        "started_at": now if status in ("running", "queued") else None,
        "heartbeat_at": now,
        "steps": {},
    }
    _write_run_state(run_id, state)
    return state


def _update_run_state(run_id: str, updates: dict) -> Optional[dict]:
    state = _read_run_state(run_id) or {"run_id": run_id, "steps": {}}
    state.update(updates or {})
    state["updated_at"] = _now_iso()
    _write_run_state(run_id, state)
    return state


def _record_step_status(run_id: str, step_id: str, status: str, meta: Optional[dict] = None):
    state = _read_run_state(run_id)
    if not state:
        return
    steps = state.setdefault("steps", {})
    step_state = steps.get(step_id, {"events": []})
    now = _now_iso()

    if status == "running" and not step_state.get("started_at"):
        step_state["started_at"] = now
    if status in ("success", "failure"):
        step_state["ended_at"] = now
        if meta and isinstance(meta, dict) and "latency_ms" in meta:
            step_state["latency_ms"] = meta["latency_ms"]
        elif step_state.get("started_at"):
            try:
                start_dt = datetime.fromisoformat(step_state["started_at"].rstrip("Z"))
                end_dt = datetime.fromisoformat(now.rstrip("Z"))
                step_state["latency_ms"] = int((end_dt - start_dt).total_seconds() * 1000)
            except Exception:
                pass
    step_state["status"] = status
    event = {"status": status, "at": now}
    if meta:
        event["meta"] = meta
    step_state.setdefault("events", []).append(event)
    steps[step_id] = step_state

    state["steps"] = steps
    state["heartbeat_at"] = now
    if status == "failure":
        state["status"] = "failed"
    elif status == "success" and state.get("status") not in ("failed", "succeeded"):
        state["status"] = state.get("status", "running")

    _write_run_state(run_id, state)


def _finalize_run_state(
    run_id: str,
    status: str,
    response_body: Optional[dict] = None,
    http_status: Optional[int] = None,
    trace: Optional[list] = None,
    degraded_reason: Optional[str] = None,
    cache_hit: bool = False,
):
    state = _read_run_state(run_id) or {"run_id": run_id, "steps": {}}
    now = _now_iso()
    state.update({
        "status": status,
        "completed_at": now,
        "heartbeat_at": now,
        "http_status": http_status,
        "cache_hit": cache_hit,
    })
    if degraded_reason:
        state["degraded_reason"] = degraded_reason
    if trace is not None:
        state["trace"] = trace
    if response_body is not None:
        state["response"] = response_body
    _write_run_state(run_id, state)


# ------------------------------------------------------------------------------
# S3 helpers
# ------------------------------------------------------------------------------
def generate_presigned_url(object_key, expiration=3600):
    """Generate a pre-signed URL to upload a file to S3."""
    try:
        return s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': object_key},
            ExpiresIn=expiration
        )
    # except (NoCredentialsError, PartialCredentialsError) as e:
    #         return f"Credentials error: {str(e)}"
    except Exception as e:
        print(f"Error generating pre-signed URL: {e}")
        return None


# ------------------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------------------
def _validate_force_failure(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize force_failure knob used by the mock engine and verify it is allowed.

    Returns:
      (normalized_value, error_message)
        - normalized_value is uppercased or None
        - error_message is None when valid, otherwise a human-friendly message
    """
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "force_failure must be a string."

    normalized = value.upper()
    if normalized not in ALLOWED_FORCE_FAILURES:
        allowed = ", ".join(sorted(ALLOWED_FORCE_FAILURES))
        return None, f"force_failure must be one of: {allowed}"
    return normalized, None


def _validate_hints(raw_hints: Any, request_id: str) -> Tuple[Dict[str, Any], Optional[Tuple], list]:
    """
    Validate optional hints passed to the mock engine.

    Args:
      raw_hints: payload["hints"] (expected dict-like)
      request_id: correlation id for error responses

    Returns:
      (normalized_hints, error_response_tuple, warnings)
        - normalized_hints: dict (may be empty)
        - error_response_tuple: (jsonify(...), status) or None
        - warnings: list of soft failures (used for degraded path)
    """
    warnings = []
    if not raw_hints:
        return {}, None, warnings
    if not isinstance(raw_hints, dict):
        warnings.append({"field": "hints", "reason": "invalid_type"})
        return {}, None, warnings

    hints = dict(raw_hints)

    # Validate height
    height_cm = hints.get("height_cm")
    if height_cm is not None:
        if not isinstance(height_cm, (int, float)) or not (MIN_HEIGHT_CM <= float(height_cm) <= MAX_HEIGHT_CM):
            warnings.append({"field": "hints.height_cm", "reason": "out_of_range"})
            hints.pop("height_cm", None)
        else:
            hints["height_cm"] = float(height_cm)

    # Validate subject_count
    subject_count = hints.get("subject_count")
    if subject_count is not None:
        if not isinstance(subject_count, int) or not (1 <= subject_count <= MAX_SUBJECTS):
            warnings.append({"field": "hints.subject_count", "reason": "invalid_subject_count"})
            hints.pop("subject_count", None)

    # Validate force_failure (dev/testing knob)
    normalized_force_failure, force_err = _validate_force_failure(hints.get("force_failure"))
    if force_err:
        warnings.append({"field": "hints.force_failure", "reason": "invalid_force_failure"})
        hints.pop("force_failure", None)
    if normalized_force_failure:
        hints["force_failure"] = normalized_force_failure

    return hints, None, warnings


def _validate_camera(raw_camera: Any, request_id: str) -> Tuple[Dict[str, Any], Optional[Tuple], list]:
    """
    Validate optional camera metadata to keep mock perspective checks bounded.

    Args:
      raw_camera: payload["camera"] (expected dict-like)
      request_id: correlation id for error responses

    Returns:
      (normalized_camera, error_response_tuple, warnings)
    """
    warnings = []
    if not raw_camera:
        return {}, None, warnings
    if not isinstance(raw_camera, dict):
        warnings.append({"field": "camera", "reason": "invalid_type"})
        return {}, None, warnings

    camera = dict(raw_camera)

    # Validate FOV
    fov = camera.get("fov_deg")
    if fov is not None:
        if not isinstance(fov, (int, float)) or not (MIN_CAMERA_FOV <= float(fov) <= MAX_CAMERA_FOV):
            warnings.append({"field": "camera.fov_deg", "reason": "out_of_range"})
            camera.pop("fov_deg", None)
        else:
            camera["fov_deg"] = float(fov)

    # Validate distance
    distance = camera.get("distance_m")
    if distance is not None:
        if not isinstance(distance, (int, float)) or not (MIN_CAMERA_DISTANCE_M <= float(distance) <= MAX_CAMERA_DISTANCE_M):
            warnings.append({"field": "camera.distance_m", "reason": "out_of_range"})
            camera.pop("distance_m", None)
        else:
            camera["distance_m"] = float(distance)

    # Validate tilt
    tilt = camera.get("tilt_deg")
    if tilt is not None:
        if not isinstance(tilt, (int, float)) or not (-MAX_TILT_DEG <= float(tilt) <= MAX_TILT_DEG):
            warnings.append({"field": "camera.tilt_deg", "reason": "out_of_range"})
            camera.pop("tilt_deg", None)
        else:
            camera["tilt_deg"] = float(tilt)

    # Validate force_failure (dev/testing knob)
    normalized_force_failure, force_err = _validate_force_failure(camera.get("force_failure"))
    if force_err:
        warnings.append({"field": "camera.force_failure", "reason": "invalid_force_failure"})
        camera.pop("force_failure", None)
    if normalized_force_failure:
        camera["force_failure"] = normalized_force_failure

    return camera, None, warnings


def _validate_body_profile_inputs(
    payload: Dict[str, Any],
    request_id: str,
    as_response: bool = True
) -> Tuple[Tuple[str, Dict[str, Any], Dict[str, Any], list], Optional[Tuple]]:
    """
    Validate and normalize the JSON payload for /body_profile.

    Required:
      - image_uri (signed URL)

    Optional:
      - hints (dict)
      - camera (dict)

    Returns:
      ((image_uri, hints, camera, warnings), error_response_tuple)
    """
    image_uri = payload.get("image_uri")
    if not image_uri or not isinstance(image_uri, str) or not image_uri.strip():
        return ("", {}, {}, []), _make_error_response(
            code="INVALID_ARGUMENT",
            message="image_uri is required",
            status=400,
            request_id=request_id,
            details={"field": "image_uri"},
            raw=not as_response,
        )

    image_uri = image_uri.strip()

    if len(image_uri) > MAX_IMAGE_URI_LENGTH:
        return ("", {}, {}, []), _make_error_response(
            code="INVALID_ARGUMENT",
            message=f"image_uri is too long (max {MAX_IMAGE_URI_LENGTH} characters)",
            status=400,
            request_id=request_id,
            details={"field": "image_uri"},
            raw=not as_response,
        )

    if not _looks_like_signed_url(image_uri):
        return ("", {}, {}, []), _make_error_response(
            code="INVALID_ARGUMENT",
            message="image_uri must be a signed URL",
            status=400,
            request_id=request_id,
            details={"field": "image_uri"},
            raw=not as_response,
        )

    hints, hints_error, warnings = _validate_hints(payload.get("hints") or {}, request_id)
    if hints_error:
        return ("", {}, {}, warnings), hints_error

    camera, camera_error, camera_warnings = _validate_camera(payload.get("camera") or {}, request_id)
    warnings.extend(camera_warnings)
    if camera_error:
        return ("", {}, {}, warnings), camera_error

    return (image_uri, hints, camera, warnings), None


def _validate_engine_output(engine_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Guardrail for mocked engine output before caching/returning to clients.

    Ensures required keys exist and types are consistent.
    Also clamps probability-like values into [0, 1].
    """
    required_fields = {"body_type", "confidence", "landmarks", "proportions", "normalized", "posture", "occlusion", "quality_flags"}
    missing = required_fields - set(engine_result.keys())
    if missing:
        return False, f"Engine output missing fields: {', '.join(sorted(missing))}"

    if not isinstance(engine_result.get("confidence"), (int, float)):
        return False, "confidence must be numeric"
    if not isinstance(engine_result.get("quality_flags"), list):
        return False, "quality_flags must be a list"
    if not isinstance(engine_result.get("landmarks"), dict):
        return False, "landmarks must be an object"
    if not isinstance(engine_result.get("occlusion"), dict):
        return False, "occlusion must be an object"

    # Clamp numeric probabilities to expected bounds
    engine_result["confidence"] = max(0.0, min(1.0, float(engine_result.get("confidence", 0.0))))

    occlusion = engine_result.get("occlusion", {})
    percent = occlusion.get("percent")
    if isinstance(percent, (int, float)):
        occlusion["percent"] = max(0.0, min(1.0, float(percent)))
        engine_result["occlusion"] = occlusion

    return True, None


def _validate_presign_params(file_extension: Any, folder_path: Any):
    """
    Validate query params for pre-signed upload URLs to avoid unsafe keys.

    Returns:
      (normalized_ext, sanitized_folder, error_response, status_code)
    """
    if not file_extension or not isinstance(file_extension, str):
        return None, None, jsonify({"message": "file_extension is required and must be a string"}), 400

    normalized_ext = file_extension.lower().lstrip(".")
    if normalized_ext not in ALLOWED_IMAGE_EXTENSIONS:
        return None, None, jsonify({"message": f"Unsupported file_extension. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"}), 400

    if folder_path is None or folder_path == "":
        return normalized_ext, "", None, None
    if not isinstance(folder_path, str):
        return None, None, jsonify({"message": "folder_path must be a string"}), 400

    sanitized_folder = folder_path.strip().replace("\\", "/").strip("/")
    if not sanitized_folder:
        return normalized_ext, "", None, None

    # Prevent path traversal / weird keys / huge folder prefixes
    if ".." in sanitized_folder or len(sanitized_folder) > MAX_FOLDER_PATH_LENGTH or not FOLDER_SAFE_PATTERN.match(sanitized_folder):
        return None, None, jsonify({"message": "folder_path contains invalid characters"}), 400

    return normalized_ext, sanitized_folder, None, None


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@bodyMorph_bp.route('/testOut', methods=['GET'])
def testOut():
    """Health-check style endpoint to verify blueprint is reachable."""
    current_app.logger.info("Hello! This will go to Gunicorn logs")
    return jsonify({"message": "testOut"}), 200


# create a presigned url to store the image in the S3
@bodyMorph_bp.route('/generate-presigned-url', methods=['GET'])
@jwt_required()
@validate_request_size(request, max_json_kb=500)
@swag_from({
    "tags": ["StyleSense / BodyMorph"],
    "summary": "Generate S3 pre-signed upload URL",
    "description": "Returns a pre-signed URL and object key for uploading a file to S3.",
    "parameters": [
        {
            "in": "query",
            "name": "file_extension",
            "required": True,
            "schema": {"type": "string", "example": "jpg"},
            "description": "File extension (without dot)."
        },
        {
            "in": "query",
            "name": "folder_path",
            "required": False,
            "schema": {"type": "string", "example": "bodymorph/uploads"},
            "description": "Optional folder prefix."
        }
    ],
    "responses": {
        200: {
            "description": "Pre-signed URL generated",
            "content": {
                "application/json": {
                    "example": {"presigned_url": "https://...", "object_key": "abc123.jpg"}
                }
            }
        },
        400: {"description": "Bad request"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    }
})
def generate_presigned_url_api():
    """
    Generate a presigned URL for S3 file upload.

    Notes:
      - Object keys are sanitized to reduce risk of path traversal / unsafe keys.
      - Requires Bearer JWT (jwt_required).
    ---
    tags:
      - StyleSense / BodyMorph
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: file_extension
        required: true
        schema:
          type: string
          example: jpg
      - in: query
        name: folder_path
        required: false
        schema:
          type: string
          example: bodymorph/uploads
    responses:
      200:
        description: Pre-signed URL generated
      400:
        description: Bad request
      401:
        description: Unauthorized
      500:
        description: Internal error
    """
    try:
        # Keep S3 object keys bounded and predictable
        file_extension, folder_path, validation_error, status = _validate_presign_params(
            request.args.get('file_extension'),
            request.args.get('folder_path', '').strip()
        )
        if validation_error:
            return validation_error, status

        if not S3_BUCKET_NAME:
            return jsonify({"message": "S3 bucket is not configured"}), 500

        unique_id = str(uuid.uuid4())
        new_file_name = f"{unique_id}.{file_extension}"
        object_key = f"{folder_path}/{new_file_name}" if folder_path else new_file_name

        presigned_url = generate_presigned_url(object_key)
        if not presigned_url:
            return jsonify({"message": "Failed to generate presigned URL"}), 500

        return jsonify({
            "presigned_url": presigned_url,
            "object_key": object_key
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error generating presigned URL: {str(e)}"}), 500


# ------------------------------------------------------------------------------
# Common response / utility helpers
# ------------------------------------------------------------------------------
def _looks_like_signed_url(url: str) -> bool:
    """
    Lightweight heuristic check to ensure caller supplies a signed URL.

    This does NOT cryptographically verify signature; it just checks for
    common signature query parameters (AWS, generic, etc).
    """
    if not isinstance(url, str):
        return False
    lower = url.lower()
    signed_tokens = ["x-amz-signature", "signature=", "sig=", "token=", "x-amz-security-token"]
    return url.startswith("http") and "?" in url and any(token in lower for token in signed_tokens)


def _make_error_response(code: str, message: str, status: int, request_id: str, details: Optional[dict] = None, raw: bool = False):
    """
    Standard error payload used across endpoints for consistent client handling.

    Args:
      code: stable machine-readable error code
      message: human-readable summary
      status: HTTP status code
      request_id: correlation id returned to clients
      details: optional structured details about the error
    """
    payload = {
        "error": {
            "code": code,
            "message": message
        },
        "request_id": request_id
    }
    if details:
        payload["error"]["details"] = details
    if raw:
        return payload, status
    return jsonify(payload), status


def _extract_response_payload(resp_tuple, as_response: bool):
    """
    Normalize response tuple into a plain dict for ledger storage.
    """
    if not resp_tuple:
        return None, None
    body, status = resp_tuple
    if as_response:
        try:
            parsed = body.get_json()
        except Exception:
            parsed = None
        return parsed, status
    return body, status


def _emit_taskpulse_event(status: str, request_id: str, meta: Optional[dict] = None, step_id: Optional[str] = None):
    """
    Emit a TaskPulseOS-compatible event (best-effort; no-op if disabled).

    The event publisher is imported lazily to avoid app-context issues
    during blueprint registration.
    """
    if not TASKPULSE_ENABLED:
        return

    payload = {
        "workflow_name": TASKPULSE_WORKFLOW_NAME,
        "run_id": request_id,
        "step_id": step_id or TASKPULSE_STEP_ID,
        "status": status,
        "component": "BodyMorph",
    }
    if meta:
        payload.update(meta)

    try:
        # Lazy import to avoid app-context issues during blueprint registration
        from backend.app.event_publisher import push_update
    except Exception as exc:  # pragma: no cover - defensive
        current_app.logger.debug({
            "component": "BodyMorph",
            "event": "taskpulse_import_failed",
            "error": str(exc)
        })
        return

    try:
        push_update(payload)
    except Exception as exc:  # pragma: no cover - defensive
        current_app.logger.debug({
            "component": "BodyMorph",
            "event": "taskpulse_emit_failed",
            "error": str(exc)
        })


def _emit_step(status: str, request_id: str, step_id: str, meta: Optional[dict] = None):
    """
    Convenience wrapper to emit step-level TaskPulse events with consistent payloads.
    """
    merged_meta = dict(meta or {})
    merged_meta.setdefault("event", step_id)
    _emit_taskpulse_event(status, request_id, merged_meta, step_id=step_id)
    _record_step_status(request_id, step_id, status, merged_meta)


def _emit_trace_events(request_id: str, trace: list):
    """
    Emit TaskPulse step success events based on the engine trace.
    """
    for entry in trace or []:
        stage = entry.get("stage")
        step_id = TRACE_STAGE_TO_STEP.get(stage)
        if not step_id:
            continue
        latency = entry.get("ms")
        meta = {"event": stage}
        if latency is not None:
            meta["latency_ms"] = latency
        _emit_step("success", request_id, step_id, meta=meta)


def _emit_run_heartbeat(request_id: str, status: str = "running", meta: Optional[dict] = None):
    """
    Record a run-level heartbeat in the ledger and optionally emit a TaskPulse heartbeat event.
    """
    merged_meta = dict(meta or {})
    merged_meta.setdefault("event", "heartbeat")
    _update_run_state(request_id, {"status": status, "heartbeat_at": _now_iso()})
    _emit_taskpulse_event(status, request_id, merged_meta, step_id=TASKPULSE_STEP_IDS["workflow"])


# ------------------------------------------------------------------------------
# Cache helpers (Redis with in-memory fallback)
# ------------------------------------------------------------------------------
def _get_redis_client():
    """
    Resolve a Redis client.

    Priority:
      1) current_app.config["REDIS_CLIENT"] if provided by app factory
      2) Create a new redis.Redis client from env (REDIS_HOST/REDIS_PORT)
    """
    client = current_app.config.get("REDIS_CLIENT")
    if client:
        return client
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, db=0)


def _cache_get(client, key: str):
    """
    Cache GET with graceful fallback to in-memory cache when Redis fails.
    """
    try:
        return client.get(key)
    except Exception as exc:
        current_app.logger.warning({
            "component": "BodyMorph",
            "event": "cache_get_failed",
            "error": str(exc)
        })
        # fallback to in-memory cache
        return _LOCAL_CACHE.get(key)


def _cache_set(client, key: str, value: str, ex: int = 3600):
    """
    Cache SET with graceful fallback to in-memory cache when Redis fails.
    """
    try:
        client.set(key, value, ex=ex)
        return
    except Exception as exc:
        current_app.logger.warning({
            "component": "BodyMorph",
            "event": "cache_set_failed",
            "error": str(exc)
        })
        _LOCAL_CACHE[key] = value


# ------------------------------------------------------------------------------
# Image loading
# ------------------------------------------------------------------------------
def _load_image_bytes(signed_url: str) -> bytes:
    """
    Load image bytes using the shared Celery task.

    If the Celery task exposes a `.run()` method, call it synchronously
    from the API code path; otherwise call it as a normal function.
    """
    # Celery task can be invoked via .run for synchronous usage in the API path
    if hasattr(load_image_from_signed_url, "run"):
        return load_image_from_signed_url.run(signed_url)
    return load_image_from_signed_url(signed_url)


# ------------------------------------------------------------------------------
# "Unknown" fallback payload helpers
# ------------------------------------------------------------------------------
def _build_unknown_payload(
    request_id: str,
    total_latency_ms: int,
    trace: list,
    reason: str,
    confidence: float = 0.3,
    extra_flags: Optional[list] = None,
) -> dict:
    """
    Construct a graceful-degradation payload that still matches the expected response contract.

    This lets clients proceed even when the image quality/perspective prevents reliable analysis.
    """
    payload = {
        "request_id": request_id,
        "latency_ms": total_latency_ms,
        "body_type": "unknown",
        "confidence": round(confidence, 2),
        "landmarks": {},
        "proportions": {},
        "normalized": {},
        "posture": {},
        "occlusion": {"percent": 0.0, "regions": []},
        "quality_flags": [reason],
        "trace": trace,
    }
    if extra_flags:
        payload["quality_flags"].extend(extra_flags)
    return payload


def _respond_with_unknown(
    reason: str,
    request_id: str,
    idem_key: str,
    request_hash: str,
    trace: list,
    total_latency_ms: int,
    redis_client,
    cache_key: Optional[str] = None,
    confidence: float = 0.3,
    extra_flags: Optional[list] = None,
    raw: bool = False,
):
    """
    Persist and return an "unknown" response.

    - Writes idempotency record so retries don't reprocess
    - Optionally caches response (if cache_key provided)
    - Emits TaskPulse event as success but includes degraded_reason
    """
    assembly_start = time.time()
    final_response = _build_unknown_payload(
        request_id,
        total_latency_ms,
        trace,
        reason,
        confidence=confidence,
        extra_flags=extra_flags,
    )
    assembly_ms = int((time.time() - assembly_start) * 1000)
    trace.append({"stage": "assemble_response", "ms": assembly_ms})

    write_idempotency(idem_key, request_hash, {"body": final_response, "_status": 200})
    if cache_key:
        cache_start = time.time()
        _cache_set(redis_client, cache_key, json.dumps(final_response), ex=3600)
        cache_ms = int((time.time() - cache_start) * 1000)
        trace.append({"stage": "cache_write", "ms": cache_ms})

    telemetry_start = time.time()
    _emit_step(
        "success",
        request_id,
        TASKPULSE_STEP_IDS["workflow"],
        {
            "event": "body_profile_completed",
            "latency_ms": total_latency_ms,
            "body_type": final_response.get("body_type"),
            "confidence": final_response.get("confidence"),
            "source": "engine",
            "degraded_reason": reason,
        },
    )
    telemetry_ms = int((time.time() - telemetry_start) * 1000)
    trace.append({"stage": "telemetry_emit", "ms": telemetry_ms})
    final_response["trace"] = trace
    _emit_trace_events(request_id, trace)

    _finalize_run_state(
        request_id,
        "degraded",
        response_body=final_response,
        http_status=200,
        trace=trace,
        degraded_reason=reason,
        cache_hit=bool(cache_key),
    )

    if raw:
        return final_response, 200
    return jsonify(final_response), 200


def _process_body_profile_request(
    payload: Dict[str, Any],
    idem_key: str,
    request_id: str,
    start_time: float,
    mode: str = "sync",
    as_response: bool = True,
):
    """
    Shared BodyMorph pipeline used by both synchronous and async/queued entrypoints.
    """
    redis_client = _get_redis_client()
    validation_start = time.time()
    (image_uri, hints, camera, warnings), validation_error = _validate_body_profile_inputs(
        payload,
        request_id,
        as_response=as_response
    )
    validation_ms = int((time.time() - validation_start) * 1000)
    trace = [{"stage": "validate_input", "ms": validation_ms}]
    validation_meta = {"latency_ms": validation_ms}
    if warnings:
        validation_meta["warnings"] = warnings
        _update_run_state(request_id, {"validation_warnings": warnings})

    if validation_error:
        _emit_step(
            "failure",
            request_id,
            TASKPULSE_STEP_IDS["validate_input"],
            {"error": "schema_validation_failed"},
        )
        _emit_step(
            "failure",
            request_id,
            TASKPULSE_STEP_IDS["workflow"],
            {"error": "schema_validation_failed"},
        )
        resp_body, resp_status = _extract_response_payload(validation_error, as_response)
        _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status, trace=trace)
        return validation_error

    _emit_step("success", request_id, TASKPULSE_STEP_IDS["validate_input"], validation_meta)
    validation_penalty = OPTIONAL_VALIDATION_CONFIDENCE_PENALTY if warnings else 0.0
    _emit_run_heartbeat(request_id, status="running")

    normalized_payload = dict(payload)
    normalized_payload["image_uri"] = image_uri
    normalized_payload["hints"] = hints
    normalized_payload["camera"] = camera
    request_hash = compute_request_hash(normalized_payload)
    _update_run_state(request_id, {"request_hash": request_hash, "idem_key": idem_key, "mode": mode})

    existing = read_idempotency(idem_key)
    if existing:
        try:
            existing_payload = json.loads(existing.response_json)
        except Exception:
            existing_payload = None

        if existing.request_hash and existing.request_hash != request_hash:
            _emit_step(
                "failure",
                request_id,
                TASKPULSE_STEP_IDS["validate_input"],
                {"error": "idempotency_payload_mismatch"},
            )
            _emit_step(
                "failure",
                request_id,
                TASKPULSE_STEP_IDS["workflow"],
                {"error": "idempotency_payload_mismatch"},
            )
            err = _make_error_response(
                code="INVALID_ARGUMENT",
                message="Payload does not match the original idempotent request",
                status=400,
                request_id=request_id,
                details={"field": "Idempotency-Key"},
                raw=not as_response,
            )
            resp_body, resp_status = _extract_response_payload(err, as_response)
            _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status, trace=trace)
            return err

        if existing_payload is not None:
            stored_body = existing_payload.get("body") if isinstance(existing_payload, dict) else existing_payload
            stored_status = (
                existing_payload.get("_status")
                if isinstance(existing_payload, dict)
                else None
            )
            current_app.logger.info({
                "component": "BodyMorph",
                "request_id": stored_body.get("request_id", request_id) if isinstance(stored_body, dict) else request_id,
                "event": "idempotent_replay"
            })
            _emit_step(
                "success",
                request_id,
                TASKPULSE_STEP_IDS["workflow"],
                {"event": "idempotent_replay"},
            )
            _finalize_run_state(
                request_id,
                "succeeded",
                response_body=stored_body if isinstance(stored_body, dict) else None,
                http_status=stored_status or 200,
                trace=stored_body.get("trace") if isinstance(stored_body, dict) else trace,
                cache_hit=True,
            )
            if as_response:
                return jsonify(stored_body), stored_status or 200
            return stored_body, stored_status or 200

    # --------------------------------------------------------------------------
    # Load image via signed URL
    # --------------------------------------------------------------------------
    _emit_step("running", request_id, TASKPULSE_STEP_IDS["load_image_reference"])
    load_start = time.time()
    try:
        imag_file = _load_image_bytes(image_uri)
        file_like = io.BytesIO(imag_file)
        img_bytes = file_like.read()
        load_ms = int((time.time() - load_start) * 1000)
        trace.append({"stage": "load_image_reference", "ms": load_ms})
        _emit_step(
            "success",
            request_id,
            TASKPULSE_STEP_IDS["load_image_reference"],
            {"latency_ms": load_ms, "source": "remote"},
        )
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response else None
        current_app.logger.warning({
            "component": "BodyMorph",
            "request_id": request_id,
            "event": "load_image_http_error",
            "status_code": status_code,
            "error": str(exc),
        })
        if ALLOW_PLACEHOLDER_IMAGE:
            img_bytes = PLACEHOLDER_IMAGE_BYTES
            load_ms = int((time.time() - load_start) * 1000)
            trace.append({"stage": "load_image_reference", "ms": load_ms})
            _emit_step(
                "success",
                request_id,
                TASKPULSE_STEP_IDS["load_image_reference"],
                {"latency_ms": load_ms, "source": "placeholder_http_error"},
            )
        else:
            _emit_step(
                "failure",
                request_id,
                TASKPULSE_STEP_IDS["load_image_reference"],
                {"error": "http_error", "http_status": status_code},
            )
            err = _make_error_response(
                code="IMAGE_FETCH_FAILED",
                message="Image URL returned an HTTP error",
                status=400,
                request_id=request_id,
                details={"http_status": status_code},
                raw=not as_response,
            )
            resp_body, resp_status = _extract_response_payload(err, as_response)
            _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status, trace=trace)
            return err
    except requests.exceptions.RequestException as exc:
        current_app.logger.warning({
            "component": "BodyMorph",
            "request_id": request_id,
            "event": "load_image_request_error",
            "error": str(exc),
        })
        if ALLOW_PLACEHOLDER_IMAGE:
            img_bytes = PLACEHOLDER_IMAGE_BYTES
            load_ms = int((time.time() - load_start) * 1000)
            trace.append({"stage": "load_image_reference", "ms": load_ms})
            _emit_step(
                "success",
                request_id,
                TASKPULSE_STEP_IDS["load_image_reference"],
                {"latency_ms": load_ms, "source": "placeholder_request_error"},
            )
        else:
            _emit_step(
                "failure",
                request_id,
                TASKPULSE_STEP_IDS["load_image_reference"],
                {"error": "request_exception"},
            )
            err = _make_error_response(
                code="IMAGE_FETCH_FAILED",
                message="Could not fetch image from signed URL",
                status=400,
                request_id=request_id,
                details={"reason": str(exc)},
                raw=not as_response,
            )
            resp_body, resp_status = _extract_response_payload(err, as_response)
            _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status, trace=trace)
            return err
    except Exception as exc:
        current_app.logger.error({
            "component": "BodyMorph",
            "request_id": request_id,
            "event": "load_image_failed",
            "error": str(exc),
        })
        if ALLOW_PLACEHOLDER_IMAGE:
            img_bytes = PLACEHOLDER_IMAGE_BYTES
            load_ms = int((time.time() - load_start) * 1000)
            trace.append({"stage": "load_image_reference", "ms": load_ms})
            _emit_step(
                "success",
                request_id,
                TASKPULSE_STEP_IDS["load_image_reference"],
                {"latency_ms": load_ms, "source": "placeholder_generic_error"},
            )
        else:
            _emit_step(
                "failure",
                request_id,
                TASKPULSE_STEP_IDS["load_image_reference"],
                {"error": "internal_load_error"},
            )
            err = _make_error_response(
                code="INTERNAL",
                message="Failed to load image from signed URL",
                status=500,
                request_id=request_id,
                raw=not as_response,
            )
            resp_body, resp_status = _extract_response_payload(err, as_response)
            _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status, trace=trace)
            return err

    # --------------------------------------------------------------------------
    # Cache lookup
    # --------------------------------------------------------------------------
    cache_key = generate_image_cache_key(image_bytes=img_bytes, hints=hints)
    cache_found = _cache_get(redis_client, cache_key)
    if cache_found:
        try:
            cached_response = json.loads(cache_found)
        except Exception:
            cached_response = None

        if cached_response:
            write_idempotency(idem_key, request_hash, {"body": cached_response, "_status": 200})
            current_app.logger.info({
                "component": "BodyMorph",
                "request_id": cached_response.get("request_id", request_id)
                if isinstance(cached_response, dict) else request_id,
                "event": "cache_hit",
                "cache_key": cache_key
            })
            total_latency_ms = int((time.time() - start_time) * 1000)
            _emit_step(
                "success",
                request_id,
                TASKPULSE_STEP_IDS["workflow"],
                {
                    "event": "body_profile_cached",
                    "latency_ms": total_latency_ms,
                    "cache_key": cache_key,
                    "source": "cache"
                },
            )
            _finalize_run_state(
                request_id,
                "succeeded",
                response_body=cached_response,
                http_status=200,
                trace=cached_response.get("trace") if isinstance(cached_response, dict) else trace,
                cache_hit=True,
            )
            if as_response:
                return jsonify(cached_response), 200
            return cached_response, 200

    # --------------------------------------------------------------------------
    # Run mocked BodyMorph engine
    # --------------------------------------------------------------------------
    engine = BodyMorphEngine()
    engine_start = time.time()
    engine_trace: list = []
    extra_flags = ["optional_input_degraded"] if validation_penalty else None
    try:
        engine_result, engine_trace = engine.run(img_bytes, hints, camera)
    except MultiSubjectError as exc:
        return _handle_engine_error(
            code="MULTI_SUBJECT",
            message=str(exc),
            status=400,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            taskpulse_meta={"latency_ms": int((time.time() - start_time) * 1000)},
            step_id=TASKPULSE_STEP_IDS["detect_person"],
            trace=trace,
            raw=not as_response,
        )
    except LowQualityError as exc:
        engine_ms = int((time.time() - engine_start) * 1000)
        total_latency_ms = int((time.time() - start_time) * 1000)
        trace.extend(engine_trace or [])
        if not any(stage.get("stage") == "engine_total" for stage in trace):
            trace.append({"stage": "engine_total", "ms": engine_ms})
        adjusted_conf = max(0.05, 0.3 - validation_penalty)
        return _respond_with_unknown(
            reason="low_quality",
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            trace=trace,
            total_latency_ms=total_latency_ms,
            redis_client=redis_client,
            cache_key=cache_key,
            confidence=adjusted_conf,
            extra_flags=extra_flags,
            raw=not as_response,
        )
    except HeavyOcclusionError as exc:
        engine_ms = int((time.time() - engine_start) * 1000)
        total_latency_ms = int((time.time() - start_time) * 1000)
        trace.extend(engine_trace or [])
        if not any(stage.get("stage") == "engine_total" for stage in trace):
            trace.append({"stage": "engine_total", "ms": engine_ms})
        adjusted_conf = max(0.05, 0.35 - validation_penalty)
        return _respond_with_unknown(
            reason="heavy_occlusion",
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            trace=trace,
            total_latency_ms=total_latency_ms,
            redis_client=redis_client,
            cache_key=cache_key,
            confidence=adjusted_conf,
            extra_flags=extra_flags,
            raw=not as_response,
        )
    except ExtremePerspectiveError as exc:
        engine_ms = int((time.time() - engine_start) * 1000)
        total_latency_ms = int((time.time() - start_time) * 1000)
        trace.extend(engine_trace or [])
        if not any(stage.get("stage") == "engine_total" for stage in trace):
            trace.append({"stage": "engine_total", "ms": engine_ms})
        adjusted_conf = max(0.05, 0.3 - validation_penalty)
        return _respond_with_unknown(
            reason="extreme_perspective",
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            trace=trace,
            total_latency_ms=total_latency_ms,
            redis_client=redis_client,
            cache_key=cache_key,
            confidence=adjusted_conf,
            extra_flags=extra_flags,
            raw=not as_response,
        )
    except UnknownSilhouetteError as exc:
        engine_ms = int((time.time() - engine_start) * 1000)
        total_latency_ms = int((time.time() - start_time) * 1000)
        trace.extend(engine_trace or [])
        if not any(stage.get("stage") == "engine_total" for stage in trace):
            trace.append({"stage": "engine_total", "ms": engine_ms})
        adjusted_conf = max(0.05, 0.25 - validation_penalty)
        return _respond_with_unknown(
            reason="unknown_silhouette",
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            trace=trace,
            total_latency_ms=total_latency_ms,
            redis_client=redis_client,
            cache_key=cache_key,
            confidence=adjusted_conf,
            extra_flags=extra_flags,
            raw=not as_response,
        )
    except BodyMorphError as exc:
        return _handle_engine_error(
            code="INTERNAL",
            message=str(exc),
            status=500,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            taskpulse_meta={"latency_ms": int((time.time() - start_time) * 1000)},
            step_id=TASKPULSE_STEP_IDS["workflow"],
            trace=trace,
            raw=not as_response,
        )
    except Exception as exc:  # pragma: no cover - safety net
        current_app.logger.exception({
            "component": "BodyMorph",
            "request_id": request_id,
            "event": "body_profile_exception",
            "error": str(exc)
        })
        return _handle_engine_error(
            code="INTERNAL",
            message="Unexpected error during body profile processing",
            status=500,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            taskpulse_meta={"latency_ms": int((time.time() - start_time) * 1000)},
            step_id=TASKPULSE_STEP_IDS["workflow"],
            trace=trace,
            raw=not as_response,
        )

    engine_ms = int((time.time() - engine_start) * 1000)
    total_latency_ms = int((time.time() - start_time) * 1000)

    # Validate engine payload contract before returning/caching
    is_valid_output, engine_output_error = _validate_engine_output(engine_result)
    if not is_valid_output:
        current_app.logger.error({
            "component": "BodyMorph",
            "request_id": request_id,
            "event": "engine_output_invalid",
            "error": engine_output_error
        })
        return _handle_engine_error(
            code="INTERNAL",
            message="Engine returned an invalid payload",
            status=500,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash,
            taskpulse_meta={"latency_ms": total_latency_ms},
            trace=trace,
            raw=not as_response,
        )

    trace.extend(engine_trace or [])
    if not any(stage.get("stage") == "engine_total" for stage in trace):
        trace.append({"stage": "engine_total", "ms": engine_ms})

    # Compose trace and final response
    assembly_start = time.time()

    final_response = {
        "request_id": request_id,
        "latency_ms": total_latency_ms,
        **engine_result,
        "trace": trace
    }

    if validation_penalty:
        final_response["confidence"] = max(0.0, round(final_response.get("confidence", 0.0) - validation_penalty, 2))
        qf = list(final_response.get("quality_flags", []))
        if "optional_input_degraded" not in qf:
            qf.append("optional_input_degraded")
        final_response["quality_flags"] = qf

    assembly_ms = int((time.time() - assembly_start) * 1000)
    trace.append({"stage": "assemble_response", "ms": assembly_ms})

    # Persist idempotency and cache response
    write_idempotency(idem_key, request_hash, {"body": final_response, "_status": 200})
    cache_start = time.time()
    _cache_set(redis_client, cache_key, json.dumps(final_response), ex=3600)
    cache_ms = int((time.time() - cache_start) * 1000)
    trace.append({"stage": "cache_write", "ms": cache_ms})

    # Observability: total trace time (useful sanity check)
    trace_ms = sum(stage.get("ms", 0) for stage in trace)

    telemetry_start = time.time()
    current_app.logger.info({
        "component": "BodyMorph",
        "request_id": request_id,
        "event": "body_profile_success",
        "body_type": final_response.get("body_type"),
        "confidence": final_response.get("confidence"),
        "latency_ms": total_latency_ms
    })

    _emit_step(
        "success",
        request_id,
        TASKPULSE_STEP_IDS["workflow"],
        {
            "event": "body_profile_completed",
            "latency_ms": total_latency_ms,
            "trace_ms": trace_ms,
            "body_type": final_response.get("body_type"),
            "confidence": final_response.get("confidence"),
            "source": "engine",
            "validation_penalty": validation_penalty if warnings else 0.0,
        },
    )
    telemetry_ms = int((time.time() - telemetry_start) * 1000)
    trace.append({"stage": "telemetry_emit", "ms": telemetry_ms})
    final_response["trace"] = trace
    _emit_trace_events(request_id, trace)
    _finalize_run_state(
        request_id,
        "succeeded",
        response_body=final_response,
        http_status=200,
        trace=trace,
        degraded_reason="optional_input_degraded" if validation_penalty else None,
    )

    if as_response:
        return jsonify(final_response), 200
    return final_response, 200


# ------------------------------------------------------------------------------
# Body profile route
# ------------------------------------------------------------------------------
@bodyMorph_bp.route("/body_profile", methods=["POST"])
@jwt_required()
@validate_request_size(request, max_json_kb=500)
@swag_from({
    "tags": ["StyleSense / BodyMorph"],
    "summary": "Analyze body profile (mocked engine)",
    "description": "Accepts an image signed URL plus optional hints/camera metadata and returns mocked body morphology output. Idempotency and caching are applied.",
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image_uri": {"type": "string", "example": "https://.../image.jpg?X-Amz-Signature=abc"},
                        "hints": {
                            "type": "object",
                            "example": {"height_cm": 168, "force_failure": "LOW_QUALITY"}
                        },
                        "camera": {
                            "type": "object",
                            "example": {"fov_deg": 60, "distance_m": 2.0}
                        }
                    },
                    "required": ["image_uri"]
                }
            }
        }
    },
    "responses": {
        200: {
            "description": "Body profile computed",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "req-123",
                        "latency_ms": 120,
                        "body_type": "hourglass",
                        "confidence": 0.82,
                        "landmarks": {"shoulder_L": [1, 2]},
                        "proportions": {"shoulder_width_px": 412},
                        "normalized": {"shoulder_to_hip_ratio": 1.02},
                        "posture": {"tilt": "neutral"},
                        "occlusion": {"percent": 0.11, "regions": []},
                        "quality_flags": ["single_subject"],
                        "trace": [{"stage": "load_image", "ms": 20}]
                    }
                }
            }
        },
        400: {
            "description": "Validation or engine failure",
            "content": {
                "application/json": {
                    "example": {
                        "error": {"code": "INVALID_ARGUMENT", "message": "image_uri is required", "details": {"field": "image_uri"}},
                        "request_id": "req-123"
                    }
                }
            }
        },
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    },
    "security": [{"BearerAuth": []}],
})
def body_profile():
    """
    Accepts JSON payload with image_uri, hints, and camera info.
    Returns mocked body analysis results.

    Key behaviors:
      - Requires Idempotency-Key header (prevents reprocessing on client retries)
      - Validates payload shape and bounds
      - Loads image from signed URL (optionally falls back to placeholder)
      - Checks cache for repeated inputs
      - Runs BodyMorphEngine and validates output
      - Persists idempotency record and caches response
      - Emits TaskPulse events if enabled
    ---
    tags:
      - StyleSense / BodyMorph
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              image_uri:
                type: string
                example: https://example.com/image.jpg?X-Amz-Signature=abc
              hints:
                type: object
                example: {"height_cm": 168, "force_failure": "LOW_QUALITY"}
              camera:
                type: object
                example: {"fov_deg": 60, "distance_m": 2.0}
            required:
              - image_uri
    responses:
      200:
        description: Body profile computed
      400:
        description: Validation or engine failure
      401:
        description: Unauthorized
      500:
        description: Internal error
    """
    start_time = time.time()

    # Correlation ID: accept client-provided X-Request-ID or generate one
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    _init_run_state(request_id, mode="sync", status="running")
    _emit_step("running", request_id, TASKPULSE_STEP_IDS["workflow"], {"event": "workflow_started", "mode": "sync"})
    _emit_step("running", request_id, TASKPULSE_STEP_IDS["validate_input"])
    _emit_run_heartbeat(request_id, status="running")

    # Client must provide an Idempotency-Key to prevent reprocessing on retries
    idem_key = request.headers.get("Idempotency-Key")
    if not idem_key:
        _emit_step(
            "failure",
            request_id,
            TASKPULSE_STEP_IDS["validate_input"],
            {"error": "missing_idempotency_key"},
        )
        _emit_step(
            "failure",
            request_id,
            TASKPULSE_STEP_IDS["workflow"],
            {"error": "missing_idempotency_key"},
        )
        resp = _make_error_response(
            code="INVALID_ARGUMENT",
            message="Idempotency-Key header is required",
            status=400,
            request_id=request_id,
            details={"header": "Idempotency-Key"}
        )
        resp_body, resp_status = _extract_response_payload(resp, as_response=True)
        _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status)
        return resp

    # Parse JSON body
    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        _emit_step(
            "failure",
            request_id,
            TASKPULSE_STEP_IDS["validate_input"],
            {"error": "invalid_json"},
        )
        _emit_step(
            "failure",
            request_id,
            TASKPULSE_STEP_IDS["workflow"],
            {"error": "invalid_json"},
        )
        resp = _make_error_response(
            code="INVALID_ARGUMENT",
            message="Invalid JSON payload",
            status=400,
            request_id=request_id
        )
        resp_body, resp_status = _extract_response_payload(resp, as_response=True)
        _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status)
        return resp

    return _process_body_profile_request(data, idem_key, request_id, start_time, mode="sync", as_response=True)


def _run_async_body_profile(app, request_id: str, idem_key: str, data: dict, start_time: float):
    """
    Background executor entrypoint for async BodyMorph runs.
    """
    with app.app_context():
        try:
            _update_run_state(request_id, {"status": "running", "started_at": _now_iso(), "mode": "async"})
            _emit_step("running", request_id, TASKPULSE_STEP_IDS["workflow"], {"event": "workflow_started", "mode": "async"})
            _emit_step("running", request_id, TASKPULSE_STEP_IDS["validate_input"])
            _emit_run_heartbeat(request_id, status="running")
            _process_body_profile_request(data, idem_key, request_id, start_time, mode="async", as_response=False)
        except Exception as exc:  # pragma: no cover - defensive
            current_app.logger.exception({
                "component": "BodyMorph",
                "request_id": request_id,
                "event": "body_profile_async_failed",
                "error": str(exc),
            })
            error_payload = {
                "error": {"code": "INTERNAL", "message": "Async BodyMorph failed"},
                "request_id": request_id,
            }
            _finalize_run_state(request_id, "failed", response_body=error_payload, http_status=500)
            _emit_taskpulse_event("failure", request_id, {"event": "body_profile_async_failed", "error": str(exc)}, step_id=TASKPULSE_STEP_IDS["workflow"])


@bodyMorph_bp.route("/body_profile/async", methods=["POST"])
@jwt_required()
@validate_request_size(request, max_json_kb=500)
@swag_from({
    "tags": ["StyleSense / BodyMorph"],
    "summary": "Queue BodyMorph analysis (mocked engine)",
    "description": "Queues a BodyMorph run and returns a run id for polling. Uses the same payload as the synchronous endpoint.",
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image_uri": {"type": "string", "example": "https://.../image.jpg?X-Amz-Signature=abc"},
                        "hints": {
                            "type": "object",
                            "example": {"height_cm": 168, "force_failure": "LOW_QUALITY"}
                        },
                        "camera": {
                            "type": "object",
                            "example": {"fov_deg": 60, "distance_m": 2.0}
                        }
                    },
                    "required": ["image_uri"]
                }
            }
        }
    },
    "responses": {
        202: {
            "description": "Run queued",
            "content": {
                "application/json": {
                    "example": {"request_id": "req-123", "status": "queued"}
                }
            }
        },
        400: {"description": "Validation failure (e.g., missing Idempotency-Key or bad JSON)"},
        401: {"description": "Unauthorized"},
        503: {"description": "Async disabled"},
        500: {"description": "Internal server error"}
    },
    "security": [{"BearerAuth": []}],
})
def body_profile_async():
    """
    Queue BodyMorph processing and return immediately with a run id.
    """
    if not BODYMORPH_ASYNC_ENABLED or not _EXECUTOR:
        return jsonify({"message": "Async BodyMorph is disabled"}), 503

    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    idem_key = request.headers.get("Idempotency-Key")
    if not idem_key:
        resp = _make_error_response(
            code="INVALID_ARGUMENT",
            message="Idempotency-Key header is required",
            status=400,
            request_id=request_id,
            details={"header": "Idempotency-Key"}
        )
        resp_body, resp_status = _extract_response_payload(resp, as_response=True)
        _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status)
        _emit_step("failure", request_id, TASKPULSE_STEP_IDS["workflow"], {"error": "missing_idempotency_key"})
        return resp

    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        resp = _make_error_response(
            code="INVALID_ARGUMENT",
            message="Invalid JSON payload",
            status=400,
            request_id=request_id
        )
        resp_body, resp_status = _extract_response_payload(resp, as_response=True)
        _finalize_run_state(request_id, "failed", response_body=resp_body, http_status=resp_status)
        _emit_step("failure", request_id, TASKPULSE_STEP_IDS["workflow"], {"error": "invalid_json"})
        return resp

    _init_run_state(request_id, mode="async", status="queued", idem_key=idem_key)
    _emit_taskpulse_event("running", request_id, {"event": "workflow_queued", "mode": "async"}, step_id=TASKPULSE_STEP_IDS["workflow"])
    _emit_run_heartbeat(request_id, status="queued")

    try:
        app = current_app._get_current_object()
        _EXECUTOR.submit(_run_async_body_profile, app, request_id, idem_key, data, start_time)
    except Exception as exc:
        current_app.logger.exception({
            "component": "BodyMorph",
            "request_id": request_id,
            "event": "body_profile_queue_failed",
            "error": str(exc),
        })
        error_payload = {
            "error": {"code": "INTERNAL", "message": "Failed to queue BodyMorph run"},
            "request_id": request_id,
        }
        _finalize_run_state(request_id, "failed", response_body=error_payload, http_status=500)
        return jsonify(error_payload), 500

    return jsonify({"request_id": request_id, "status": "queued"}), 202


@bodyMorph_bp.route("/body_profile/runs/<string:run_id>", methods=["GET"])
@jwt_required()
@swag_from({
    "tags": ["StyleSense / BodyMorph"],
    "summary": "Get BodyMorph run status",
    "description": "Poll the status/trace/response for a BodyMorph run (sync or async).",
    "parameters": [
        {
            "in": "path",
            "name": "run_id",
            "required": True,
            "schema": {"type": "string"},
            "description": "Run id returned by the sync or async BodyMorph call."
        }
    ],
    "responses": {
        200: {
            "description": "Run state",
            "content": {
                "application/json": {
                    "example": {
                        "run_id": "req-123",
                        "status": "succeeded",
                        "response": {"body_type": "hourglass", "confidence": 0.82},
                        "trace": [{"stage": "load_image_reference", "ms": 25}]
                    }
                }
            }
        },
        401: {"description": "Unauthorized"},
        404: {"description": "Run not found"},
        500: {"description": "Internal server error"}
    },
    "security": [{"BearerAuth": []}],
})
def body_profile_status(run_id: str):
    """
    Poll run status/trace for either sync or async BodyMorph runs.
    """
    state = _read_run_state(run_id)
    if not state:
        return jsonify({"message": "Run not found", "run_id": run_id}), 404
    return jsonify(state), 200


# ------------------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------------------
def _handle_engine_error(
    code: str,
    message: str,
    status: int,
    request_id: str,
    idem_key: str,
    request_hash: str,
    taskpulse_meta: Optional[dict] = None,
    step_id: Optional[str] = None,
    trace: Optional[list] = None,
    raw: bool = False,
):
    """
    Standardize engine error responses.

    - Emits TaskPulse failure event (if enabled)
    - Writes idempotency record with error body and HTTP status
    - Returns JSON error body
    """
    error_body = {
        "error": {
            "code": code,
            "message": message
        },
        "request_id": request_id
    }

    meta = {"event": "body_profile_failed", "error_code": code, "http_status": status}
    if taskpulse_meta:
        meta.update(taskpulse_meta)
    _emit_step("failure", request_id, step_id or TASKPULSE_STEP_IDS["workflow"], meta)

    write_idempotency(idem_key, request_hash, {"body": error_body, "_status": status})
    _finalize_run_state(request_id, "failed", response_body=error_body, http_status=status, trace=trace)
    current_app.logger.warning({
        "component": "BodyMorph",
        "request_id": request_id,
        "event": "body_profile_error",
        "error_code": code,
        "status": status
    })
    if raw:
        return error_body, status
    return jsonify(error_body), status
