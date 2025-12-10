import io
import json
import os
import time
import uuid

import redis
import requests
from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import jwt_required
from typing import Optional
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

bodyMorph_bp = Blueprint('bodyMorph_bp', __name__, url_prefix="/stylesense")

# Maximum file size (for frontend validation reference)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Allow a lightweight placeholder image when fetch fails (useful for dev/test)
ALLOW_PLACEHOLDER_IMAGE = os.getenv("BODYMORPH_ALLOW_PLACEHOLDER", "true").lower() == "true"
PLACEHOLDER_IMAGE_BYTES = b"placeholder-bodymorph-image"

# Initialize Boto3 S3 client
s3_client = AWSConfig.get_s3_client()
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Fallback in-memory cache if Redis is unavailable
_LOCAL_CACHE = {}

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
    
@bodyMorph_bp.route('/testOut', methods=['GET'])
def testOut():
    current_app.logger.info("Hello! This will go to Gunicorn logs")
    return jsonify({"message": "testOut"}), 200
    
# create a presigned url to store the image in the S3
@bodyMorph_bp.route('/generate-presigned-url', methods=['GET'])
@jwt_required()
@validate_request_size(request,max_json_kb=500)
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
        # Get file extension from query parameters
        file_extension = request.args.get('file_extension')
        if not file_extension:
            return jsonify({"message": "file_extension is required"}), 400

        # Ensure file extension starts with a dot
        file_extension = f".{file_extension}"

        # Get optional folder path if provided
        folder_path = request.args.get('folder_path', '').strip()
        
        # Generate a unique filename using UUID
        unique_id = str(uuid.uuid4())
        new_file_name = f"{unique_id}{file_extension}"

        # Set the object key with folder path if specified
        object_key = f"{folder_path}/{new_file_name}" if folder_path else new_file_name

        # Generate the pre-signed URL
        presigned_url = generate_presigned_url(object_key)
        if not presigned_url:
            return jsonify({"message": "Failed to generate presigned URL"}), 500

        return jsonify({
            "presigned_url": presigned_url,
            "object_key": object_key
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error generating presigned URL: {str(e)}"}), 500


def _looks_like_signed_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    lower = url.lower()
    signed_tokens = ["x-amz-signature", "signature=", "sig=", "token=", "x-amz-security-token"]
    return url.startswith("http") and "?" in url and any(token in lower for token in signed_tokens)


def _make_error_response(code: str, message: str, status: int, request_id: str, details: Optional[dict] = None):
    payload = {
        "error": {
            "code": code,
            "message": message
        },
        "request_id": request_id
    }
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status


def _get_redis_client():
    client = current_app.config.get("REDIS_CLIENT")
    if client:
        return client
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, db=0)


def _cache_get(client, key: str):
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


def _load_image_bytes(signed_url: str) -> bytes:
    # Celery task can be invoked via .run for synchronous usage in the API path
    if hasattr(load_image_from_signed_url, "run"):
        return load_image_from_signed_url.run(signed_url)
    return load_image_from_signed_url(signed_url)


@bodyMorph_bp.route("/body_profile", methods=["POST"])
@jwt_required() 
@validate_request_size(request,max_json_kb=500)
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
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    idem_key = request.headers.get("Idempotency-Key")
    if not idem_key:
        return _make_error_response(
            code="INVALID_ARGUMENT",
            message="Idempotency-Key header is required",
            status=400,
            request_id=request_id,
            details={"header": "Idempotency-Key"}
        )

    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        return _make_error_response(
            code="INVALID_ARGUMENT",
            message="Invalid JSON payload",
            status=400,
            request_id=request_id
        )

    image_uri = data.get("image_uri")
    hints = data.get("hints") or {}
    camera = data.get("camera") or {}

    if not image_uri or not isinstance(image_uri, str) or not image_uri.strip():
        return _make_error_response(
            code="INVALID_ARGUMENT",
            message="image_uri is required",
            status=400,
            request_id=request_id,
            details={"field": "image_uri"}
        )
    if not _looks_like_signed_url(image_uri):
        return _make_error_response(
            code="INVALID_ARGUMENT",
            message="image_uri must be a signed URL",
            status=400,
            request_id=request_id,
            details={"field": "image_uri"}
        )
    if hints and not isinstance(hints, dict):
        return _make_error_response(
            code="INVALID_ARGUMENT",
            message="hints must be an object",
            status=400,
            request_id=request_id,
            details={"field": "hints"}
        )
    if camera and not isinstance(camera, dict):
        return _make_error_response(
            code="INVALID_ARGUMENT",
            message="camera must be an object",
            status=400,
            request_id=request_id,
            details={"field": "camera"}
        )

    request_hash = compute_request_hash(data)
    existing = read_idempotency(idem_key)
    if existing:
        try:
            existing_payload = json.loads(existing.response_json)
        except Exception:
            existing_payload = None

        if existing.request_hash and existing.request_hash != request_hash:
            return _make_error_response(
                code="INVALID_ARGUMENT",
                message="Payload does not match the original idempotent request",
                status=400,
                request_id=request_id,
                details={"field": "Idempotency-Key"}
            )

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
            return jsonify(stored_body), stored_status or 200

    redis_client = _get_redis_client()

    # Load image via signed URL
    load_start = time.time()
    try:
        imag_file = _load_image_bytes(image_uri)
        file_like = io.BytesIO(imag_file)
        img_bytes = file_like.read()
        load_ms = int((time.time() - load_start) * 1000)
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
        else:
            return _make_error_response(
                code="IMAGE_FETCH_FAILED",
                message="Image URL returned an HTTP error",
                status=400,
                request_id=request_id,
                details={"http_status": status_code}
            )
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
        else:
            return _make_error_response(
                code="IMAGE_FETCH_FAILED",
                message="Could not fetch image from signed URL",
                status=400,
                request_id=request_id,
                details={"reason": str(exc)}
            )
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
        else:
            return _make_error_response(
                code="INTERNAL",
                message="Failed to load image from signed URL",
                status=500,
                request_id=request_id
            )

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
            return jsonify(cached_response), 200

    # Run mocked BodyMorph engine
    engine = BodyMorphEngine()
    engine_start = time.time()
    try:
        engine_result, engine_trace = engine.run(img_bytes, hints, camera)
    except MultiSubjectError as exc:
        return _handle_engine_error(
            code="MULTI_SUBJECT",
            message=str(exc),
            status=400,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash
        )
    except LowQualityError as exc:
        return _handle_engine_error(
            code="LOW_QUALITY",
            message=str(exc),
            status=400,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash
        )
    except HeavyOcclusionError as exc:
        return _handle_engine_error(
            code="HEAVY_OCCLUSION",
            message=str(exc),
            status=400,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash
        )
    except ExtremePerspectiveError as exc:
        return _handle_engine_error(
            code="EXTREME_PERSPECTIVE",
            message=str(exc),
            status=400,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash
        )
    except UnknownSilhouetteError as exc:
        return _handle_engine_error(
            code="UNKNOWN",
            message=str(exc),
            status=400,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash
        )
    except BodyMorphError as exc:
        return _handle_engine_error(
            code="INTERNAL",
            message=str(exc),
            status=500,
            request_id=request_id,
            idem_key=idem_key,
            request_hash=request_hash
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
            request_hash=request_hash
        )

    engine_ms = int((time.time() - engine_start) * 1000)
    total_latency_ms = int((time.time() - start_time) * 1000)

    trace = [
        {"stage": "load_image", "ms": load_ms},
        {"stage": "mock_bodymorph", "ms": engine_ms},
    ]
    if engine_trace:
        trace.extend(engine_trace)

    final_response = {
        "request_id": request_id,
        "latency_ms": total_latency_ms,
        **engine_result,
        "trace": trace
    }

    write_idempotency(idem_key, request_hash, {"body": final_response, "_status": 200})
    _cache_set(redis_client, cache_key, json.dumps(final_response), ex=3600)

    current_app.logger.info({
        "component": "BodyMorph",
        "request_id": request_id,
        "event": "body_profile_success",
        "body_type": final_response.get("body_type"),
        "confidence": final_response.get("confidence"),
        "latency_ms": total_latency_ms
    })

    return jsonify(final_response), 200


def _handle_engine_error(code: str, message: str, status: int, request_id: str, idem_key: str, request_hash: str):
    error_body = {
        "error": {
            "code": code,
            "message": message
        },
        "request_id": request_id
    }
    write_idempotency(idem_key, request_hash, {"body": error_body, "_status": status})
    current_app.logger.warning({
        "component": "BodyMorph",
        "request_id": request_id,
        "event": "body_profile_error",
        "error_code": code,
        "status": status
    })
    return jsonify(error_body), status
