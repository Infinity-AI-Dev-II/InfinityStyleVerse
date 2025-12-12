from typing import Any, Dict, Optional, Tuple, List

from .engine import BodyMorphEngine
from .exceptions import BodyMorphError

# ------------------------------------------------------------------------------
# Validation Bounds (kept in sync with API)
# ------------------------------------------------------------------------------
# Mirror validation bounds from the API layer to keep the mock predictable.
ALLOWED_FORCE_FAILURES = {"MULTI_SUBJECT", "LOW_QUALITY", "HEAVY_OCCLUSION", "EXTREME_PERSPECTIVE", "UNKNOWN"}
MIN_HEIGHT_CM = 80
MAX_HEIGHT_CM = 260
MAX_SUBJECTS = 5
MIN_CAMERA_DISTANCE_M = 0.2
MAX_CAMERA_DISTANCE_M = 20.0
MIN_CAMERA_FOV = 1.0
MAX_CAMERA_FOV = 180.0
MAX_TILT_DEG = 90.0


# ------------------------------------------------------------------------------
# Normalization / Validation Helpers
# ------------------------------------------------------------------------------
def _normalize_force_failure(value: Any) -> Optional[str]:
    """
    Normalize and validate the optional `force_failure` knob.

    This is used to force deterministic failures from the mock engine during tests/dev.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise BodyMorphError("force_failure must be a string if provided.")

    normalized = value.upper()
    if normalized not in ALLOWED_FORCE_FAILURES:
        allowed = ", ".join(sorted(ALLOWED_FORCE_FAILURES))
        raise BodyMorphError(f"force_failure must be one of: {allowed}")
    return normalized


def _validate_hints(raw_hints: Any) -> Dict[str, Any]:
    """
    Validate and normalize `hints`.

    Expected schema:
      - height_cm: number in [MIN_HEIGHT_CM, MAX_HEIGHT_CM]
      - subject_count: int in [1, MAX_SUBJECTS]
      - force_failure: one of ALLOWED_FORCE_FAILURES (case-insensitive)

    Returns:
      A normalized hints dict (may be empty).
    """
    if not raw_hints:
        return {}
    if not isinstance(raw_hints, dict):
        raise BodyMorphError("hints must be an object.")

    hints = dict(raw_hints)

    # height_cm: clamp by bounds and normalize to float
    height_cm = hints.get("height_cm")
    if height_cm is not None:
        if not isinstance(height_cm, (int, float)) or not (MIN_HEIGHT_CM <= float(height_cm) <= MAX_HEIGHT_CM):
            raise BodyMorphError(f"height_cm must be between {MIN_HEIGHT_CM} and {MAX_HEIGHT_CM}.")
        hints["height_cm"] = float(height_cm)

    # subject_count: bounded integer
    subject_count = hints.get("subject_count")
    if subject_count is not None:
        if not isinstance(subject_count, int) or not (1 <= subject_count <= MAX_SUBJECTS):
            raise BodyMorphError(f"subject_count must be an integer between 1 and {MAX_SUBJECTS}.")

    # force_failure: optional enum
    force_failure = _normalize_force_failure(hints.get("force_failure"))
    if force_failure:
        hints["force_failure"] = force_failure

    return hints


def _validate_camera(raw_camera: Any) -> Dict[str, Any]:
    """
    Validate and normalize `camera`.

    Expected schema:
      - fov_deg: number in [MIN_CAMERA_FOV, MAX_CAMERA_FOV]
      - distance_m: number in [MIN_CAMERA_DISTANCE_M, MAX_CAMERA_DISTANCE_M]
      - tilt_deg: number in [-MAX_TILT_DEG, MAX_TILT_DEG]
      - force_failure: one of ALLOWED_FORCE_FAILURES (case-insensitive)

    Returns:
      A normalized camera dict (may be empty).
    """
    if not raw_camera:
        return {}
    if not isinstance(raw_camera, dict):
        raise BodyMorphError("camera must be an object.")

    camera = dict(raw_camera)

    # fov_deg: bounded float
    fov = camera.get("fov_deg")
    if fov is not None:
        if not isinstance(fov, (int, float)) or not (MIN_CAMERA_FOV <= float(fov) <= MAX_CAMERA_FOV):
            raise BodyMorphError(f"fov_deg must be between {MIN_CAMERA_FOV} and {MAX_CAMERA_FOV}.")
        camera["fov_deg"] = float(fov)

    # distance_m: bounded float
    distance = camera.get("distance_m")
    if distance is not None:
        if not isinstance(distance, (int, float)) or not (MIN_CAMERA_DISTANCE_M <= float(distance) <= MAX_CAMERA_DISTANCE_M):
            raise BodyMorphError(f"distance_m must be between {MIN_CAMERA_DISTANCE_M} and {MAX_CAMERA_DISTANCE_M} meters.")
        camera["distance_m"] = float(distance)

    # tilt_deg: bounded float (can be negative)
    tilt = camera.get("tilt_deg")
    if tilt is not None:
        if not isinstance(tilt, (int, float)) or not (-MAX_TILT_DEG <= float(tilt) <= MAX_TILT_DEG):
            raise BodyMorphError(f"tilt_deg must be between {-MAX_TILT_DEG} and {MAX_TILT_DEG}.")
        camera["tilt_deg"] = float(tilt)

    # force_failure: optional enum
    force_failure = _normalize_force_failure(camera.get("force_failure"))
    if force_failure:
        camera["force_failure"] = force_failure

    return camera


def _validate_output_shape(engine_result: Dict[str, Any]) -> None:
    """
    Guard mocked results before returning them to tests or callers.

    Ensures required keys exist and types are correct, and clamps probability-like
    values into [0, 1] to keep callers/tests stable.
    """
    required_fields = {"body_type", "confidence", "landmarks", "proportions", "normalized", "posture", "occlusion", "quality_flags"}
    missing = required_fields - set(engine_result.keys())
    if missing:
        raise BodyMorphError(f"Engine output missing fields: {', '.join(sorted(missing))}")

    if not isinstance(engine_result.get("confidence"), (int, float)):
        raise BodyMorphError("confidence must be numeric.")
    if not isinstance(engine_result.get("quality_flags"), list):
        raise BodyMorphError("quality_flags must be a list.")
    if not isinstance(engine_result.get("landmarks"), dict):
        raise BodyMorphError("landmarks must be an object.")
    if not isinstance(engine_result.get("occlusion"), dict):
        raise BodyMorphError("occlusion must be an object.")

    # Normalize / clamp confidence
    engine_result["confidence"] = max(0.0, min(1.0, float(engine_result.get("confidence", 0.0))))

    # Normalize / clamp occlusion percent if present
    occlusion = engine_result.get("occlusion", {})
    percent = occlusion.get("percent")
    if isinstance(percent, (int, float)):
        occlusion["percent"] = max(0.0, min(1.0, float(percent)))
        engine_result["occlusion"] = occlusion


# ------------------------------------------------------------------------------
# Public Utility Entrypoint
# ------------------------------------------------------------------------------
def run_mock(
    image_bytes: bytes,
    hints: Optional[Dict[str, Any]] = None,
    camera: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Utility entrypoint for tests/dev to run the BodyMorphEngine with strict validation.

    Args:
      image_bytes:
        Raw bytes of the image input (must be non-empty). Accepts bytes or bytearray.
      hints:
        Optional dict of hints (validated and normalized via _validate_hints).
      camera:
        Optional dict of camera metadata (validated and normalized via _validate_camera).

    Returns:
      (result, trace)
        - result: validated engine output dict
        - trace: list of trace dicts emitted by the engine

    Raises:
      BodyMorphError:
        - if inputs are invalid
        - if engine output contract is violated
    """
    # Validate image bytes
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise BodyMorphError("image_bytes must be non-empty bytes.")
    if isinstance(image_bytes, bytearray):
        image_bytes = bytes(image_bytes)

    # Validate/normalize optional inputs
    validated_hints = _validate_hints(hints)
    validated_camera = _validate_camera(camera)

    # Run engine and validate its output contract
    engine = BodyMorphEngine()
    result, trace = engine.run(image_bytes, validated_hints, validated_camera)
    _validate_output_shape(result)

    return result, trace
