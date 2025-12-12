import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .exceptions import (
    BodyMorphError,
    ExtremePerspectiveError,
    HeavyOcclusionError,
    LowQualityError,
    MultiSubjectError,
    UnknownSilhouetteError,
)

# ------------------------------------------------------------------------------
# Validation / Bounds (aligned with API)
# ------------------------------------------------------------------------------
# Keep validation rules aligned with the API layer so the mock engine is safe to call directly.
ALLOWED_FORCE_FAILURES = {"MULTI_SUBJECT", "LOW_QUALITY", "HEAVY_OCCLUSION", "EXTREME_PERSPECTIVE", "UNKNOWN"}
MIN_HEIGHT_CM = 80
MAX_HEIGHT_CM = 260
MAX_SUBJECTS = 5
MIN_CAMERA_DISTANCE_M = 0.2
MAX_CAMERA_DISTANCE_M = 20.0
MIN_CAMERA_FOV = 1.0
MAX_CAMERA_FOV = 180.0
MAX_TILT_DEG = 90.0


class BodyMorphEngine:
    """
    Mocked BodyMorph engine that simulates a full pipeline:
      person detection -> pose estimation -> segmentation -> proportions ->
      silhouette classification -> posture heuristics.

    This engine is deterministic for the same (image_bytes, hints, camera) inputs.
    It is designed to be safe to call directly (e.g., tests/dev) by validating inputs
    and guarding output shape.
    """

    def __init__(self, seed_salt: str = "bodymorph"):
        """
        Args:
          seed_salt: additional salt used to generate deterministic per-input seeds.
                     Changing this will change outputs for the same inputs.
        """
        self.seed_salt = seed_salt

    def run(
        self,
        image_bytes: bytes,
        hints: Optional[Dict[str, Any]],
        camera: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run the mocked BodyMorph pipeline with guardrails for inputs and outputs.

        Args:
          image_bytes: raw bytes of the image (must be non-empty bytes/bytearray)
          hints: optional hint metadata (validated/normalized)
          camera: optional camera metadata (validated/normalized)

        Returns:
          (result, trace)
            - result: dict matching response contract
            - trace: list of per-stage timing dicts

        Raises:
          MultiSubjectError: when multiple subjects detected (hard failure)
          UnknownSilhouetteError: when classification confidence is too low
          BodyMorphError / LowQualityError: on invalid inputs or internal contract violations
        """
        image_bytes, hints, camera = self._validate_inputs(image_bytes, hints, camera)

        start = time.time()
        rng_seed = self._seed(image_bytes, hints, camera)

        trace: List[Dict[str, Any]] = []

        # ----------------------------------------------------------------------
        # Person detection / quality gate
        # ----------------------------------------------------------------------
        detect_start = time.time()
        quality = self._simulate_quality(rng_seed, hints, camera)
        trace.append({"stage": "detect_person", "ms": int((time.time() - detect_start) * 1000)})

        # May raise MultiSubjectError; otherwise returns degrade reason
        degrade_reason = self._guard_quality(quality)

        # ----------------------------------------------------------------------
        # Pose estimation and segmentation (mocked landmarks)
        # ----------------------------------------------------------------------
        pose_start = time.time()
        landmarks = self._mock_landmarks(rng_seed)
        trace.append({"stage": "pose_estimate", "ms": int((time.time() - pose_start) * 1000)})

        # ----------------------------------------------------------------------
        # Proportion estimation and silhouette classification
        # ----------------------------------------------------------------------
        props_start = time.time()
        proportions = self._estimate_proportions(rng_seed, hints)
        normalized = self._normalize(proportions)
        body_type, confidence = self._classify_silhouette(normalized, quality)
        trace.append({"stage": "classify_silhouette", "ms": int((time.time() - props_start) * 1000)})

        # Graceful degradation: keep processing but mark results as unknown with low confidence
        if degrade_reason:
            body_type = "unknown"
            confidence = min(confidence, 0.35 if degrade_reason == "heavy_occlusion" else 0.3)

        # If classification is too weak and we didn't already degrade to unknown, fail as unknown silhouette
        if confidence < 0.4 and body_type != "unknown":
            raise UnknownSilhouetteError("Unable to confidently classify silhouette.")

        # ----------------------------------------------------------------------
        # Posture heuristics (mock)
        # ----------------------------------------------------------------------
        posture = self._posture_estimate(rng_seed, quality)

        # Final result contract
        result = {
            "body_type": body_type,
            "confidence": round(confidence, 2),
            "landmarks": landmarks,
            "proportions": proportions,
            "normalized": normalized,
            "posture": posture,
            "occlusion": {
                "percent": round(quality["occlusion"], 2),
                "regions": quality.get("occluded_regions", []),
            },
            "quality_flags": self._quality_flags(quality, degrade_reason),
        }

        # Total engine elapsed (mock of combined pipeline)
        trace.append({"stage": "engine_total", "ms": int((time.time() - start) * 1000)})

        # Output guardrail
        self._validate_output_shape(result)

        return result, trace

    # --------------------------------------------------------------------------
    # Input validation / normalization
    # --------------------------------------------------------------------------
    def _validate_inputs(
        self,
        image_bytes: Optional[bytes],
        hints: Optional[Dict[str, Any]],
        camera: Optional[Dict[str, Any]],
    ) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
        """
        Validate and normalize inputs so downstream steps can assume shape and bounds.
        """
        # Image bytes must exist and be bytes-like
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            # LowQualityError is used here to align with API behavior for bad/empty inputs
            raise LowQualityError("image_bytes must be non-empty bytes.")
        if isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)

        # hints must be dict-like
        hints = hints or {}
        if not isinstance(hints, dict):
            raise BodyMorphError("hints must be an object.")
        hints = dict(hints)

        # camera must be dict-like
        camera = camera or {}
        if not isinstance(camera, dict):
            raise BodyMorphError("camera must be an object.")
        camera = dict(camera)

        # --------------------------
        # Clamp and validate hint fields
        # --------------------------
        height_cm = hints.get("height_cm")
        if height_cm is not None:
            if not isinstance(height_cm, (int, float)) or not (MIN_HEIGHT_CM <= float(height_cm) <= MAX_HEIGHT_CM):
                raise BodyMorphError(f"height_cm must be between {MIN_HEIGHT_CM} and {MAX_HEIGHT_CM}.")
            hints["height_cm"] = float(height_cm)

        subject_count = hints.get("subject_count")
        if subject_count is not None:
            if not isinstance(subject_count, int) or not (1 <= subject_count <= MAX_SUBJECTS):
                raise BodyMorphError(f"subject_count must be an integer between 1 and {MAX_SUBJECTS}.")

        # --------------------------
        # Clamp and validate camera fields
        # --------------------------
        fov = camera.get("fov_deg")
        if fov is not None:
            if not isinstance(fov, (int, float)) or not (MIN_CAMERA_FOV <= float(fov) <= MAX_CAMERA_FOV):
                raise BodyMorphError(f"fov_deg must be between {MIN_CAMERA_FOV} and {MAX_CAMERA_FOV}.")
            camera["fov_deg"] = float(fov)

        distance = camera.get("distance_m")
        if distance is not None:
            if not isinstance(distance, (int, float)) or not (MIN_CAMERA_DISTANCE_M <= float(distance) <= MAX_CAMERA_DISTANCE_M):
                raise BodyMorphError(f"distance_m must be between {MIN_CAMERA_DISTANCE_M} and {MAX_CAMERA_DISTANCE_M} meters.")
            camera["distance_m"] = float(distance)

        tilt = camera.get("tilt_deg")
        if tilt is not None:
            if not isinstance(tilt, (int, float)) or not (-MAX_TILT_DEG <= float(tilt) <= MAX_TILT_DEG):
                raise BodyMorphError(f"tilt_deg must be between {-MAX_TILT_DEG} and {MAX_TILT_DEG}.")
            camera["tilt_deg"] = float(tilt)

        # --------------------------
        # Normalize force_failure knobs used by tests
        # --------------------------
        for container, field in ((hints, "force_failure"), (camera, "force_failure")):
            normalized = self._normalize_force_failure(container.get(field))
            if normalized:
                container[field] = normalized

        return image_bytes, hints, camera

    def _normalize_force_failure(self, value: Any) -> Optional[str]:
        """
        Normalize and validate `force_failure` used to force deterministic failure modes.
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

    # --------------------------------------------------------------------------
    # Determinism helpers
    # --------------------------------------------------------------------------
    def _seed(self, image_bytes: bytes, hints: Dict[str, Any], camera: Dict[str, Any]) -> str:
        """
        Produce a deterministic hash seed from image_bytes + normalized hints/camera + seed_salt.
        """
        payload = image_bytes + json.dumps(hints or {}, sort_keys=True).encode("utf-8")
        payload += json.dumps(camera or {}, sort_keys=True).encode("utf-8")
        payload += self.seed_salt.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _hash_float(self, seed: str, tag: str, min_v: float = 0.0, max_v: float = 1.0) -> float:
        """
        Map (seed, tag) to a deterministic float in [min_v, max_v].
        """
        digest = hashlib.sha256(f"{seed}:{tag}".encode("utf-8")).hexdigest()
        val = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
        return min_v + (max_v - min_v) * val

    # --------------------------------------------------------------------------
    # Mock pipeline stages
    # --------------------------------------------------------------------------
    def _simulate_quality(self, seed: str, hints: Dict[str, Any], camera: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate basic detection/quality outcomes:
          - subjects count (from hint or forced failure)
          - occlusion percent
          - overall quality score
          - perspective classification from camera metadata

        A forced failure (hints/camera force_failure) overrides the simulated values to make tests deterministic.
        """
        subjects = hints.get("subject_count") or 1
        if subjects < 1:
            subjects = 1

        occlusion = self._hash_float(seed, "occlusion", 0.08, 0.48)
        quality_score = self._hash_float(seed, "quality", 0.35, 0.98)

        # Camera-derived perspective check (simple heuristic)
        perspective = "normal"
        if abs(camera.get("tilt_deg", 0)) > 20 or camera.get("distance_m", 1.2) < 0.6:
            perspective = "extreme"
        if camera.get("fov_deg") and camera["fov_deg"] > 110:
            perspective = "extreme"

        # Allow explicit forcing of conditions for testing/QA
        forced = (hints or {}).get("force_failure") or camera.get("force_failure")
        if forced == "MULTI_SUBJECT":
            subjects = 2
        if forced == "LOW_QUALITY":
            quality_score = 0.2
        if forced == "HEAVY_OCCLUSION":
            occlusion = 0.6
        if forced == "EXTREME_PERSPECTIVE":
            perspective = "extreme"
        if forced == "UNKNOWN":
            quality_score = 0.4

        occluded_regions = ["lower_arm_R"] if occlusion > 0.25 else []
        return {
            "subjects": subjects,
            "occlusion": occlusion,
            "quality_score": quality_score,
            "perspective": perspective,
            "occluded_regions": occluded_regions,
            "forced_failure": forced,
        }

    def _guard_quality(self, quality: Dict[str, Any]) -> Optional[str]:
        """
        Enforce hard failure on multi-subject; otherwise return a degrade reason so the mock can emit an unknown result.

        Returns:
          degrade_reason: one of {"low_quality", "heavy_occlusion", "extreme_perspective"} or None

        Raises:
          MultiSubjectError: if more than one subject is detected
        """
        if quality["subjects"] > 1:
            raise MultiSubjectError("Multiple people detected; provide single-subject image.")

        degrade_reason = None
        if quality["quality_score"] < 0.3:
            degrade_reason = "low_quality"
        if quality["occlusion"] > 0.5:
            degrade_reason = degrade_reason or "heavy_occlusion"
        if quality["perspective"] == "extreme":
            degrade_reason = degrade_reason or "extreme_perspective"
        return degrade_reason

    def _mock_landmarks(self, seed: str) -> Dict[str, List[int]]:
        """
        Produce deterministic pseudo-landmarks.
        """
        base = int(self._hash_float(seed, "base_landmark", 400, 520))
        return {
            "shoulder_L": [base, base + 4],
            "shoulder_R": [base + 14, base + 6],
            "hip_L": [base - 10, base + 80],
            "hip_R": [base + 30, base + 78],
            "waist": [base + 5, base + 50],
            "knee_L": [base - 20, base + 220],
            "ankle_L": [base - 18, base + 320],
        }

    def _estimate_proportions(self, seed: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate deterministic pseudo-proportions, optionally scaled by height_cm.
        """
        shoulder_width = int(self._hash_float(seed, "shoulder_width", 380, 460))
        hip_width = int(self._hash_float(seed, "hip_width", 360, 450))
        waist_width = int(self._hash_float(seed, "waist_width", 290, 360))
        torso_len = int(self._hash_float(seed, "torso_len", 620, 720))
        leg_len = int(self._hash_float(seed, "leg_len", 840, 960))

        # Apply small deterministic adjustments from hints to keep behavior repeatable
        height_cm = hints.get("height_cm")
        if isinstance(height_cm, (int, float)):
            scale = max(1.0, min(1.2, height_cm / 170.0))
            shoulder_width = int(shoulder_width * scale)
            hip_width = int(hip_width * scale)
            waist_width = int(waist_width * scale)
            torso_len = int(torso_len * scale)
            leg_len = int(leg_len * scale)

        return {
            "shoulder_width_px": shoulder_width,
            "hip_width_px": hip_width,
            "waist_width_px": waist_width,
            "torso_len_px": torso_len,
            "leg_len_px": leg_len,
        }

    def _normalize(self, proportions: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert absolute pseudo-proportions into normalized ratios used by the classifier.
        """
        shoulder = proportions["shoulder_width_px"]
        hip = max(1, proportions["hip_width_px"])
        waist = max(1, proportions["waist_width_px"])
        torso = proportions["torso_len_px"]
        leg = max(1, proportions["leg_len_px"])

        return {
            "shoulder_to_hip_ratio": round(shoulder / hip, 2),
            "waist_to_hip_ratio": round(waist / hip, 2),
            "torso_to_leg_ratio": round(torso / leg, 2),
        }

    def _classify_silhouette(self, normalized: Dict[str, float], quality: Dict[str, Any]) -> Tuple[str, float]:
        """
        Rule-based buckets aligned to v1 spec:
          - hourglass
          - triangle
          - inverted
          - rectangle
          - athletic
          - unknown (on forced failure)

        Returns:
          (label, confidence)
        """
        if quality.get("forced_failure") == "UNKNOWN":
            return "unknown", 0.2

        s_to_h = normalized["shoulder_to_hip_ratio"]
        w_to_h = normalized["waist_to_hip_ratio"]
        t_to_l = normalized["torso_to_leg_ratio"]

        abs_s_h = abs(s_to_h - 1.0)

        if abs_s_h <= 0.08 and w_to_h <= 0.80:
            label = "hourglass"
        elif s_to_h < 0.92 and w_to_h <= 0.88:
            label = "triangle"
        elif s_to_h > 1.08 and w_to_h <= 0.90:
            label = "inverted"
        elif w_to_h >= 0.88 and abs_s_h <= 0.10:
            label = "rectangle"
        elif s_to_h >= 1.02 and t_to_l <= 0.80:
            label = "athletic"
        else:
            label = "rectangle"

        # Confidence reflects quality, occlusion, and closeness to ideal ratios
        shape_score = 1.0 - abs(1.0 - s_to_h)
        waist_score = 1.0 - abs(0.78 - w_to_h)
        quality_score = quality["quality_score"]
        occlusion_penalty = 0.25 * quality["occlusion"]

        confidence = max(0.0, min(1.0, 0.45 + 0.35 * quality_score + 0.15 * shape_score + 0.05 * waist_score))
        confidence -= occlusion_penalty

        return label, confidence

    def _posture_estimate(self, seed: str, quality: Dict[str, Any]) -> Dict[str, str]:
        """
        Mock posture estimation based on deterministic hashes and camera perspective.
        """
        tilt_score = self._hash_float(seed, "tilt", -3, 3)
        slouch_score = self._hash_float(seed, "slouch", 0, 1)
        stance_score = self._hash_float(seed, "stance", 0, 1)

        tilt = "neutral"
        if tilt_score > 1:
            tilt = "right"
        elif tilt_score < -1:
            tilt = "left"

        slouch = "low"
        if slouch_score > 0.66:
            slouch = "high"
        elif slouch_score > 0.33:
            slouch = "medium"

        stance = "closed" if stance_score < 0.4 else "open"
        if quality["perspective"] == "extreme":
            stance = "distorted"

        return {"tilt": tilt, "slouch": slouch, "stance": stance}

    def _quality_flags(self, quality: Dict[str, Any], degrade_reason: Optional[str]) -> List[str]:
        """
        Produce human-readable quality flags for clients/QA.
        """
        flags = []
        flags.append("single_subject" if quality["subjects"] == 1 else "multi_subject")
        flags.append("front_view_detected")
        flags.append("lighting_ok" if quality["quality_score"] > 0.5 else "lighting_low")
        if quality["occlusion"] > 0.25:
            flags.append("occlusion_warning")
        if quality["perspective"] == "extreme":
            flags.append("extreme_perspective")
        if degrade_reason:
            flags.append(degrade_reason)
        return flags

    # --------------------------------------------------------------------------
    # Output validation / guardrails
    # --------------------------------------------------------------------------
    def _validate_output_shape(self, engine_result: Dict[str, Any]) -> None:
        """
        Ensure the mock engine never emits structurally invalid payloads.

        This is intentionally strict so callers can rely on response shape.
        It also clamps probability-like fields to [0, 1].
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

        # Clamp confidence to [0, 1]
        engine_result["confidence"] = max(0.0, min(1.0, float(engine_result.get("confidence", 0.0))))

        # Clamp occlusion percent to [0, 1] if present
        occlusion = engine_result.get("occlusion", {})
        percent = occlusion.get("percent")
        if isinstance(percent, (int, float)):
            occlusion["percent"] = max(0.0, min(1.0, float(percent)))
            engine_result["occlusion"] = occlusion
