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


class BodyMorphEngine:
    """
    Mocked BodyMorph engine that simulates a full pipeline:
    person detection -> pose estimation -> segmentation -> proportions ->
    silhouette classification -> posture heuristics.
    """

    def __init__(self, seed_salt: str = "bodymorph"):
        self.seed_salt = seed_salt

    def run(
        self, image_bytes: bytes, hints: Optional[Dict[str, Any]], camera: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not image_bytes:
            raise LowQualityError("Empty image payload provided.")

        hints = hints or {}
        camera = camera or {}
        start = time.time()
        rng_seed = self._seed(image_bytes, hints, camera)

        trace: List[Dict[str, Any]] = []

        # Person detection / quality gate
        detect_start = time.time()
        quality = self._simulate_quality(rng_seed, hints, camera)
        trace.append({"stage": "detect_person", "ms": int((time.time() - detect_start) * 1000)})
        self._guard_quality(quality)

        # Pose estimation and segmentation (mocked landmarks)
        pose_start = time.time()
        landmarks = self._mock_landmarks(rng_seed)
        trace.append({"stage": "pose_estimate", "ms": int((time.time() - pose_start) * 1000)})

        # Proportion estimation and silhouette classification
        props_start = time.time()
        proportions = self._estimate_proportions(rng_seed, hints)
        normalized = self._normalize(proportions)
        body_type, confidence = self._classify_silhouette(normalized, quality)
        trace.append({"stage": "classify_silhouette", "ms": int((time.time() - props_start) * 1000)})

        if confidence < 0.4:
            raise UnknownSilhouetteError("Unable to confidently classify silhouette.")

        posture = self._posture_estimate(rng_seed, quality)
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
            "quality_flags": self._quality_flags(quality),
        }

        # Total engine elapsed (mock of combined pipeline)
        trace.append({"stage": "engine_total", "ms": int((time.time() - start) * 1000)})
        return result, trace

    # ------------------------------
    # Helpers
    # ------------------------------
    def _seed(self, image_bytes: bytes, hints: Dict[str, Any], camera: Dict[str, Any]) -> str:
        payload = image_bytes + json.dumps(hints or {}, sort_keys=True).encode("utf-8")
        payload += json.dumps(camera or {}, sort_keys=True).encode("utf-8")
        payload += self.seed_salt.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _hash_float(self, seed: str, tag: str, min_v: float = 0.0, max_v: float = 1.0) -> float:
        digest = hashlib.sha256(f"{seed}:{tag}".encode("utf-8")).hexdigest()
        val = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
        return min_v + (max_v - min_v) * val

    def _simulate_quality(self, seed: str, hints: Dict[str, Any], camera: Dict[str, Any]) -> Dict[str, Any]:
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

    def _guard_quality(self, quality: Dict[str, Any]) -> None:
        if quality["subjects"] > 1:
            raise MultiSubjectError("Multiple people detected; provide single-subject image.")
        if quality["quality_score"] < 0.3:
            raise LowQualityError("Image quality too low for reliable body analysis.")
        if quality["occlusion"] > 0.5:
            raise HeavyOcclusionError("Too much occlusion to estimate body shape.")
        if quality["perspective"] == "extreme":
            raise ExtremePerspectiveError("Extreme camera perspective detected; capture a straight-on view.")

    def _mock_landmarks(self, seed: str) -> Dict[str, List[int]]:
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
        if quality.get("forced_failure") == "UNKNOWN":
            return "unknown", 0.2

        s_to_h = normalized["shoulder_to_hip_ratio"]
        w_to_h = normalized["waist_to_hip_ratio"]

        if 0.95 <= s_to_h <= 1.08 and w_to_h <= 0.8:
            label = "hourglass"
        elif s_to_h > 1.12 and w_to_h >= 0.75:
            label = "inverted_triangle"
        elif s_to_h < 0.88 and w_to_h >= 0.8:
            label = "triangle"
        elif w_to_h >= 0.9 and 0.9 <= s_to_h <= 1.1:
            label = "rectangle"
        else:
            label = "oval"

        # Confidence reflects quality, occlusion, and closeness to ideal ratios
        shape_score = 1.0 - abs(1.0 - s_to_h)
        waist_score = 1.0 - abs(0.78 - w_to_h)
        quality_score = quality["quality_score"]
        occlusion_penalty = 0.25 * quality["occlusion"]
        confidence = max(0.0, min(1.0, 0.45 + 0.35 * quality_score + 0.15 * shape_score + 0.05 * waist_score))
        confidence -= occlusion_penalty
        return label, confidence

    def _posture_estimate(self, seed: str, quality: Dict[str, Any]) -> Dict[str, str]:
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

    def _quality_flags(self, quality: Dict[str, Any]) -> List[str]:
        flags = []
        flags.append("single_subject" if quality["subjects"] == 1 else "multi_subject")
        flags.append("front_view_detected")
        flags.append("lighting_ok" if quality["quality_score"] > 0.5 else "lighting_low")
        if quality["occlusion"] > 0.25:
            flags.append("occlusion_warning")
        if quality["perspective"] == "extreme":
            flags.append("extreme_perspective")
        return flags
