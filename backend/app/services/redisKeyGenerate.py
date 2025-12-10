#this creates a redis key to store response data for a given image and hints (in bodyMorph routes)
import hashlib
import json


def generate_image_cache_key(image_bytes: bytes, hints: dict) -> str:
    """
    Generate a deterministic Redis key that ties together the raw image bytes
    and the provided hints. Spec requires sha256(image_bytes + hints_json).
    """
    hints_json = json.dumps(hints or {}, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(image_bytes + b"|hints|" + hints_json).hexdigest()
    return f"imgcache:{digest}"
