#this creates a redis key to store response data for a given image and hints (in bodyMorph routes)
import hashlib
import json

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def generate_image_cache_key(image_bytes: bytes, hints: dict) -> str:
    #Hash image bytes
    img_hash = sha256_bytes(image_bytes)

    #Deterministic hints hash (sort keys!)
    hints_json = json.dumps(hints, sort_keys=True).encode("utf-8")
    hints_hash = sha256_bytes(hints_json)

    #Build Redis key
    redis_key = f"imgcache:{img_hash}:{hints_hash}"

    return redis_key