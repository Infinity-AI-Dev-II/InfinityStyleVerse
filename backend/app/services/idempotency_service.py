import json
from sqlalchemy.exc import IntegrityError

from ..models.echo_models import IdempotencyKey
from ..database import db

# Local in-memory fallback when DB is unavailable or table is missing.
_LOCAL_IDEMP_CACHE = {}


class _LocalRow:
    def __init__(self, key: str, request_hash: str, response_json: str):
        self.key = key
        self.request_hash = request_hash
        self.response_json = response_json


def read_idempotency(key: str):
    if not key:
        return None
    try:
        return IdempotencyKey.query.get(key)
    except Exception:
        # Fallback to in-memory store if DB/table is unavailable.
        return _LOCAL_IDEMP_CACHE.get(key)


def write_idempotency(key: str, request_hash: str, response_json: dict):
    if not key:
        return
    payload = json.dumps(response_json)

    # Try DB first; on failure, use in-memory cache.
    try:
        rec = IdempotencyKey(
            key=key,
            request_hash=request_hash,
            response_json=payload
        )
        db.session.add(rec)
        db.session.commit()
        _LOCAL_IDEMP_CACHE[key] = rec
        return
    except IntegrityError:
        db.session.rollback()
        return
    except Exception:
        # Use fallback cache
        _LOCAL_IDEMP_CACHE[key] = _LocalRow(key, request_hash, payload)
        return


def compute_request_hash(body: dict):
    return json.dumps(body, sort_keys=True)
