import json
from ..models.echo_models import IdempotencyKey
from ..database import db
from sqlalchemy.exc import IntegrityError

def read_idempotency(key: str):
    if not key:
        return None
    return IdempotencyKey.query.get(key)

def write_idempotency(key: str, request_hash: str, response_json: dict):
    if not key:
        return
    rec = IdempotencyKey(
        key=key,
        request_hash=request_hash,
        response_json=json.dumps(response_json)
    )
    try:
        db.session.add(rec)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()

def compute_request_hash(body: dict):
    return json.dumps(body, sort_keys=True)
