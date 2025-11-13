import time
import uuid
from flask import g, request, current_app
from ..models.echo_models import RequestLog
from ..database import db
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

def record_request_log(response_status: int):
    try:
        rid = getattr(g, "request_id", None) or str(uuid.uuid4())
        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except Exception:
            pass

        start = getattr(g, "start_time", None)
        latency_ms = (time.time() - start) * 1000.0 if start else None

        rl = RequestLog(
            request_id=rid,
            user_id=int(user_id) if user_id else None,
            endpoint=request.path,
            method=request.method,
            status_code=response_status,
            latency_ms=latency_ms,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string[:255] if request.user_agent else None
        )
        db.session.add(rl)
        db.session.commit()
    except Exception as e:
        current_app.logger.debug("Failed to write request log: %s", e)
        db.session.rollback()
