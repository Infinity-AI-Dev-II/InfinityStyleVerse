from flask import Blueprint, Response
from queue import SimpleQueue
from backend.app.event_publisher import register_client, unregister_client, sse_stream

sse_bp = Blueprint("sse", __name__)

@sse_bp.route("/events/stream")
def stream_events():
    q = SimpleQueue()
    register_client(q)
    try:
        return Response(sse_stream(q), mimetype="text/event-stream")
    finally:
        unregister_client(q)
