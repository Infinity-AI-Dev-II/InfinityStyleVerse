# backend/app/sse.py
from flask import Blueprint, Response
import redis
import json

# Use a distinct blueprint name to avoid clashing with flask_sse's built-in "sse" blueprint
sse_bp = Blueprint("workflow_sse", __name__)
r = redis.Redis(host='localhost', port=6379, db=0)

def event_stream(run_id):
    pubsub = r.pubsub()
    pubsub.subscribe(f"workflow_{run_id}")
    for message in pubsub.listen():
        if message['type'] == 'message':
            yield f"data: {message['data'].decode()}\n\n"

@sse_bp.route("/stream/<int:run_id>")
def stream(run_id):
    return Response(event_stream(run_id), mimetype="text/event-stream")
