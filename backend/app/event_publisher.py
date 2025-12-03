#This code provides a real-time event broadcasting system using 
# Server-Sent Events and keeps an external service (TaskPulseOS) notified about 
# workflow updates via API calls. It’s typically part of a monitoring dashboard or 
# workflow management system where users can watch live process updates in their browser.
import json
import requests
from datetime import datetime
from flask import Response
from threading import Lock
from backend.app.services.kafka_service import get_kafka_producer

clients = []
clients_lock = Lock()

TASKPULSEOS_URL = "https://taskpulseos.example.com/api/workflow-update"
producer = get_kafka_producer()
def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for {msg.topic()}: {err}")
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}")
def push_update(data: dict):
    """
    Push update to all SSE clients and TaskPulseOS.
    """
    data["timestamp"] = datetime.utcnow().isoformat()

    # Push to SSE clients
    with clients_lock:
        for q in clients:
            q.put(data)

    # Push to TaskPulseOS (async / best-effort)
    try:
        stepStatus = data.get("status")
        if stepStatus == "success":
            producer.produce(
                topic='step.succeeded',
                value=json.dumps(data).encode("utf-8"),
                callback=delivery_report
            )
            # Let producer serve delivery callbacks and flush buffer
            producer.poll(0)
        elif stepStatus == "failure":
            producer.produce(
                topic='step.failed',
                value=json.dumps(data).encode("utf-8"),
                callback=delivery_report
            )
            producer.poll(0) 
        elif stepStatus == "retry":
            producer.produce(
                topic='step.retrying',
                value=json.dumps(data).encode("utf-8"),
                callback=delivery_report
            )
            producer.poll(0)      
        elif stepStatus == "compensation":
            producer.produce(
                topic='step.compensating',
                value=json.dumps(data).encode("utf-8"),
                callback=delivery_report
            )
            producer.poll(0)     
        # requests.post(TASKPULSEOS_URL, json=data, timeout=1)
    except Exception as e:
        print(f"TaskPulseOS push failed: {e}")

def register_client(queue):
    with clients_lock:
        clients.append(queue)

def unregister_client(queue):
    with clients_lock:
        if queue in clients:
            clients.remove(queue)

def sse_stream(queue):
    while True:
        data = queue.get()
        yield f"data: {json.dumps(data)}\n\n"


# -------------------------
# SSE
# -------------------------
def sse_publish_event(run_id, event_type, data):
    """
    Publish an event to SSE clients.
    """
    # You can add run_id to the payload if needed
    data_with_run = data.copy()
    data_with_run["run_id"] = run_id
    data_with_run["timestamp"] = datetime.utcnow().isoformat()  # optional

    payload = f"event: {event_type}\ndata: {json.dumps(data_with_run)}\n\n"
    return Response(payload, mimetype='text/event-stream')

# -------------------------
# TaskPulseOS integration
# -------------------------
def push_taskpulseos_update(run_id, workflow_name, step_id, status):
    payload = {
        "run_id": run_id,
        "workflow_name": workflow_name,
        "step_id": step_id,
        "status": status
    }
    requests.post("https://taskpulseos.example.com/update", json=payload)