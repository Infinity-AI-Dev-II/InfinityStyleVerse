import pytest
import json
from threading import Thread
import time
from flask import Flask
from backend.app.sse import sse_bp, event_stream
from backend.app.event_publisher import publish_step_event
import redis

# Use a test Redis database
r = redis.Redis(host='localhost', port=6379, db=1)

@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(sse_bp)
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

def test_event_stream_sse(client):
    run_id = 999
    events_received = []

    # Thread to simulate subscribing to SSE
    def listen_events():
        for event in event_stream(run_id):
            events_received.append(json.loads(event.split("data: ")[1]))
            if len(events_received) >= 3:
                break

    listener = Thread(target=listen_events)
    listener.start()

    # Simulate workflow step events
    publish_step_event(run_id, "start", "completed")
    publish_step_event(run_id, "task1", "completed")
    publish_step_event(run_id, "end", "completed")

    listener.join(timeout=2)

    assert len(events_received) == 3
    step_ids = [e['step_id'] for e in events_received]
    assert "start" in step_ids
    assert "task1" in step_ids
    assert "end" in step_ids

def test_taskpulseos_push(monkeypatch):
    run_id = 1000
    pushed_events = []

    # Mock requests.post to TaskPulseOS
    def fake_post(url, json, timeout):
        pushed_events.append(json)
        return type("Response", (), {"status_code": 200})()

    monkeypatch.setattr("backend.app.event_publisher.requests.post", fake_post)

    publish_step_event(run_id, "taskA", "completed")
    publish_step_event(run_id, "taskB", "failed")

    assert len(pushed_events) == 2
    statuses = [e["status"] for e in pushed_events]
    assert "completed" in statuses
    assert "failed" in statuses
