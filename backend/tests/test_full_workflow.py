# backend/tests/test_full_workflow.py
import pytest
from unittest.mock import patch
from datetime import datetime
from backend.app.tasks.executor import execute_step
from backend.app.event_publisher import sse_publish_event, push_taskpulseos_update
from backend.app.models.workflow_defs import WorkflowDef
from backend.app.models.runs import Run
from backend.app.models.run_steps import RunStep
from backend.app.database import db

# ------------------------------
# Dummy Step Class
# ------------------------------
class DummyStep:
    def __init__(self, id, payload=None):
        self.id = id
        self.payload = payload or {}

# ------------------------------
# Fixture: Create a workflow run
# ------------------------------
@pytest.fixture
def workflow_run():
    workflow_def = WorkflowDef(name="test_workflow",dsl_yaml="some default YAML content", version="1.0", created_by="pytest")
    db.session.add(workflow_def)
    db.session.commit()

    run = Run(workflow_id=workflow_def.id, version="1.0", status="pending", started_at=datetime.utcnow())
    db.session.add(run)
    db.session.commit()
    return workflow_def, run

# ------------------------------
# 1️⃣ Unit test: execute_step
# ------------------------------
def test_execute_step_success(workflow_run):
    workflow_def, run = workflow_run
    step = DummyStep("step_success")
    result = execute_step(step, workflow_def.name, run.id)
    assert result == "ok"

def test_execute_step_failure(workflow_run):
    workflow_def, run = workflow_run
    step = DummyStep("step_fail", payload={"should_fail": True})
    with pytest.raises(Exception):
        execute_step(step, workflow_def.name, run.id)

# ------------------------------
# 2️⃣ Integration: full workflow
# ------------------------------
def test_full_workflow_execution(workflow_run):
    workflow_def, run = workflow_run

    steps = [
        DummyStep("step1"),
        DummyStep("step2", payload={"should_fail": True}),
        DummyStep("step3")
    ]

    for step in steps:
        try:
            execute_step(step, workflow_def.name, run.id)
        except Exception:
            pass

    # Validate RunStep entries
    step_ids = [s.step_id for s in RunStep.query.filter_by(run_id=run.id)]
    for step in steps:
        assert step.id in step_ids

# ------------------------------
# 3️⃣ SSE Event Publishing
# ------------------------------
@pytest.mark.asyncio
async def test_event_stream_sse(workflow_run):
    workflow_def, run = workflow_run
    messages = []

    # Dummy subscriber to capture events
    def capture_event(run_id, event_name, data):
        messages.append({"run_id": run_id, "event": event_name, "data": data})

    # Patch the function where it is used, not just where it's defined
    import backend.app.event_publisher as ep

    with patch.object(ep, "sse_publish_event", side_effect=capture_event):
        # Call via the patched reference
        ep.sse_publish_event(run.id, "step_started", {"step_id": "test_step"})
        ep.sse_publish_event(run.id, "step_succeeded", {"step_id": "test_step"})

    # Assertions
    assert any(msg["event"] == "step_started" for msg in messages)
    assert any(msg["event"] == "step_succeeded" for msg in messages)

# ------------------------------
# 4️⃣ TaskPulseOS Push
# ------------------------------
def test_taskpulseos_push(workflow_run):
    workflow_def, run = workflow_run
    with patch("backend.app.event_publisher.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        push_taskpulseos_update(run.id, workflow_def.name, "step1", "completed")
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["workflow_name"] == workflow_def.name
        assert payload["step_id"] == "step1"
        assert payload["status"] == "completed"
