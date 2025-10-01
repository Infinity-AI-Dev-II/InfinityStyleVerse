import pytest
from backend.app.services.executor import start_run_from_dag, run_workflow
from backend.app.models import WorkflowDef
from backend.app.database import db

@pytest.fixture
def workflow_instance():
    """Creates a minimal workflow in the database for testing."""
    workflow = WorkflowDef(
        name="Test Workflow",
        version="1.0",
        dsl_yaml="steps: []",  # minimal DSL to satisfy creation
        created_by="test_user"
    )
    db.session.add(workflow)
    db.session.commit()
    return workflow
# tests/test_workflow_endpoints.py
from backend.app.services.executor import run_workflow

def test_run_execution_and_completion(client, workflow_instance):
    wf = workflow_instance

    steps = [
        {"id": "step1", "action": "do_a"},
        {"id": "step2", "action": "do_b", "depends_on": ["step1"]}
    ]

    payload = {
        "workflow_id": wf.id,
        "version": wf.version,
        "dsl": {"steps": steps}  # endpoint expects steps inside 'dsl'
    }

    resp = client.post("/flow/run/start", json=payload)
    print(resp.status_code, resp.data)  # debug line
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "completed"


def test_signal_resume_step(client, workflow_instance):
    wf = workflow_instance

    steps = [{"id": "step1", "action": "wait_for_signal"}]

    payload = {
        "workflow_id": wf.id,
        "version": wf.version,
        "dsl": {"steps": steps},
    }

    resp = client.post("/flow/run/start", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] in ["pending", "waiting_for_signal"]

    # Use proper app context
    with client.application.app_context():
        from backend.app.services.executor import run_workflow
        run_workflow(data["run_id"])

    resp2 = client.get(f"/flow/flow/run/{data['run_id']}")
    print(resp2.status_code, resp2.data) 
    data2 = resp2.get_json()
    step_status = data2["steps"][0]["status"]
    assert step_status == "waiting_for_signal"


def test_step_failure_triggers_compensation(client, workflow_instance):
    wf = workflow_instance

    steps = [{"id": "step1", "action": "fail"}]

    payload = {
        "workflow_id": wf.id,
        "version": wf.version,
        "dsl": {"steps": steps},
    }

    resp = client.post("/flow/run/start", json=payload)
    data = resp.get_json()
    assert "run_id" in data, f"Run creation failed: {data}"

    # Use proper app context
    with client.application.app_context():
        from backend.app.services.executor import run_workflow
        run_workflow(data["run_id"])

    resp2 = client.get(f"/flow/flow/run/{data['run_id']}")
    print(resp2.status_code, resp2.data) 
    data2 = resp2.get_json()
    assert data2["steps"][0]["status"] == "failed"


def test_cancel_run(client, workflow_instance):
    wf = workflow_instance

    steps = [{"id": "step1", "action": "wait_for_signal"}]

    payload = {
        "workflow_id": wf.id,
        "version": wf.version,
        "dsl": {"steps": steps},
    }

    resp = client.post("/flow/run/start", json=payload)
    run_data = resp.get_json()
    assert "run_id" in run_data, f"Run creation failed: {run_data}"

    cancel_resp = client.post(f"/flow/run/{run_data['run_id']}/cancel")
    cancel_data = cancel_resp.get_json()
    assert cancel_data["status"] == "cancelled"
