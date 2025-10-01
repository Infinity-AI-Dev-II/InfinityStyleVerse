import pytest
from backend.app import create_app
from backend.app.database import db
from backend.app.models.user import User
from backend.app.models.workflow_defs import WorkflowDef
from backend.app.models.runs import Run
from backend.app.models.run_steps import RunStep
import json

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.app_context():
        db.create_all()
        yield app.test_client()

@pytest.fixture
def auth_headers(client):
    # Create a test user
    user = User(email="testuser@example.com")
    user.password = "testpass"  # sets password_hash
    db.session.add(user)
    db.session.commit()

    # Log in to get access token (adjust endpoint if needed)
    resp = client.post("/auth/login", json={
        "email": "testuser@example.com",
        "password": "testpass"
    })
    token = resp.get_json().get("access_token")
    return {"Authorization": f"Bearer {token}"}

def test_register_and_run_workflow(client, auth_headers):
    #  Register a workflow
    yaml_spec = """
    steps:
      - id: A
        type: task
      - id: B
        type: task
        depends_on: A
    """
    resp = client.post(
        "/flow/workflows/",
        json={"name": "test_wf", "dsl_yaml": yaml_spec},
        headers=auth_headers
    )
    assert resp.status_code == 201
    wf_id = resp.get_json()["id"]

    #  Start the workflow
    resp = client.post(f"/flow/workflows/{wf_id}/start", headers=auth_headers)
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["status"] == "running"  # initial status is "running"

    #  Verify Run and RunSteps in the DB
    with client.application.app_context():
        run = Run.query.filter_by(workflow_id=wf_id).first()
        assert run is not None
        assert run.status == "running"

        steps = RunStep.query.filter_by(run_id=run.id).all()
        assert len(steps) == 2
        step_ids = [s.step_id for s in steps]
        assert "A" in step_ids and "B" in step_ids

        #  Simulate step execution for test purposes
        for step in steps:
            step.status = "success"
            db.session.commit()

        # Update run status to success
        run.status = "success"
        db.session.commit()

        # Re-fetch and assert final status
        run = Run.query.get(run.id)
        steps = RunStep.query.filter_by(run_id=run.id).all()
        assert run.status == "success"
        for s in steps:
            assert s.status == "success"

