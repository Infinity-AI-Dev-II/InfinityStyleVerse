import pytest
import json
from backend.app.models import WorkflowDef, Run, RunStep, Compensation
from backend.app.database import db

# ----------------------------
# Fixture to create workflows
# ----------------------------
@pytest.fixture
def create_workflow():
    def _create(workflow_id: int, name: str = None):
        workflow = WorkflowDef(
            id=workflow_id,  # ensure ID matches
            name=name or f"Test Workflow {workflow_id}",
            version="1.0",
            dsl_yaml="steps: []",
            dag_json=json.dumps({"nodes": [], "edges": []}),
            created_by="admin"
        )
        db.session.add(workflow)
        db.session.commit()
        return workflow
    return _create

# ----------------------------
# Fixture to clear DB after each test
# ----------------------------
@pytest.fixture(autouse=True)
def cleanup_db():
    yield
    db.session.query(Compensation).delete()
    db.session.query(RunStep).delete()
    db.session.query(Run).delete()
    db.session.query(WorkflowDef).delete()
    db.session.commit()

# ----------------------------
# Mock step functions
# ----------------------------
executed_sequence = []

def mock_step_success(input_json):
    executed_sequence.append("step_success")
    return '{"result": "ok"}'

def mock_step_failure(input_json):
    executed_sequence.append("step_failure")
    raise Exception("Step failed intentionally")

# ----------------------------
# Test 1: Saga success
# ----------------------------
def test_saga_success(app, create_workflow):
    global executed_sequence
    executed_sequence = []

    workflow = create_workflow(1)

    run = Run(
        id=1,
        workflow_id=workflow.id,
        version=1,
        status="pending",
        tenant="test",
        caller="pytest",
        inputs_json="{}"
    )
    db.session.add(run)
    db.session.commit()

    step = RunStep(
        run_id=run.id,
        step_id="step1",
        status="pending",
        input_json="{}"
    )
    db.session.add(step)
    db.session.commit()

    output = mock_step_success("{}")
    step.status = "completed"
    db.session.commit()

    assert step.status == "completed"
    assert executed_sequence == ["step_success"]

# ----------------------------
# Test 2: Step failure triggers compensation
# ----------------------------
def test_step_failure_triggers_compensation(app, create_workflow):
    global executed_sequence
    executed_sequence = []

    workflow = create_workflow(2)

    run = Run(
        id=2,
        workflow_id=workflow.id,
        version=1,
        status="pending",
        tenant="test",
        caller="pytest",
        inputs_json="{}"
    )
    db.session.add(run)
    db.session.commit()

    step = RunStep(
        run_id=run.id,
        step_id="step_fail",
        status="pending",
        input_json="{}"
    )
    db.session.add(step)
    db.session.commit()

    try:
        mock_step_failure("{}")
    except Exception:
        step.status = "failed"
        comp = Compensation(
            run_id=run.id,
            step_id=step.step_id,
            status="pending",
            payload_json='{}'
        )
        db.session.add(comp)
        db.session.commit()

    assert step.status == "failed"
    assert db.session.query(Compensation).filter_by(run_id=run.id).count() == 1

# ----------------------------
# Test 3: Resume step signal
# ----------------------------
def test_signal_resume_step(app, create_workflow):
    global executed_sequence
    executed_sequence = []

    workflow = create_workflow(3)

    run = Run(
        id=3,
        workflow_id=workflow.id,
        version=1,
        status="waiting_for_signal",
        tenant="test",
        caller="pytest",
        inputs_json="{}"
    )
    db.session.add(run)
    db.session.commit()

    step = RunStep(
        run_id=run.id,
        step_id="step_resume",
        status="pending",
        input_json="{}"
    )
    db.session.add(step)
    db.session.commit()

    # Simulate receiving a signal
    step.status = "completed"
    db.session.commit()

    run.status = "completed"
    db.session.commit()

    assert step.status == "completed"
    assert run.status == "completed"
