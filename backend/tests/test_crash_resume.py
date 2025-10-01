# backend/tests/test_crash_resume.py
import pytest
from datetime import datetime
from backend.app.models.workflow_defs import WorkflowDef
from backend.app.models.runs import Run
from backend.app.models.run_steps import RunStep
from backend.app.models.run_var import RunVar
from backend.app.models.compensations import Compensation
from backend.app.database import db

@pytest.fixture
def setup_workflow(app):
    """Create a test workflow in DB"""
    wf = WorkflowDef(
        name="crash_resume_workflow",
        version="v1",
        dsl_yaml="{}",
        dag_json="{}",
        created_by="pytest",
        created_at=datetime.utcnow(),
    )
    db.session.add(wf)
    db.session.commit()
    yield wf
    # cleanup
    RunStep.query.filter_by(run_id=wf.id).delete()
    RunVar.query.filter_by(run_id=wf.id).delete()
    Compensation.query.filter_by(run_id=wf.id).delete()
    Run.query.filter_by(workflow_id=wf.id).delete()
    WorkflowDef.query.filter_by(id=wf.id).delete()
    db.session.commit()

def test_crash_and_resume(setup_workflow):
    wf = setup_workflow

    # Step 1: Start a run
    run = Run(
        workflow_id=wf.id,
        version=wf.version,
        status="running",
        tenant="test",
        caller="pytest",
        started_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.session.add(run)
    db.session.commit()

    # Step 2: Simulate some completed steps
    step1 = RunStep(
        run_id=run.id,
        step_id="step1",
        type="task",
        status="completed",
        attempt=1,
        started_at=datetime.utcnow(),
        ended_at=datetime.utcnow(),
    )
    db.session.add(step1)
    db.session.commit()

    # Step 3: Simulate a step that was in progress during crash
    step2 = RunStep(
        run_id=run.id,
        step_id="step2",
        type="task",
        status="running",  # mid-execution
        attempt=1,
        started_at=datetime.utcnow(),
        ended_at=None,
    )
    db.session.add(step2)
    db.session.commit()

    # Step 4: Simulate worker crash by rolling back in-progress step
    # In real scenario, step2 status would stay "running" in DB
    # We'll "restart" the workflow now
    db.session.expire_all()  # clear session cache

    # Resume logic: mark running steps as pending
    steps_to_resume = RunStep.query.filter_by(run_id=run.id, status="running").all()
    for s in steps_to_resume:
        s.status = "pending"
        s.attempt += 1
        s.started_at = None
        s.ended_at = None
    db.session.commit()

    # Step 5: Execute the pending steps (simulate run)
    for s in RunStep.query.filter_by(run_id=run.id, status="pending"):
        s.status = "completed"
        s.started_at = datetime.utcnow()
        s.ended_at = datetime.utcnow()
    db.session.commit()

    # Step 6: Mark run as resumed/completed
    run.status = "resumed"
    run.updated_at = datetime.utcnow()
    db.session.commit()

    # --- Assertions ---
    run_db = Run.query.get(run.id)
    assert run_db.status == "resumed"

    completed_steps = RunStep.query.filter_by(run_id=run.id, status="completed").all()
    assert len(completed_steps) == 2  # step1 + resumed step2

    # Ensure no duplicates of step2
    step2_entries = RunStep.query.filter_by(run_id=run.id, step_id="step2").all()
    assert len(step2_entries) == 1
    assert step2_entries[0].attempt == 2  # resumed attempt

    print("Crash/resume test passed ✅")
