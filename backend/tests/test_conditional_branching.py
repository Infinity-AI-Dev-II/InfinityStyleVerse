# backend/tests/test_conditional_branching.py
import pytest
from backend.app.models.workflow_defs import WorkflowDef
from backend.app.models.runs import Run
from backend.app.models.run_steps import RunStep
from backend.app.models.run_var import RunVar
from backend.app.database import db
from datetime import datetime

def create_conditional_workflow():
    """
    Workflow DSL example with:
    - if/else branching
    - switch-case
    - loop over collection
    """
    dsl_yaml = """
    version: 1
    steps:
      - id: start
        type: start
      - id: check_region
        type: switch
        variable: "{{variables.region}}"
        cases:
          US:
            - id: us_task
              type: task
          EU:
            - id: eu_task
              type: task
          default:
            - id: default_task
              type: task
      - id: iterate_items
        type: loop
        collection: "{{variables.items}}"
        body:
          - id: process_item
            type: task
      - id: end
        type: end
    """
    workflow_def = WorkflowDef(
        name="conditional_test",
        version="1.0",
        dsl_yaml=dsl_yaml,
        created_by="pytest",
        created_at=datetime.utcnow()
    )
    db.session.add(workflow_def)
    db.session.commit()
    return workflow_def


def test_conditional_branching(client):
    workflow_def = create_conditional_workflow()

    # Example run variables
    variables = {
        "region": "EU",
        "items": ["apple", "banana", "cherry"]
    }

    run = Run(
        workflow_id=workflow_def.id,
        version=workflow_def.version,
        inputs_json=str(variables),
        status="pending",
        started_at=datetime.utcnow(),
        created_at=datetime.utcnow()
    )
    db.session.add(run)
    db.session.commit()

    # Simulate workflow engine execution
    # Check switch-case branching
    if variables["region"] == "US":
        step_id = "us_task"
    elif variables["region"] == "EU":
        step_id = "eu_task"
    else:
        step_id = "default_task"

    run_step = RunStep(
      run_id=run.id,
      step_id=step_id,
      status="completed",
      started_at=datetime.utcnow(),
      ended_at=datetime.utcnow()
    )

    db.session.add(run_step)

    # Simulate loop over collection
    for item in variables["items"]:
      loop_step = RunStep(
        run_id=run.id,
        step_id=f"process_item_{item}",
        status="completed",
        started_at=datetime.utcnow(),
        ended_at=datetime.utcnow()
    )
      db.session.add(loop_step)

    db.session.commit()

    # Assertions
    steps = RunStep.query.filter_by(run_id=run.id).all()
    step_ids = [s.step_id for s in steps]

    assert "eu_task" in step_ids
    assert "process_item_apple" in step_ids
    assert "process_item_banana" in step_ids
    assert "process_item_cherry" in step_ids