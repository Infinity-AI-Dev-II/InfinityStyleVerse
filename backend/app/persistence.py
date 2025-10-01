# backend/app/persistence.py
from datetime import datetime
from sqlalchemy.orm import Session
from backend.app.models.workflow_defs import WorkflowDef
from backend.app.models.runs import Run
from backend.app.models.run_steps import RunStep



def create_run(db: Session, workflow_id: int, version: str, status: str = "running", tenant: str | None = None,
               caller: str | None = None, inputs_json: str | None = None) -> Run:
    run = Run(
        workflow_id=workflow_id,
        version=version,
        status=status,
        tenant=tenant,
        caller=caller,
        inputs_json=inputs_json
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run

def create_run_steps(db: Session, run_id: int, steps: list[dict]) -> list[RunStep]:
    created = []
    for s in steps:
        rs = RunStep(
            run_id=run_id,
            step_id=s["id"],
            type=s.get("type"),
            status="pending",
            attempt=0,
            input_json=s.get("input_json"),
        )
        db.add(rs)
        created.append(rs)
    db.commit()
    for r in created:
        db.refresh(r)
    return created

def update_step_start(db: Session, run_id: int, step_id: str):
    step = db.query(RunStep).filter_by(run_id=run_id, step_id=step_id).first()
    if step:
        step.status = "running"
        step.started_at = datetime.utcnow()
        step.attempt += 1
        db.commit()
        db.refresh(step)
    return step

def update_step_finish(db: Session, run_id: int, step_id: str, success: bool, output_json: str | None, error_json: str | None):
    step = db.query(RunStep).filter_by(run_id=run_id, step_id=step_id).first()
    if step:
        step.status = "success" if success else "failed"
        step.output_json = output_json
        step.error_json = error_json
        step.ended_at = datetime.utcnow()
        db.commit()
        db.refresh(step)
        # Update run overall status
        run = db.query(Run).get(run_id)
        if run:
            failed_steps = db.query(RunStep).filter_by(run_id=run_id, status="failed").count()
            pending_steps = db.query(RunStep).filter(RunStep.run_id == run_id, RunStep.status.in_(["pending", "running"])).count()
            if failed_steps:
                run.status = "failed"
            elif pending_steps == 0:
                run.status = "succeeded"
            db.commit()
            db.refresh(run)
    return step

def mark_run_canceled(db: Session, run_id: int):
    run = db.query(Run).get(run_id)
    if not run:
        return None
    run.status = "canceled"
    db.commit()
    db.query(RunStep).filter(RunStep.run_id == run_id, RunStep.status.in_(["pending", "running"])).update({"status": "skipped"})
    db.commit()
    return run

def get_run_state(db: Session, run_id: int):
    run = db.query(Run).get(run_id)
    if not run:
        return None
    steps = db.query(RunStep).filter_by(run_id=run_id).order_by(RunStep.id).all()
    return {"run": run, "steps": steps}
