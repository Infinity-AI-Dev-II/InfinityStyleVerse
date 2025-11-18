# backend/app/tasks/workflow_tasks.py
from celery import Celery
from ..database import db
from ..models import Run, RunStep
from datetime import datetime
from tasks.anomaly_detector_functions import detect_anomalies
from backend.celery_app import celery_app as celery

# celery = Celery("workflow_tasks", broker="redis://localhost:6379/0")  # Update Redis URL if needed

@celery.task(bind=True)
def execute_step(self, step_id, run_id):
    step = RunStep.query.filter_by(run_id=run_id, step_id=step_id).first()
    if not step or step.status in ["success", "skipped"]:
        return

    step.status = "running"
    step.started_at = datetime.utcnow()  # optional timestamp
    db.session.commit()

    try:
        # Simulate or run actual step logic
        result = f"Executed step {step_id}"  # Replace with real logic
        step.output = {"result": result}
        step.status = "success"
        fallbackStat = False
    except Exception as e:
        step.status = "failed"
        step.output = {"error": str(e)}
        fallbackStat = True
    finally:
        step.ended_at = datetime.utcnow()  # optional timestamp
        # Compute duration
        if step.started_at and step.ended_at:
            duration = step.ended_at - step.started_at
            duration_ms = int(duration.total_seconds() * 1000)  # milliseconds
        else:
            duration_ms = None
            db.session.commit()
            # task_payload = {
            # "id": None,                    
            # "run_id": run_id,                
            # "step_id": step.id,               
            # "type": step.type,    
            # "status": step.status,           
            # "attempt": step.attempt,                 
            # "started_at": step.started_at,            
            # "ended_at": step.ended_at,              
            # "input_json": step.input_json,  
            # "output_json": step.output_json,           
            # "error_json": step.error_json,            
            # "duration_ms": duration_ms,           
            # "fallback_triggered": fallbackStat,   
            # "worker_heartbeat": '2025-09-18 05:29:18.881673+05:30', #FIXME: For now     
            # "error_code": 'S000', #FIXME: For now           
            # }
            # #flowGate OS
            # detect_anomalies.delay(task_payload)

    # Update run status if all steps done
    run = Run.query.get(run_id)
    pending = RunStep.query.filter_by(run_id=run_id, status="pending").count()
    running = RunStep.query.filter_by(run_id=run_id, status="running").count()
    if pending + running == 0:
        run.status = "success"
        db.session.commit()
