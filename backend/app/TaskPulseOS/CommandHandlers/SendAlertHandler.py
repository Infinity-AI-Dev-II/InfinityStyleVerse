import datetime
import json
from backend.app.TaskPulseOS.Commands import SendAlertCommand
from backend.app.database import get_db_session
from backend.app.models.alerts import Alert
from backend.app.models.run_steps import RunStep
from backend.app.models.runs import Run
from backend.app.services.kafka_service import get_kafka_producer


class CreateAlertHandler:
    def __init__(self):
        self.kafka_producer = get_kafka_producer()
    
    def handle(self, command: SendAlertCommand):
        try:
            with get_db_session() as session:
                results = session.query(RunStep.step_id, RunStep.run_id).filter(
                    RunStep.worker_id == command.workerID,
                    RunStep.status == 'stale'
                ).all()
            
            with get_db_session() as session:
                for step_id, run_id in results:
                    alert = Alert(
                        run_id=run_id,
                        step_id=step_id,
                        severity=command.severity,
                        code=command.code,
                        message=command.message,
                        created_at=command.created_at,
                        acknowledged_by=command.acknowledged_by
                    )
                    session.add(alert)
                    
                    runResults = session.query(Run.tenant).filter(
                    Run.id == run_id
                    )
                    
                    event = {
                    "event": "alert.raised",
                    "tenant": runResults[0].tenant if runResults.count() > 0 else None,
                    "step_id": step_id,
                    "status": "failed",
                    "run_id": command.run_id,
                    "ts": datetime.utcnow().isoformat(),
                    "payload": {"error_code":command.code,"message":command.message,"severity": command.severity}
                    }
                    self.kafka_producer.produce(
                        topic='step.failed',
                        value=json.dumps(event).encode("utf-8")
                    )
            return {"Status": "Success", "Message": alert}
        except Exception as e:
            return {"Status": "Error", "Message": str(e)}
