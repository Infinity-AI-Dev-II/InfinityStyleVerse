
from datetime import datetime
from typing import List

from flask import json, jsonify
import requests
from backend.app.TaskPulseOS.Quarries.getRunInfoIDQuery import GetRunInfoIDQuery
from backend.app.database import get_db_session
from backend.app.models.alerts import Alert
from backend.app.models.run_steps import RunStep
from backend.app.models.runs import Run
from backend.app.models.workflow_defs import WorkflowDef
from backend.app import alertRule
from backend.app.services.kafka_service import get_kafka_producer

class GetRunInfoIDQueryHandler:
    def __init__(self):
        self.kafka_producer = get_kafka_producer()
    def getWorkflowRun(self, command: GetRunInfoIDQuery):
        try:
            with get_db_session() as session:
                runID = command.run_id
                #get workflowID by run_id
                runData = session.query(Run.workflow_id).filter(
                    Run.id == runID
                )
                #workflow info
                workFlow = session.query(WorkflowDef).filter(
                    WorkflowDef.id == runData.workflow_id
                ).first()
                #steps
                data: List[RunStep] = session.query(RunStep).filter(
                    RunStep.run_id == runID
                )
            payload = jsonify({"run_steps": data,"workflow_def": workFlow})     
            #calls the internal route
            response = requests.post("http://localhost:8000/predict/sla_risk", json=payload)
            config = alertRule.config_json
            with get_db_session() as session:
                #FIXME: this is only the at_risk part. add more conditions for other risk levels. The banner can be created using serverity levels (only at_risl for now)
                if response.json["risk"] > config["at_risk"] and response.json["critical_path"] > config["critical_path"] or response.json["fallback_rate"] > config["fallback_rate"]:
                    #update run table
                    result = session.query(Run).filter(
                        Run.id == runID
                    ).update({
                        Run.status: 'at_risk'
                    }, synchronize_session=False)
                    #create the message
                    if response.json["risk"] > config["at_risk"] and response.json["critical_path"] > config["critical_path"] and response.json["fallback_rate"] > config["fallback_rate"]:
                        msg = "The SLA risk for this step is at risk due to high critical path and high fallback rate"
                    elif response.json["risk"] > config["at_risk"] and response.json["critical_path"] > config["critical_path"]:
                        msg = "The SLA risk for this step is at risk due to high critical path" 
                    elif response.json["fallback_rate"] > config["fallback_rate"]:
                        msg = "The SLA risk for this step is at risk due to high fallback rate"       
                    #create alerts into alert table
                    for gatheredData in data:
                        alert = Alert(
                            run_id=gatheredData.run_id,
                            step_id=gatheredData.step_id,
                            serverity="at_risk",
                            code="SLA_AT_RISK",
                            message=msg,
                        )
                        session.add(alert)
                    #push to kafka consumer        
                    event = {
                    "event": "alert.raised",
                    "severity": "at_risk",
                    "code": "SLA_AT_RISK",
                    "run_id": runID,
                    "message": msg,
                    "ts": datetime.utcnow().isoformat()
                    }  
                    self.kafka_producer.produce(
                        topic='alert.raised',
                        value=json.dumps(event).encode("utf-8")
                    )
                #TODO: Update promethus metrices
                #TODO: Produce traces and logs       
            session.commit()  
            #TODO: return the KPI data too for now return the sla risk data only as kpi
            return {"status": "success","runData": runData,"workFlowData": workFlow,"KPI":response.json}   
        except Exception as e:
            return {"status": "error", "Message": str(e)}    


