
from backend.app.TaskPulseOS.Quarries.AlertQuery import AlertQuery
from backend.app.database import get_db_session
from backend.app.models.alerts import Alert
from backend.app.models.runs import Run


def alert_query_handler(command: AlertQuery):
    try:
        with get_db_session() as session:
            results = session.query(Run.id).filter(
                Run.tenant == command.tenant
            ).all()
            for runID in results:
                allAlerts = session.query(Alert).filter(
                Alert.run_id == runID,
                Alert.severity == command.severity
                ).all()
        return {"Status": "Success", "Message": allAlerts}        
    except Exception as e:
        return {"Status": "Error", "Message": str(e)}    

