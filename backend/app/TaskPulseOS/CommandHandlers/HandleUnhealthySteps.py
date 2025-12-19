from backend.app.database import get_db_session
from backend.app.models.alerts import Alert
from backend.app.models.run_steps import RunStep


def HandleUnhealthySteps(command):
    try:
        worker_id = command.worker_id
        #TODO: Update steps process (in steps table)
        with get_db_session() as session:
            result = session.query(RunStep).filter(
                RunStep.worker_id == worker_id,
                RunStep.status == 'running'
            ).update({
                RunStep.status: 'stale'  
            }, synchronize_session=False)
            session.commit()
        return {"Status": "Success", "Message": f"Updated {result} steps to stale for worker {worker_id}"}    
    except Exception as e:
        return {"Status": "Error", "Message": str(e)}    
    
    
    
    