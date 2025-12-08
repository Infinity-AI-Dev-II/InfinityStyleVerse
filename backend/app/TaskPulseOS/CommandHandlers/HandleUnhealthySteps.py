from backend.app.database import get_db_session
from backend.app.models import run_steps
from backend.app.models.alerts import Alert


def HandleUnhealthySteps(command):
    try:
        worker_id = command.worker_id
        #TODO: Update steps process (in steps table)
        with get_db_session() as session:
            result = session.query(run_steps).filter(
                run_steps.worker_id == worker_id,
                run_steps.status == 'running'
            ).update({
                run_steps.status: 'stale'  # Use colon, not ==
            }, synchronize_session=False)
            session.commit()
        return {"Status": "Success", "Message": f"Updated {result} steps to stale for worker {worker_id}"}    
    except Exception as e:
        return {"Status": "Error", "Message": str(e)}    
    
    
    
    