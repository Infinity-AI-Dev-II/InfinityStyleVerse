
from backend.app.database import get_db_session
from backend.app.models import Heartbeat

def SaveHeartbeatCmdHnd(command):
    """Handles the SaveHeartbeat command to store heartbeat data."""
    try:
        newHB = Heartbeat(worker_id=command.worker_id,
                          kind=command.kind,
                          last_seen_at=command.last_seen_at,
                          meta_json=command.meta_json)
        with get_db_session() as session:
            session.add(newHB)
        return {"Status": "Success", "Message": newHB}    
    except Exception as e:
        # Handle exceptions and return an error response
        return {"Status": "Error", "Message": str(e)}