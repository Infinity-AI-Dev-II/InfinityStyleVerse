#select the latest heartbeat entry
from backend.app.database import get_db_session
from backend.app.models import Heartbeat

def SelectLatestHB():
    try:
        with get_db_session() as session:
            LastHB = session.query(Heartbeat).order_by(Heartbeat.id.desc()).first()
        return {"Status":"Success","Message":LastHB}
    except Exception as e:
        return {"Status":"Error","Message":str(e)}    