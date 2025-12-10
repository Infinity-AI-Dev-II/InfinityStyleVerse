from ..database import get_db_session
from ..models.echo_models import Policy


def getLatestWorkerPolicy():
    with get_db_session() as session:
        return session.query(Policy).filter(Policy.task == "heartbeat").first()
