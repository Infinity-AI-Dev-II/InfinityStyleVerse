from email.policy import Policy
from ..database import db, get_db_session
def getLatestWorkerPolicy():
    with get_db_session() as session:
      policy = session.query(Policy).filter(Policy.task == "heartbeat").first()
      return policy