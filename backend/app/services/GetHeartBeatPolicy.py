from backend.app.models.echo_models import Policy
from ..database import db, get_db_session
import logging
def getLatestWorkerPolicy():
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  with get_db_session() as session:
    policy = session.query(Policy).filter(Policy.task == "heartbeat").first()
    # if policy is None:
    #   logger.info("Data is null")
    # else:  
    #   logger.info(f"TEST DATA: {policy.config_json}")
    return policy
  
  
  