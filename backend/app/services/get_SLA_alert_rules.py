from backend.app.models.echo_models import Policy
from ..database import db, get_db_session
def get_Latest_SLA_rule():
    with get_db_session() as session:
      alertRule = session.query(Policy).filter(Policy.task == "sla_rule").first()
      return alertRule
  
  