import enum
from ..database import db
from datetime import datetime

class SeverityLevel(enum.Enum):
    """Enum for alert severity levels"""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

class Alert(db.Model):
    """Alert model with enhanced features"""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_id = db.Column(db.Integer, nullable=False, index=True)
    step_id = db.Column(db.String, nullable=False, index=True)
    severity = db.Column(db.String(20), nullable=False, index=True)
    code = db.Column(db.String(50), nullable=False, index=True)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    acknowledged_by = db.Column(db.String(100), nullable=True)

