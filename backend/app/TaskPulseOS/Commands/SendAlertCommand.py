from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import text

from backend.app.models.alerts import SeverityLevel


@dataclass
class SendAlertCommand:
    severity: SeverityLevel
    code: str
    message: str
    created_at: datetime
    acknowledged_by: str
    workerID: str

