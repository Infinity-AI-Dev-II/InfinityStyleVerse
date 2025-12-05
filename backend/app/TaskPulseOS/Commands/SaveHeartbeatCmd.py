from ast import Dict
from dataclasses import dataclass
from datetime import datetime
from typing import Any
@dataclass
class SaveHeartbeatCmd:
    worker_id: str
    kind: str
    last_seen_at: datetime
    meta_json: Dict[str, Any]
