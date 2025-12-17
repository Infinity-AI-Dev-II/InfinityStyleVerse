from dataclasses import dataclass

@dataclass
class AlertQuery:
    tenant: str
    severity: str

