from dataclasses import dataclass
from typing import List

from backend.app.models.run_steps import RunStep
from backend.app.models.workflow_defs import WorkflowDef


@dataclass
class GetRunInfoIDQuery:
    run_id: str