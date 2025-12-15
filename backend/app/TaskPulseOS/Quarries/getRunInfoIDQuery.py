from dataclasses import dataclass
from typing import List

from backend.app.models.run_steps import RunStep
from backend.app.models.workflow_defs import WorkflowDef


@dataclass
class GetRunInfoIDQuery:
    # runSteps: List[RunStep] = []
    # workflow: WorkflowDef
    run_id: str