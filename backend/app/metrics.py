# backend/app/metrics.py
from prometheus_client import Counter, Histogram

# Count how many tasks are started and failed
TASKS_STARTED = Counter("tasks_started_total", "Total tasks started", ["name"])
TASKS_FAILED  = Counter("tasks_failed_total", "Total tasks failed", ["name"])

# Measure task duration in seconds
TASKS_TIME = Histogram("task_duration_seconds", "Task execution duration", ["name"])

# Step counters
STEP_SUCCESS_COUNTER = Counter(
    "workflow_step_success_total",
    "Total number of workflow steps completed successfully",
    ["workflow_name", "step_id"]
)

STEP_FAILURE_COUNTER = Counter(
    "workflow_step_failure_total",
    "Total number of workflow steps failed",
    ["workflow_name", "step_id"]
)

# Step duration histogram (seconds)
STEP_DURATION_HISTOGRAM = Histogram(
    "workflow_step_duration_seconds",
    "Duration of workflow steps in seconds",
    ["workflow_name", "step_id"]
)

# Compensation / retry counters
STEP_RETRY_COUNTER = Counter(
    "workflow_step_retry_total",
    "Total number of workflow step retries",
    ["workflow_name", "step_id"]
)

STEP_COMPENSATION_COUNTER = Counter(
    "workflow_step_compensation_total",
    "Total number of compensations executed",
    ["workflow_name", "step_id"]
)
