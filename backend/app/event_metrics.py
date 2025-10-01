import time
from .metrics import (
    STEP_SUCCESS_COUNTER, STEP_FAILURE_COUNTER,
    STEP_DURATION_HISTOGRAM, STEP_RETRY_COUNTER,
    STEP_COMPENSATION_COUNTER
)
from .tracing import tracer

def track_step_execution(workflow_name, step_id, func, *args, **kwargs):
    """
    Executes a workflow step and updates metrics + tracing.
    """
    start_time = time.time()
    with tracer.start_as_current_span(f"{workflow_name}.{step_id}") as span:
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            STEP_SUCCESS_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()
            STEP_DURATION_HISTOGRAM.labels(workflow_name=workflow_name, step_id=step_id).observe(duration)
            span.set_attribute("status", "success")
            span.set_attribute("duration_seconds", duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            STEP_FAILURE_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()
            span.set_attribute("status", "failure")
            span.set_attribute("duration_seconds", duration)
            span.record_exception(e)
            raise

def track_retry(workflow_name, step_id):
    STEP_RETRY_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()

def track_compensation(workflow_name, step_id):
    STEP_COMPENSATION_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()
