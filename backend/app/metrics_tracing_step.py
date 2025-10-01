import time
from queue import SimpleQueue
from backend.app.metrics import (
    STEP_SUCCESS_COUNTER, STEP_FAILURE_COUNTER, STEP_DURATION_HISTOGRAM,
    STEP_RETRY_COUNTER, STEP_COMPENSATION_COUNTER
)
from backend.app.tracing import tracer
from backend.app.event_publisher import push_update

def track_step(workflow_name, run_id, step_id, func, *args, **kwargs):
    start_time = time.time()
    with tracer.start_as_current_span(f"{workflow_name}.{step_id}") as span:
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Prometheus metrics
            STEP_SUCCESS_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()
            STEP_DURATION_HISTOGRAM.labels(workflow_name=workflow_name, step_id=step_id).observe(duration)

            # SSE / TaskPulseOS push
            push_update({
                "workflow_name": workflow_name,
                "run_id": run_id,
                "step_id": step_id,
                "status": "success",
                "duration_seconds": duration
            })

            # Tracing attributes
            span.set_attribute("status", "success")
            span.set_attribute("duration_seconds", duration)
            return result

        except Exception as e:
            duration = time.time() - start_time
            STEP_FAILURE_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()

            push_update({
                "workflow_name": workflow_name,
                "run_id": run_id,
                "step_id": step_id,
                "status": "failure",
                "duration_seconds": duration,
                "error": str(e)
            })

            span.record_exception(e)
            span.set_attribute("status", "failure")
            span.set_attribute("duration_seconds", duration)
            raise

def track_retry(workflow_name, run_id, step_id):
    STEP_RETRY_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()
    push_update({
        "workflow_name": workflow_name,
        "run_id": run_id,
        "step_id": step_id,
        "status": "retry"
    })

def track_compensation(workflow_name, run_id, step_id):
    STEP_COMPENSATION_COUNTER.labels(workflow_name=workflow_name, step_id=step_id).inc()
    push_update({
        "workflow_name": workflow_name,
        "run_id": run_id,
        "step_id": step_id,
        "status": "compensation"
    })
