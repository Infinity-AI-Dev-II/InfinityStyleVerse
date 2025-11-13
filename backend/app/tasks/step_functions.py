# backend/app/tasks/step_functions.py
#   Implements a Saga pattern executor for workflow steps.
"""
Define main step functions and compensation functions here.
STEP_REGISTRY maps step_id -> {"execute_fn": ..., "compensate_fn": ...}
"""

def step_a(input_json):
    print(f"Executing Step A with input: {input_json}")
    # Example logic: fail if input_json contains "fail"
    if input_json and "fail" in input_json:
        raise Exception("Step A failed")
    return '{"result": "ok"}'


def compensate_a(payload_json):
    print(f"Compensating Step A with payload: {payload_json}")


def step_b(input_json):
    print(f"Executing Step B with input: {input_json}")
    return '{"result": "ok"}'


def compensate_b(payload_json):
    print(f"Compensating Step B with payload: {payload_json}")


# STEP_REGISTRY maps step_id to its functions
STEP_REGISTRY = {
    "step_a": {"execute_fn": step_a, "compensate_fn": compensate_a},
    "step_b": {"execute_fn": step_b, "compensate_fn": compensate_b},
}
