# tests/test_join_map_runners.py
import pytest
from concurrent.futures import ThreadPoolExecutor

# --- Example runners to test ---
def map_runner(user_func, items):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(user_func, item): item for item in items}
        for future in futures:
            item = futures[future]
            try:
                output = future.result()
                results.append({"item": item, "status": "completed", "output": output, "error": None})
            except Exception as e:
                results.append({"item": item, "status": "failed", "output": None, "error": str(e)})
    return results

def join_runner(step_outputs):
    combined_output = []
    overall_status = "completed"
    for res in step_outputs:
        combined_output.append(res["output"])
        if res["status"] == "failed":
            overall_status = "failed"
    return {"status": overall_status, "output": combined_output}

# --- Tests ---
def test_map_runner_success():
    items = [1, 2, 3, 4]

    def square(x):
        return x * x

    results = map_runner(square, items)
    assert all(r["status"] == "completed" for r in results)
    assert [r["output"] for r in results] == [1, 4, 9, 16]

def test_map_runner_failure():
    items = [1, 0, 2]

    def risky_divide(x):
        return 10 / x  # Will raise ZeroDivisionError for x=0

    results = map_runner(risky_divide, items)
    statuses = [r["status"] for r in results]
    outputs = [r["output"] for r in results]
    errors = [r["error"] for r in results]

    assert statuses.count("failed") == 1
    assert statuses.count("completed") == 2
    assert outputs[1] is None
    assert "division by zero" in errors[1]

def test_join_runner():
    step_outputs = [
        {"status": "completed", "output": [1, 2]},
        {"status": "completed", "output": [3, 4]},
    ]
    result = join_runner(step_outputs)
    assert result["status"] == "completed"
    assert result["output"] == [[1, 2], [3, 4]]

def test_join_runner_with_failure():
    step_outputs = [
        {"status": "completed", "output": [1, 2]},
        {"status": "failed", "output": None},
    ]
    result = join_runner(step_outputs)
    assert result["status"] == "failed"
    assert result["output"] == [[1, 2], None]
