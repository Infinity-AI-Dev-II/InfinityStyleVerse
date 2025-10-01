# test_task.py
from backend.celery_app import celery_app
from workflow import http_call, python_fn

if __name__ == "__main__":
    # --- Test HTTP task (I/O-bound) ---
    print("Sending HTTP task to I/O queue...")
    http_result = http_call.apply_async(
        args=("https://httpbin.org/get", "GET"),
        queue="io"
    )
    print(f"HTTP task ID: {http_result.id}")

    # --- Test Python CPU task (CPU-bound) ---
    print("Sending Python function task to CPU queue...")
    cpu_result = python_fn.apply_async(
        args=(10, 5),
        kwargs={"operation": "mul", "fn": "multiply_step"},
        queue="cpu"
    )
    print(f"Python task ID: {cpu_result.id}")

    # Optionally, wait for results (blocking)
    print("Waiting for results...")
    print("HTTP result:", http_result.get(timeout=10))
    print("Python result:", cpu_result.get(timeout=10))