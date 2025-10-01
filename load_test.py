import asyncio
import time
import httpx
from statistics import mean

# Configuration
API_ENDPOINT = "http://localhost:8000/start_workflow"  # Replace with your workflow trigger endpoint
CONCURRENT_WORKFLOWS = 100
WORKFLOW_PAYLOAD = {
    "workflow_name": "sample_workflow",
    "input_data": {"key": "value"}  # Customize as needed
}

# Metrics
durations = []
failures = 0
retries = 0  # Track retries if your API returns info

async def trigger_workflow(client: httpx.AsyncClient, workflow_id: int):
    global failures, retries
    start_time = time.time()
    try:
        response = await client.post(API_ENDPOINT, json=WORKFLOW_PAYLOAD)
        if response.status_code != 200:
            failures += 1
            print(f"[Workflow {workflow_id}] Failed with status {response.status_code}")
        else:
            data = response.json()
            # Example: track retries or compensation triggered from response
            retries += data.get("retries", 0)
    except Exception as e:
        failures += 1
        print(f"[Workflow {workflow_id}] Exception: {e}")
    finally:
        duration = time.time() - start_time
        durations.append(duration)

async def run_load_test():
    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [
            trigger_workflow(client, workflow_id=i)
            for i in range(CONCURRENT_WORKFLOWS)
        ]
        await asyncio.gather(*tasks)

def print_metrics():
    print("\n--- Load Test Metrics ---")
    print(f"Total workflows: {CONCURRENT_WORKFLOWS}")
    print(f"Successful: {CONCURRENT_WORKFLOWS - failures}")
    print(f"Failed: {failures}")
    print(f"Total retries triggered: {retries}")
    print(f"Average duration: {mean(durations):.2f} seconds")
    print(f"Max duration: {max(durations):.2f} seconds")
    print(f"Min duration: {min(durations):.2f} seconds")

if __name__ == "__main__":
    start = time.time()
    asyncio.run(run_load_test())
    end = time.time()
    print_metrics()
    print(f"Total test time: {end - start:.2f} seconds")
