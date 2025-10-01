import random
from datetime import datetime, timedelta
import pandas as pd

# Generating sample data
data = []
for i in range(200):  
    run_id = i
    step_id = f"step_{random.randint(1, 20)}"
    step_type = random.choice(["model_call", "http_call", "join", "map", "transform"])
    status = random.choice(["succeeded", "failed", "running"])
    attempt = random.randint(0, 3) 
    # Latency between 200ms and 6000ms
    base_latency_seconds = random.uniform(0.2, 6.0)
    started_at = datetime.now() - timedelta(minutes=random.randint(1, 300), seconds=random.uniform(0, 59))
    ended_at = started_at + timedelta(seconds=base_latency_seconds)
    input_json = '{"param1": "value1", "param2": ' + str(random.randint(1, 100)) + '}'
    output_json = '{"result": "success"}' if status == "succeeded" else '{"error": "timeout"}' if status == "failed" else None
    error_json = '{"error": "network_fail"}' if status == "failed" else None

    data.append({
        "id": i + 1,  
        "run_id": run_id,
        "step_id": step_id,
        "type": step_type,
        "status": status,
        "attempt": attempt,
        "started_at": started_at,
        "ended_at": ended_at,
        "input_json": input_json,
        "output_json": output_json,
        "error_json": error_json
    })

# Creating a DataFrame and saving to CSV
df = pd.DataFrame(data)
df.to_csv('run_steps_sample_data.csv', index=False)
print("Sample data saved to run_steps_sample_data.csv")