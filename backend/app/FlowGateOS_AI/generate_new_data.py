import pandas as pd
import numpy as np
from faker import Faker
import datetime
from dateutil import parser
from datetime import timezone, timedelta

# Initializing Faker
fake = Faker()

# Defining Sri Lankan timezone (+5:30)
sri_lanka_tz = timezone(timedelta(hours=5, minutes=30))

# Setting seed for reproducibility
np.random.seed(42)

# Parameters
n_rows = 500
run_ids = np.random.randint(0, 30, n_rows)  # 30 unique runs
step_types = ['model_call', 'http_call', 'sql_query', 'python_fn', 'map', 'join', 'wait', 'fallback', 'compensate', 'switch']
statuses = ['succeeded', 'failed', 'running', 'waiting', 'compensating']
error_codes = {'timeout_error': 'T001', 'network_fail': 'N002', 'api_down': 'N003', 'validation_fail': 'V004'}
current_time = parser.parse("2025-09-23 13:20:00+0530")

# Generating started_at times
started_times = [fake.date_time_between(start_date="-4w", end_date="now", tzinfo=sri_lanka_tz) for _ in range(n_rows)]

# Generating data
data = {
    'id': range(1, n_rows + 1),
    'run_id': run_ids,
    'step_id': [f'step_{i}' for i in range(n_rows)],
    'type': np.random.choice(step_types, n_rows),
    'status': np.random.choice(statuses, n_rows, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
    'attempt': np.random.choice([0, 1, 2, 3, 4, 5], n_rows, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]),
    'started_at': started_times,
    'ended_at': [
        fake.date_time_between(start_date=start_time, end_date=current_time, tzinfo=sri_lanka_tz)
        if status in ['succeeded', 'failed', 'compensating'] else None
        for start_time, status in zip(started_times, np.random.choice(statuses, n_rows, p=[0.5, 0.2, 0.1, 0.1, 0.1]))
    ],
    'input_json': [f'{{"param1": "{fake.word()}", "param2": {fake.random_int(min=10, max=100)}}}' for _ in range(n_rows)],
    'output_json': [f'{{"result": "{fake.word()}"}}' if s == 'succeeded' else None for s in np.random.choice(statuses, n_rows, p=[0.5, 0.2, 0.1, 0.1, 0.1])],
    'error_json': [
        f'{{"error": "{err}", "code": "{error_codes[err]}"}}'
        if s == 'failed' or (s in ['succeeded', 'compensating'] and np.random.random() < 0.1)
        else None
        for s, err in zip(np.random.choice(statuses, n_rows, p=[0.5, 0.2, 0.1, 0.1, 0.1]), np.random.choice(list(error_codes.keys()), n_rows))
    ]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Converting to datetime
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')

# Calculating duration_ms
df['duration_ms'] = df.apply(
    lambda x: int((x['ended_at'] - x['started_at']).total_seconds() * 1000)
    if pd.notnull(x['ended_at']) else
    int((current_time - x['started_at']).total_seconds() * 1000),
    axis=1
)

# Calculating additional columns
df['fallback_triggered'] = df.apply(
    lambda x: True if x['status'] == 'succeeded' and x['error_json'] and any(e in x['error_json'] for e in error_codes.keys())
    else np.random.choice([True, False], p=[0.1, 0.9]),
    axis=1
)

df['worker_heartbeat'] = df.apply(
    lambda x: x['started_at'] - datetime.timedelta(seconds=np.random.uniform(10, 60))
    if np.random.random() > 0.1 else x['started_at'] - datetime.timedelta(seconds=np.random.uniform(61, 300)),
    axis=1
)

df['error_code'] = df.apply(
    lambda x: error_codes.get(next((e for e in error_codes.keys() if e in str(x['error_json'])), None), 'S000'),
    axis=1
)

# Saving run_steps data
df[['id', 'run_id', 'step_id', 'type', 'status', 'attempt', 'started_at', 'ended_at', 'input_json', 'output_json', 'error_json']].to_csv(
    'run_steps_sample_data.csv', index=False
)

# Saving anomaly data
df.to_csv('anomaly_data.csv', index=False)

print("Sample data generated and saved.")