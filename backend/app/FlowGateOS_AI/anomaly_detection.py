import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv(r"C:\Users\acer\Desktop\Infinity AI Work\InfinityStyleVerse\backend\app\FlowGateOS_AI\anomaly_data.csv")

# Converting timestamps to datetime
df['started_at'] = pd.to_datetime(df['started_at'], format='mixed', utc=False, errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed', utc=False, errors='coerce')
df['worker_heartbeat'] = pd.to_datetime(df['worker_heartbeat'], format='mixed', utc=False, errors='coerce')

# Setting current time
current_time = pd.to_datetime("2025-09-24 11:31:00+0530")

# Anomaly detection
# Fallback rate (daily, 14-day window)
df['date'] = df['started_at'].dt.date
fallback_daily = df.groupby('date')['fallback_triggered'].mean().reset_index()
fallback_daily.columns = ['date', 'fallback_rate']
fallback_daily['ewma'] = fallback_daily['fallback_rate'].ewm(span=14).mean()
fallback_daily['std'] = fallback_daily['fallback_rate'].rolling(window=14).std().fillna(0)
fallback_daily['upper_bound'] = fallback_daily['ewma'] + 2.5 * fallback_daily['std']  # Adjusted to 2.5 * std
fallback_daily['anomaly'] = (fallback_daily['fallback_rate'] > fallback_daily['upper_bound']) & (fallback_daily['fallback_rate'] > 0.15)

# Retry bursts (per hour, 48-hour window)
df['hour'] = df['started_at'].dt.floor('H')
retry_hourly = df.groupby(['hour', 'run_id'])['attempt'].sum().reset_index()
retry_hourly = retry_hourly.groupby('hour')['attempt'].mean().reset_index()
retry_hourly['ewma'] = retry_hourly['attempt'].ewm(span=48).mean()
retry_hourly['std'] = retry_hourly['attempt'].rolling(window=48).std().fillna(0)
retry_hourly['upper_bound'] = retry_hourly['ewma'] + 2.5 * retry_hourly['std']
retry_hourly['anomaly'] = (retry_hourly['attempt'] > retry_hourly['upper_bound']) & (retry_hourly['attempt'] > 2.0)

# Worker heartbeat gaps (daily, 14-day window)
df = df.sort_values('worker_heartbeat')
df['heartbeat_gap'] = df['worker_heartbeat'].diff().dt.total_seconds().fillna(0)
heartbeat_gaps = df.groupby(df['worker_heartbeat'].dt.floor('D'))['heartbeat_gap'].mean().reset_index()
heartbeat_gaps['ewma'] = heartbeat_gaps['heartbeat_gap'].ewm(span=14).mean()
heartbeat_gaps['std'] = heartbeat_gaps['heartbeat_gap'].rolling(window=14).std().fillna(0)
heartbeat_gaps['upper_bound'] = heartbeat_gaps['ewma'] + 2.5 * heartbeat_gaps['std']
heartbeat_gaps['anomaly'] = (heartbeat_gaps['heartbeat_gap'] > heartbeat_gaps['upper_bound']) & (heartbeat_gaps['heartbeat_gap'] > 60)

# Displaying results
print("Fallback Rate Anomalies:")
print(fallback_daily[fallback_daily['anomaly']])
print("\nRetry Bursts Anomalies:")
print(retry_hourly[retry_hourly['anomaly']])
print("\nHeartbeat Gap Anomalies:")
print(heartbeat_gaps[heartbeat_gaps['anomaly']])