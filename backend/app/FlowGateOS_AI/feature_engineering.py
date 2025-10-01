import pandas as pd
import numpy as np
from datetime import datetime

# Loading the sample data 
df = pd.read_csv('C://Users//acer//Desktop//Infinity AI Work//InfinityStyleVerse//backend//app//FlowGateOS_AI//run_steps_sample_data.csv', parse_dates=['started_at', 'ended_at'])

# Calculating the features
df['latency_ms'] = (df['ended_at'] - df['started_at']).dt.total_seconds() * 1000  # Converting to milliseconds
df['queue_time_ms'] = df.groupby('run_id')['started_at'].diff().dt.total_seconds().fillna(0) * 1000  # Time since last step
df['is_parallel'] = df['step_id'].str.contains('parallel').astype(int)  # Detecting parallel steps
df['sla_breach'] = (df['latency_ms'] > 2500).astype(int)  # SLA threshold of 2500ms

# Aggregating historical means and variances per step type
historical_stats = df.groupby('type').agg({
    'latency_ms': ['mean', 'var']
}).reset_index()
historical_stats.columns = ['type', 'mean_latency_ms', 'var_latency_ms']
df = df.merge(historical_stats, on='type', how='left')

# Selecting relevant columns for output
output_df = df[['run_id', 'step_id', 'type', 'status', 'started_at', 'ended_at', 'latency_ms',
                'queue_time_ms', 'is_parallel', 'sla_breach', 'mean_latency_ms', 'var_latency_ms']]

# Saving to CSV
output_df.to_csv('features.csv', index=False)
print("Features extracted and saved to features.csv")