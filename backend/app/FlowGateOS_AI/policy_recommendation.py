import pandas as pd
import os

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'anomaly_data.csv')

# Loading the data
df = pd.read_csv(csv_path)

# Converting timestamps to datetime
df['started_at'] = pd.to_datetime(df['started_at'], format='ISO8601')
df['ended_at'] = pd.to_datetime(df['ended_at'], format='ISO8601')

# Filtering the successful steps
success_df = df[df['status'] == 'succeeded']

# Grouping by the type and calculating the success rate vs attempts

type_attempts = df.groupby('type', as_index=False)['attempt'].mean()

success_rate = (
    success_df.groupby('type').size() / df.groupby('type').size()
).reset_index(name='success_rate')

policy_recommendations = pd.merge(type_attempts, success_rate, on='type', how='left').fillna(0)

# Suggesting adjustments
recommendations = []
for _, row in policy_recommendations.iterrows():
    if row['attempt'] > 2 and row['success_rate'] < 0.8:
        recommendations.append(f"For {row['type']}, increase retries to {int(row['attempt']) + 1} and set timeout to 1000ms due to high attempts and low success.")
    elif row['success_rate'] < 0.5:
        recommendations.append(f"For {row['type']}, increase retries to 4 and use exponential backoff due to low success rate.")
    else:
        recommendations.append(f"For {row['type']}, keep retries at 3 and timeout at 500ms as success rate is acceptable.")

print("Policy Recommendations:")
for rec in recommendations:
    print(rec)