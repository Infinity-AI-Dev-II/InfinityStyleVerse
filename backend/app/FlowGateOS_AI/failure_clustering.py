import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime
import os

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'anomaly_data.csv')

# Loading the data
df = pd.read_csv(csv_path)

# Convert timestamps to datetime
df['started_at'] = pd.to_datetime(df['started_at'])

# Prepare data for clustering
# Combine error_json and error_code with type
df['error_text'] = df['error_json'].fillna('') + ' ' + df['error_code'] + ' ' + df['type']
# Filter rows with errors or anomalies
error_df = df[df['status'].isin(['failed', 'compensating']) | df['fallback_triggered'] | (df['attempt'] > 2)]

# Vectorize text for clustering
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(error_df['error_text'])

# Apply KMeans clustering (k=5 as a starting point)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to dataframe
error_df['cluster'] = clusters

# Group by week and cluster to find top 3 per week
error_df['week'] = error_df['started_at'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_clusters = error_df.groupby(['week', 'cluster']).size().reset_index(name='count')
top_clusters = weekly_clusters.sort_values(['week', 'count'], ascending=[True, False]).groupby('week').head(3)

# Display results
print("Top 3 Failure Clusters per Week:")
for week, group in top_clusters.groupby('week'):
    print(f"\nWeek starting {week.strftime('%Y-%m-%d')}:")
    for _, row in group.iterrows():
        cluster_errors = error_df[error_df['cluster'] == row['cluster']]['error_text'].head(1).values[0]
        print(f"  Cluster {row['cluster']}: {row['count']} incidents - Example: {cluster_errors}")