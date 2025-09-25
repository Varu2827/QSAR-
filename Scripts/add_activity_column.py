import pandas as pd

# Load the merged dataset from CSV
data = pd.read_csv("merged_qsar.csv")

# Add Activity column based on Binding Affinity
data['Activity'] = data['Binding Affinity'].apply(lambda x: 1 if x <= -7 else 0)

# Save to new file
data.to_csv("merged_qsar_with_activity.csv", index=False)

print("✅ 'Activity' column added and saved to 'merged_qsar_with_activity.csv'")
