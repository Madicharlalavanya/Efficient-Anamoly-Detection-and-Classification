import numpy as np
import pandas as pd
import pickle

# Load selected features from saved model
selected_features = pickle.load(open("models/selected_features.pkl", "rb"))

# Number of rows you want
num_samples = 1000

data = {}

for feature in selected_features:
    
    # Generate different ranges for realism
    if "Port" in feature:
        data[feature] = np.random.randint(1, 65535, num_samples)
        
    elif "Duration" in feature or "IAT" in feature:
        data[feature] = np.random.randint(0, 1000000, num_samples)
        
    elif "Pkts" in feature or "Byts" in feature:
        data[feature] = np.random.randint(0, 500000, num_samples)
        
    elif "Size" in feature:
        data[feature] = np.random.uniform(20, 1500, num_samples)
        
    else:
        data[feature] = np.random.uniform(0, 10000, num_samples)

# Create dataframe
df_random = pd.DataFrame(data)

# Save CSV
df_random.to_csv("demo_test.csv", index=False)

print("✅ Random test file created successfully!")