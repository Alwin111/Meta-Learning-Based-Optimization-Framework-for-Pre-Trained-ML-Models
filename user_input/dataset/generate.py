import numpy as np
import pandas as pd

# PARAMETERS (match model)
samples = 100
timesteps = 16
features = 220

# Generate random data
data = np.random.rand(samples, timesteps, features)

# Flatten for CSV (since CSV can't store 3D directly)
data_flat = data.reshape(samples, timesteps * features)

# Create column names
columns = [f"f_{i}" for i in range(timesteps * features)]

df = pd.DataFrame(data_flat, columns=columns)

# Add dummy labels (if needed)
df["target"] = np.random.randint(0, 2, size=samples)

# Save
df.to_csv("onnx_compatible_dataset.csv", index=False)

print("✅ Dataset created: onnx_compatible_dataset.csv")
