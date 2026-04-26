import numpy as np
import pandas as pd

# detect required features
n_features = model.n_features_in_

# create dummy dataset
X_fake = np.random.rand(100, n_features)

# create fake labels (binary)
y_fake = np.random.randint(0, 2, 100)

# convert to dataframe (important)
columns = [f"feature_{i}" for i in range(n_features)]
df = pd.DataFrame(X_fake, columns=columns)
df["target"] = y_fake

# save
df.to_csv("compatible_dataset.csv", index=False)

print("✅ Dataset created: compatible_dataset.csv")
