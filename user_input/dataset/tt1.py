import joblib
import numpy as np
import pandas as pd

# =========================
# LOAD MODEL (FIX)
# =========================
model = joblib.load("fakenews.pkl")   # <-- CHANGE THIS PATH

# =========================
# GET REQUIRED FEATURES
# =========================
if hasattr(model, "n_features_in_"):
    n_features = model.n_features_in_
else:
    raise ValueError("❌ Model does not expose n_features_in_")

print(f"Model expects {n_features} features")

# =========================
# CREATE DATASET
# =========================
X_fake = np.random.rand(100, n_features)
y_fake = np.random.randint(0, 2, 100)

columns = [f"feature_{i}" for i in range(n_features)]

df = pd.DataFrame(X_fake, columns=columns)
df["target"] = y_fake

# =========================
# SAVE
# =========================
df.to_csv("compatible_dataset.csv", index=False)

print("✅ Dataset created: compatible_dataset.csv")
