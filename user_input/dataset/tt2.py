import joblib
import numpy as np
import pandas as pd

# =========================
# LOAD MODEL
# =========================
model = joblib.load("fakenews.pkl")  # change path

# =========================
# GET FEATURE COUNT (FIXED)
# =========================
if hasattr(model, "n_features_in_"):
    n_features = model.n_features_in_

elif hasattr(model, "coef_"):
    n_features = model.coef_.shape[1]

elif hasattr(model, "support_vectors_"):
    n_features = model.support_vectors_.shape[1]

else:
    # fallback (manual guess)
    print("⚠️ Could not detect feature count. Defaulting to 10")
    n_features = 10

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
