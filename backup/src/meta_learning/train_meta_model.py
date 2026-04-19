import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("experiments/experiment_log.csv")

print("📊 Dataset loaded:")
print(df.head())

# Features (input to meta-model)
X = df[["latency", "model_size", "accuracy"]]

# Target (what we want to predict)
y = df["optimization"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "meta_model.pkl")

print("✅ Meta-learning model trained and saved as meta_model.pkl")
