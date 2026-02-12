import numpy as np
import pandas as pd
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from models.baseline_random_forest import get_model

# ===============================
# CHECKPOINT 1: Load Dataset
# ===============================
print("Loading dataset...")
data = load_iris()
X = data.data
y = data.target

print("Dataset Loaded")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# ===============================
# CHECKPOINT 2: Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test split completed")
print("Training samples:", X_train.shape[0])

# ===============================
# CHECKPOINT 3: Initialize Model
# ===============================
print("Initializing Random Forest...")
model = get_model(n_estimators=100)

# ===============================
# CHECKPOINT 4: Training
# ===============================
print("Training started...")
model.fit(X_train, y_train)
print("Training completed")

# ===============================
# CHECKPOINT 5: Evaluation
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# CHECKPOINT 6: Save Model
# ===============================
os.makedirs("checkpoints", exist_ok=True)

joblib.dump(model, "checkpoints/random_forest_model.pkl")
print("Model saved at checkpoints/random_forest_model.pkl")
