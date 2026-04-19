import joblib
import numpy as np
import os

MODEL_PATH = "models/meta_model.pkl"

if os.path.exists(MODEL_PATH):
    meta_model = joblib.load(MODEL_PATH)
else:
    meta_model = None

def predict_best_optimization(X):
    if meta_model is None:
        return "Meta-model not found"

    num_samples = X.shape[0]
    num_features = X.shape[1]

    # 🔥 ADD THIRD FEATURE (IMPORTANT FIX)
    dummy_latency = 0.01  # placeholder

    features = np.array([[num_samples, num_features, dummy_latency]])

    prediction = meta_model.predict(features)

    return prediction[0]
