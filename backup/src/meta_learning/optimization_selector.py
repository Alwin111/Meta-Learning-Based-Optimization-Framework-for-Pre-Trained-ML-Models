# src/meta_learning/optimization_selector.py

import joblib
import pandas as pd


def recommend_optimization(latency, model_size, accuracy):
    """
    Recommend best optimization using trained meta-learning model
    """

    # Load trained meta-model
    model = joblib.load("meta_model.pkl")

    # Create DataFrame (fixes sklearn warning)
    input_data = pd.DataFrame(
        [[latency, model_size, accuracy]],
        columns=["latency", "model_size", "accuracy"]
    )

    print(f"Inputs → Latency: {latency}, Size: {model_size}, Accuracy: {accuracy}")

    # Predict
    prediction = model.predict(input_data)

    recommendation = prediction[0]

    print(f"[Meta Model] Recommended: {recommendation}")

    return recommendation
