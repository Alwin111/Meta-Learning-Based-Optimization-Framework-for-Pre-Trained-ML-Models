import numpy as np
import joblib
import os
import pandas as pd

from src.evaluation.metrics import (
    compute_accuracy,
    measure_latency_sklearn,
    get_model_size_mb
)

DATA_PATH = "data/processed/cifar10"
MODEL_PATH = "models/random_forest_baseline.pkl"


def load_data():
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    # Flatten images
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_test, y_test


def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading test data...")
    X_test, y_test = load_data()

    print("Running inference...")
    predictions = model.predict(X_test)

    print("Computing accuracy...")
    accuracy = compute_accuracy(y_test, predictions)

    print("Measuring latency...")
    avg_inference_time = measure_latency_sklearn(model, X_test)

    print("Measuring model size...")
    model_size_mb = get_model_size_mb(MODEL_PATH)

    print("\n=== BASELINE RESULTS ===")
    print(f"Accuracy: {accuracy}")
    print(f"Average Inference Time (full test set): {avg_inference_time:.6f} seconds")
    print(f"Model Size: {model_size_mb:.2f} MB")

    # Save results
    results = pd.DataFrame([{
        "model": "random_forest_baseline",
        "accuracy": accuracy,
        "avg_inference_time_sec": avg_inference_time,
        "model_size_mb": model_size_mb
    }])

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/baseline_metrics.csv", index=False)

    print("\nResults saved to results/baseline_metrics.csv")


if __name__ == "__main__":
    main()
