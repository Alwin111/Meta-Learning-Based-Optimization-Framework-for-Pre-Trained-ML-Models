import numpy as np
import onnxruntime as ort
import os
import pandas as pd

from src.evaluation.metrics import (
    compute_accuracy,
    measure_latency_onnx,
    get_model_size_mb
)

DATA_PATH = "data/processed/cifar10"
ONNX_PATH = "models/random_forest_baseline.onnx"


def load_data():
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    # Flatten + convert to float32 for ONNX
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    return X_test, y_test


def main():
    print("Loading ONNX model...")
    session = ort.InferenceSession(ONNX_PATH)

    print("Loading test data...")
    X_test, y_test = load_data()

    input_name = session.get_inputs()[0].name

    print("Running inference...")
    outputs = session.run(None, {input_name: X_test})
    predictions = outputs[0]  # ONNX returns class labels

    print("Computing accuracy...")
    accuracy = compute_accuracy(y_test, predictions)

    print("Measuring latency...")
    avg_time = measure_latency_onnx(session, X_test)

    print("Measuring model size...")
    model_size_mb = get_model_size_mb(ONNX_PATH)

    print("\n=== ONNX RESULTS ===")
    print(f"Accuracy: {accuracy}")
    print(f"Average Inference Time: {avg_time:.6f} seconds")
    print(f"Model Size: {model_size_mb:.2f} MB")

    # Append results to CSV
    results_path = "results/baseline_metrics.csv"

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
    else:
        df = pd.DataFrame()

    new_row = {
        "model": "random_forest_onnx",
        "accuracy": accuracy,
        "avg_inference_time_sec": avg_time,
        "model_size_mb": model_size_mb
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(results_path, index=False)

    print("\nONNX results appended to results/baseline_metrics.csv")


if __name__ == "__main__":
    main()
