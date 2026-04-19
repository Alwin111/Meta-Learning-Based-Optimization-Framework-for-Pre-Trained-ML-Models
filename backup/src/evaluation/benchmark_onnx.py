import numpy as np
import onnxruntime as ort
import os
import pandas as pd
import argparse

from src.evaluation.metrics import (
    compute_accuracy,
    get_model_size_mb
)

DATA_PATH = "data/processed/cifar10"


def load_data(model_name):
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    if model_name == "random_forest":
        X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    if model_name == "mobilenet":
        X_test = X_test.astype(np.float32)

    return X_test, y_test


def measure_latency(session, X, model_name, runs=5):
    input_name = session.get_inputs()[0].name
    import time

    total = 0.0

    for _ in range(runs):
        start = time.perf_counter()

        if model_name == "mobilenet":
            for i in range(X.shape[0]):
                sample = X[i:i+1]
                session.run(None, {input_name: sample})
        else:
            session.run(None, {input_name: X})

        end = time.perf_counter()
        total += (end - start)

    return total / runs


def main(model_name):
    onnx_path = f"models/{model_name}.onnx"

    print("Loading ONNX model...")
    session = ort.InferenceSession(onnx_path)

    print("Loading test data...")
    X_test, y_test = load_data(model_name)

    input_name = session.get_inputs()[0].name

    print("Running inference...")

    if model_name == "mobilenet":
        predictions = []
        for i in range(X_test.shape[0]):
            sample = X_test[i:i+1]
            output = session.run(None, {input_name: sample})
            pred = np.argmax(output[0], axis=1)
            predictions.append(pred[0])
        predictions = np.array(predictions)
    else:
        outputs = session.run(None, {input_name: X_test})
        predictions = outputs[0]

    print("Computing accuracy...")
    accuracy = compute_accuracy(y_test, predictions)

    print("Measuring latency...")
    avg_time = measure_latency(session, X_test, model_name)

    print("Measuring model size...")
    model_size_mb = get_model_size_mb(onnx_path)

    print("\n=== ONNX RESULTS ===")
    print(f"Model: {model_name}_onnx")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Inference Time: {avg_time:.6f} seconds")
    print(f"Model Size: {model_size_mb:.2f} MB")

    results_path = "results/baseline_metrics.csv"
    os.makedirs("results", exist_ok=True)

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
    else:
        df = pd.DataFrame()

    new_row = {
        "model": f"{model_name}_onnx",
        "accuracy": accuracy,
        "avg_inference_time_sec": avg_time,
        "model_size_mb": model_size_mb
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(results_path, index=False)

    print("\nONNX results appended to results/baseline_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    main(args.model)

