import numpy as np
import joblib
import os
import time
import pandas as pd
import argparse
import torch
from sklearn.metrics import accuracy_score
from src.models.mobilenet_model import get_model

DATA_PATH = "data/processed/cifar10"


def load_data(model_name):
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    if model_name == "random_forest":
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_test, y_test


def measure_inference_time_sklearn(model, X, runs=10):
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(X)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs


def measure_inference_time_mobilenet(model, X_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_time = 0.0

    with torch.no_grad():
        for i in range(X_tensor.shape[0]):
            sample = X_tensor[i:i+1]
            start = time.perf_counter()
            model(sample)
            end = time.perf_counter()
            total_time += (end - start)

    return total_time / X_tensor.shape[0]


def main(model_name):

    print("Loading model...")

    if model_name == "random_forest":
        model_path = "models/random_forest_baseline.pkl"
        model = joblib.load(model_path)

    elif model_name == "mobilenet":
        model_path = "models/mobilenet.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(num_classes=10, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    else:
        raise ValueError("Unsupported model")

    print("Loading test data...")
    X_test, y_test = load_data(model_name)

    print("Measuring accuracy...")

    if model_name == "random_forest":
        predictions = model.predict(X_test)

    elif model_name == "mobilenet":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert HWC → CHW
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        predictions = []

        with torch.no_grad():
            for i in range(X_test_tensor.shape[0]):
                sample = X_test_tensor[i:i+1]
                output = model(sample)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                predictions.append(pred[0])

        predictions = np.array(predictions)

    accuracy = accuracy_score(y_test, predictions)

    print("Measuring inference time...")

    if model_name == "random_forest":
        avg_inference_time = measure_inference_time_sklearn(model, X_test)

    elif model_name == "mobilenet":
        avg_inference_time = measure_inference_time_mobilenet(model, X_test_tensor)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print("\n=== BASELINE RESULTS ===")
    print(f"Model: {model_name}_baseline")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
    print(f"Model Size: {model_size_mb:.2f} MB")

    os.makedirs("results", exist_ok=True)
    results_path = "results/baseline_metrics.csv"

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
    else:
        df = pd.DataFrame()

    new_row = {
        "model": f"{model_name}_baseline",
        "accuracy": accuracy,
        "avg_inference_time_sec": avg_inference_time,
        "model_size_mb": model_size_mb
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(results_path, index=False)

    print("\nResults saved to results/baseline_metrics.csv")

    return {
        "model": f"{model_name}_baseline",
        "accuracy": float(accuracy),
        "avg_inference_time_sec": float(avg_inference_time),
        "model_size_mb": float(model_size_mb)
    }


def benchmark_model(model_name: str):
    return main(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.model)
