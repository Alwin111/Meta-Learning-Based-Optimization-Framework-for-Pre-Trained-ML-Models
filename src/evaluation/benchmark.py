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


# -----------------------------
# Load Data
# -----------------------------
def load_data(model_name):
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    if model_name == "random_forest":
        # Flatten CIFAR images (32x32x3 → 3072)
        X_test = X_test.reshape(X_test.shape[0], -1)

    if model_name == "mobilenet":
        X_test = X_test.astype(np.float32)

    return X_test, y_test


# -----------------------------
# Sklearn Inference Timing
# -----------------------------
def measure_inference_time_sklearn(model, X, runs=10):
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(X)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs


# -----------------------------
# MobileNet Inference Timing
# -----------------------------
def measure_inference_time_mobilenet(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_time = 0.0

    with torch.no_grad():
        for i in range(X.shape[0]):
            sample = torch.tensor(X[i:i+1]).to(device)
            start = time.perf_counter()
            model(sample)
            end = time.perf_counter()
            total_time += (end - start)

    return total_time / X.shape[0]


# -----------------------------
# Main Benchmark
# -----------------------------
def main(model_name):
    print("Loading model...")

    if model_name == "random_forest":
        model_path = "models/random_forest_baseline.pkl"
        model = joblib.load(model_path)

    elif model_name == "mobilenet":
        model_path = "models/mobilenet.pth"   # ✅ FIXED PATH
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(num_classes=10, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    else:
        raise ValueError("Unsupported model")

    print("Loading test data...")
    X_test, y_test = load_data(model_name)

    print("Measuring accuracy...")

    if model_name == "random_forest":
        predictions = model.predict(X_test)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(X_test.shape[0]):
                sample = torch.tensor(X_test[i:i+1]).to(device)
                output = model(sample)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                predictions.append(pred[0])

        predictions = np.array(predictions)

    accuracy = accuracy_score(y_test, predictions)

    print("Measuring inference time...")

    if model_name == "random_forest":
        avg_inference_time = measure_inference_time_sklearn(model, X_test)
    else:
        avg_inference_time = measure_inference_time_mobilenet(model, X_test)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print("\n=== BASELINE RESULTS ===")
    print(f"Model: {model_name}_baseline")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
    print(f"Model Size: {model_size_mb:.2f} MB")

    # Save Results
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.model)
