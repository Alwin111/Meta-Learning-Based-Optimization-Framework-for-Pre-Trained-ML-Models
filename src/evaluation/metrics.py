import time
import os
import numpy as np
from sklearn.metrics import accuracy_score


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return accuracy_score(y_true, y_pred)


def measure_latency_sklearn(model, X, runs=10):
    """
    Measure average inference time for sklearn model.
    """
    total_time = 0.0

    for _ in range(runs):
        start = time.perf_counter()
        model.predict(X)
        end = time.perf_counter()
        total_time += (end - start)

    return total_time / runs


def measure_latency_onnx(session, X):
    """
    Measure average inference time PER SAMPLE
    using full batch execution.
    Requires ONNX model exported with dynamic batch size.
    """
    input_name = session.get_inputs()[0].name

    # Warmup run (important for fair timing)
    session.run(None, {input_name: X[:1]})

    start = time.perf_counter()
    session.run(None, {input_name: X})
    end = time.perf_counter()

    total_time = end - start

    # Return average per sample
    return total_time / X.shape[0]


def get_model_size_mb(model_path):
    """
    Get model file size in MB.
    """
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)
