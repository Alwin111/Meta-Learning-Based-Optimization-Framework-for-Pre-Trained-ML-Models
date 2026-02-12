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


def measure_latency_onnx(session, X, runs=10):
    """
    Measure average inference time for ONNX Runtime model.
    """
    input_name = session.get_inputs()[0].name
    total_time = 0.0

    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: X})
        end = time.perf_counter()
        total_time += (end - start)

    return total_time / runs


def get_model_size_mb(model_path):
    """
    Get model file size in MB.
    """
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)
