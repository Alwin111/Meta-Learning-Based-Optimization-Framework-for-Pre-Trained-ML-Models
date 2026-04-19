import os
import joblib
import time
import numpy as np
import random

# 🔒 Global seed for reproducibility
np.random.seed(42)
random.seed(42)


# 📦 Get model size (more realistic)
def get_model_size(model, filename="temp_model.pkl"):
    joblib.dump(model, filename)
    size = os.path.getsize(filename) / (1024 * 1024)  # MB

    # Cleanup temp file
    try:
        os.remove(filename)
    except:
        pass

    return size


# ⚙️ Simulate Quantization
def simulate_quantization(model):
    """
    Simulate quantization:
    - Strong size reduction
    - Good latency improvement
    - No accuracy change (handled in app if needed)
    """
    np.random.seed(42)

    quant_model = model
    quant_model._size_factor = 0.5
    quant_model._latency_factor = 0.8

    return quant_model


# ✂️ Simulate Pruning
def simulate_pruning(model):
    """
    Simulate pruning:
    - Moderate size reduction
    - Moderate latency improvement
    """
    np.random.seed(42)

    prune_model = model
    prune_model._size_factor = 0.7
    prune_model._latency_factor = 0.85

    return prune_model


# 🧠 NEW: Simulate Distillation
def simulate_distillation(model):
    """
    Simulate knowledge distillation:
    - Moderate size reduction
    - Strong latency improvement
    - Slight accuracy drop (handled in app)
    """
    np.random.seed(42)

    distilled_model = model
    distilled_model._size_factor = 0.6
    distilled_model._latency_factor = 0.75

    return distilled_model


# 📊 Evaluate Model (STABLE VERSION)
def evaluate_model(model, X, y, runs=5):
    """
    Stable evaluation:
    - Averages latency over multiple runs
    - Applies optimization simulation factors
    """

    times = []
    preds = None

    for _ in range(runs):
        start = time.time()
        preds = model.predict(X)
        end = time.time()
        times.append(end - start)

    # ✅ Average latency (stable)
    latency = sum(times) / len(times)

    # ✅ Apply latency factor
    latency_factor = getattr(model, "_latency_factor", 1.0)
    latency *= latency_factor

    # ✅ Throughput
    throughput = len(X) / latency

    # ✅ Size
    size = get_model_size(model)
    size_factor = getattr(model, "_size_factor", 1.0)
    size *= size_factor

    return preds, latency, throughput, size
