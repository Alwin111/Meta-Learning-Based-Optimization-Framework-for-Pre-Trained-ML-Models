import os
import joblib
import time
import numpy as np

# =========================
# MODEL SIZE
# =========================
def get_model_size(model, filename="temp_model.pkl"):
    joblib.dump(model, filename)
    size = os.path.getsize(filename) / (1024 * 1024)
    try:
        os.remove(filename)
    except:
        pass
    return size


# =========================
# MODEL TYPE DETECTION
# =========================
def detect_model_type(model):
    import sklearn.base
    if isinstance(model, sklearn.base.BaseEstimator):
        return "sklearn"
    else:
        return "unknown"


# =========================
# REAL QUANTIZATION (SKLEARN)
# =========================
def simulate_quantization(model):
    """
    Convert float64 → float32 to reduce size
    """
    for attr in dir(model):
        val = getattr(model, attr)
        if isinstance(val, np.ndarray):
            try:
                setattr(model, attr, val.astype(np.float32))
            except:
                pass
    return model


# =========================
# REAL PRUNING (SKLEARN)
# =========================
def simulate_pruning(model, threshold=1e-3):
    """
    Remove small weights
    """
    for attr in dir(model):
        val = getattr(model, attr)
        if isinstance(val, np.ndarray):
            try:
                pruned = np.where(np.abs(val) < threshold, 0, val)
                setattr(model, attr, pruned)
            except:
                pass
    return model


# =========================
# DISTILLATION (OPTIONAL)
# =========================
def simulate_distillation(model, X):
    """
    Train smaller student model
    """
    from sklearn.tree import DecisionTreeClassifier

    student = DecisionTreeClassifier(max_depth=5)
    student.fit(X, model.predict(X))

    return student


# =========================
# EVALUATION
# =========================
def evaluate_model(model, X, y, runs=5):
    times = []

    for _ in range(runs):
        start = time.time()
        preds = model.predict(X)
        end = time.time()
        times.append(end - start)

    latency = sum(times) / len(times)
    throughput = len(X) / latency
    size = get_model_size(model)

    return preds, latency, throughput, size
