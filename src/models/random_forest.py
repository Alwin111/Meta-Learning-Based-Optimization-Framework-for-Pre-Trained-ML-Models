from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

import os
import joblib
import time


def train_random_forest(config):

    print("Training Random Forest from config...")

    data = load_iris()
    X = data.data
    y = data.target

    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    n_estimators = config["model"]["n_estimators"]
    max_depth = config["model"]["max_depth"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # Accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Sklearn Accuracy: {accuracy:.4f}")

    # Inference latency measurement
    start = time.time()
    model.predict(X_test)
    end = time.time()

    latency = (end - start) / len(X_test)

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/random_forest.pkl"
    joblib.dump(model, model_path)

    # Model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)

    return {
        "accuracy": accuracy,
        "avg_inference_time_sec": latency,
        "model_size_mb": model_size
    }
