import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/processed/cifar10"


def load_data():
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    # Flatten images (32x32x3 → 3072)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, X_test, y_train, y_test


def main():
    print("Loading CIFAR dataset...")
    X_train, X_test, y_train, y_test = load_data()

    print("Training SMALL Random Forest...")

    model = RandomForestClassifier(
        n_estimators=50,      # smaller forest
        max_depth=10,         # limit tree depth
        n_jobs=-1,
        random_state=42
    )

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_end = time.perf_counter()

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    infer_start = time.perf_counter()
    model.predict(X_test)
    infer_end = time.perf_counter()

    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_baseline.pkl"
    joblib.dump(model, model_path)

    print("\n=== TRAINING RESULTS ===")
    print("Model: random_forest_baseline")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {(train_end - train_start):.6f} seconds")
    print(f"Inference Time: {(infer_end - infer_start):.6f} seconds")
    print(f"Model Size: {os.path.getsize(model_path)/(1024*1024):.2f} MB")
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    main()
