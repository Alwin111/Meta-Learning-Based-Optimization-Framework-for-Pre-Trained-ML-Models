import pickle
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/cifar-10-batches-py")
PROCESSED_DIR = Path("data/processed/cifar10")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_batch(batch_path):
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    X = batch[b"data"]
    y = batch[b"labels"]
    return X, y


def preprocess_cifar10():
    print("🔹 Loading CIFAR-10 training batches...")
    X_train_list, y_train_list = [], []

    for i in range(1, 6):
        batch_file = RAW_DIR / f"data_batch_{i}"
        X, y = load_batch(batch_file)
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)

    print("🔹 Loading CIFAR-10 test batch...")
    X_test, y_test = load_batch(RAW_DIR / "test_batch")

    print("🔹 Reshaping images...")
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    print("🔹 Normalizing pixel values...")
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    print("🔹 Saving processed data...")
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print("✅ CIFAR-10 preprocessing complete!")
    print(f"Saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_cifar10()
