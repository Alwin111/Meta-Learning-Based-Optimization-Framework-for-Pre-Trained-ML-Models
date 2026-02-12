import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score
from src.models.baseline_random_forest import get_model

DATA_PATH = "data/processed/cifar10"

def load_data():
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    # Flatten images for Random Forest
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test

def train():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    print("Initializing model...")
    model = get_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {accuracy}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_baseline.pkl")

    print("Model saved successfully!")

if __name__ == "__main__":
    train()
