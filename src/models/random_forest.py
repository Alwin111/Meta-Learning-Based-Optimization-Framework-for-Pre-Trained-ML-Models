from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

import os
import joblib


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

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Sklearn Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(model, "checkpoints/random_forest.pkl")

    return model, X_test
