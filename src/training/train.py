import numpy as np
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from src.models.baseline_random_forest import get_model as get_rf_model
from src.models.mobilenet_model import get_model as get_mobilenet_model

DATA_PATH = "data/processed/cifar10"


# -----------------------------
# Load Data
# -----------------------------
def load_data(flatten=False):
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test


# -----------------------------
# Random Forest Training
# -----------------------------
def train_random_forest():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(flatten=True)

    print("Initializing Random Forest...")
    model = get_rf_model()

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_baseline.pkl")

    print("Random Forest model saved.")


# -----------------------------
# MobileNet Training
# -----------------------------
def train_mobilenet():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(flatten=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_mobilenet_model(num_classes=10, pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert HWC → CHW
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 🔥 Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("Training MobileNet (1 epoch)...")
    model.train()

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Evaluating...")
    model.eval()
    correct = 0

    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch = X_test[i:i+64].to(device)
            labels = y_test_tensor[i:i+64].to(device)

            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(X_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mobilenet.pth")

    print("MobileNet model saved.")

# -----------------------------
# Unified Interface
# -----------------------------
def train_model(model_name: str):

    if model_name == "random_forest":
        train_random_forest()

    elif model_name == "mobilenet":
        train_mobilenet()

    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    train_model("random_forest")
