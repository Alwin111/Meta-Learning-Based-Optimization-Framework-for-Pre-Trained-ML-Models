import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier

dataset_path = "experiments/meta_dataset.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError("Dataset not found")

df = pd.read_csv(dataset_path)

feature_cols = ["Latency", "Accuracy", "Size (MB)", "Throughput"]

if "Dataset_Size" in df.columns:
    feature_cols.append("Dataset_Size")

if "Num_Features" in df.columns:
    feature_cols.append("Num_Features")

X = df[feature_cols]
y = df["Best"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(n_estimators=150)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "meta_model.pkl")

print("✅ Improved Meta-model saved")
