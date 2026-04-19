import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 📁 Check if dataset exists
dataset_path = "experiments/meta_dataset.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError("❌ Dataset not found. Run the app to generate data first.")

# 📊 Load dataset
df = pd.read_csv(dataset_path)

print("📊 Dataset loaded")
print("Total samples:", len(df))
print("Columns:", df.columns.tolist())

# 🚨 Basic validation
required_cols = ["Latency", "Accuracy", "Size (MB)", "Throughput", "Best"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

# 🧠 Features (use extended features if available)
feature_cols = ["Latency", "Accuracy", "Size (MB)", "Throughput"]

# Optional advanced features
if "Dataset_Size" in df.columns:
    feature_cols.append("Dataset_Size")

if "Num_Features" in df.columns:
    feature_cols.append("Num_Features")

X = df[feature_cols]
y = df["Best"]

print("✅ Features used:", feature_cols)

# ✂️ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🌲 Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 📈 Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n📈 Model Evaluation")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 💾 Save model
joblib.dump(model, "meta_model.pkl")

print("\n✅ Meta-model trained and saved successfully!")
