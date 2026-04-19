import os
import joblib
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from src.meta_learning.optimization_selector import recommend_optimization
from src.utils.result_logger import log_meta_result


# ===============================
# 📥 LOAD MODEL
# ===============================
model_folder = "user_input/model"

model_file = None
for file in os.listdir(model_folder):
    if file.endswith(".pkl") and "optimized" not in file:
        model_file = os.path.join(model_folder, file)
        break

if model_file is None:
    raise Exception("❌ No model found")

model = joblib.load(model_file)
print(f"✅ Loaded Model: {model_file}")


# ===============================
# 📥 LOAD DATASET
# ===============================
dataset_folder = "user_input/dataset"

dataset_file = None
for file in os.listdir(dataset_folder):
    if file.endswith(".csv"):
        dataset_file = os.path.join(dataset_folder, file)
        break

if dataset_file is None:
    raise Exception("❌ No dataset found")

df = pd.read_csv(dataset_file)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"✅ Loaded Dataset: {dataset_file}")


# ===============================
# ⏱️ BASELINE METRICS
# ===============================
start = time.time()
y_pred_base = model.predict(X)
end = time.time()

baseline_latency = end - start
baseline_accuracy = accuracy_score(y, y_pred_base)
baseline_size = os.path.getsize(model_file) / (1024 * 1024)


# ===============================
# 🔧 MULTIPLE OPTIMIZATION TECHNIQUES
# ===============================
results = []

# Baseline
results.append({
    "technique": "baseline",
    "latency": baseline_latency,
    "accuracy": baseline_accuracy,
    "size": baseline_size,
    "model": model
})


def evaluate_model(name, model_obj):
    model_obj.fit(X, y)

    start = time.time()
    pred = model_obj.predict(X)
    latency = time.time() - start

    acc = accuracy_score(y, pred)

    temp_path = f"temp_{name}.pkl"
    joblib.dump(model_obj, temp_path)
    size = os.path.getsize(temp_path) / (1024 * 1024)

    return {
        "technique": name,
        "latency": latency,
        "accuracy": acc,
        "size": size,
        "model": model_obj
    }


# 🔹 Quantization (simulated)
results.append(evaluate_model(
    "quantization",
    RandomForestClassifier(n_estimators=20, max_depth=5)
))

# 🔹 Pruning
results.append(evaluate_model(
    "pruning",
    RandomForestClassifier(n_estimators=5, max_depth=4)
))

# 🔹 Hyperparameter Optimization
results.append(evaluate_model(
    "hyperparameter_optimization",
    RandomForestClassifier(n_estimators=150, max_depth=10)
))


# ===============================
# 🧠 META LEARNING (DECIDES BEST)
# ===============================
model_name = "random_forest_external"
model_type = "tabular"

recommended = recommend_optimization(model_name)


# ===============================
# 🧠 SELECT BEST MODEL (FROM META)
# ===============================
best = None

for r in results:
    if r["technique"] == recommended:
        best = r
        break

# fallback safety
if best is None:
    best = results[1]


# Save best model
best_model_path = "user_input/model/best_optimized_model.pkl"
joblib.dump(best["model"], best_model_path)


# ===============================
# 💾 LOG RESULTS
# ===============================
log_meta_result(
    model_name=model_name,
    model_type=model_type,
    optimization=best["technique"],
    accuracy=best["accuracy"],
    latency=best["latency"],
    model_size=best["size"]
)


# ===============================
# 🎯 FINAL OUTPUT
# ===============================
print("\n================= OPTIMIZATION COMPARISON =================")

for r in results:
    print(f"\nTechnique: {r['technique']}")
    print(f"Latency:   {r['latency']:.6f} sec")
    print(f"Accuracy:  {r['accuracy']:.4f}")
    print(f"Size:      {r['size']:.2f} MB")

print("\n🏆 BEST TECHNIQUE:", best["technique"])
print(f"\nMeta-Learning Recommendation: {recommended}")

print(f"\n✅ Best model saved at: {best_model_path}")

print("===========================================================")
