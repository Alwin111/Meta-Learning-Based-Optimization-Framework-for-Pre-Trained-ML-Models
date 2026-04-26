import joblib
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score

# LOAD MODELS
orig = joblib.load("fakenews.pkl")
opt = joblib.load("optimized_fakenews.pkl")

# LOAD DATASET (use your compatible_dataset.csv)
data = pd.read_csv("fake_news.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ---------- ORIGINAL ----------
start = time.time()
preds_orig = orig.predict(X)
lat_orig = time.time() - start

# ---------- OPTIMIZED ----------
start = time.time()
preds_opt = opt.predict(X)
lat_opt = time.time() - start

# ---------- METRICS ----------
acc_orig = accuracy_score(y, preds_orig)
acc_opt = accuracy_score(y, preds_opt)

size_orig = os.path.getsize("fakenews.pkl") / (1024 * 1024)
size_opt = os.path.getsize("optimized_fakenews.pkl") / (1024 * 1024)

# ---------- PRINT ----------
print("\n===== FINAL COMPARISON =====")
print(f"Accuracy: {acc_orig:.4f} → {acc_opt:.4f}")
print(f"Latency: {lat_orig:.4f}s → {lat_opt:.4f}s")
print(f"Size: {size_orig:.4f}MB → {size_opt:.4f}MB")

# ---------- DEEP CHECKS ----------
print("\n===== DEEP ANALYSIS =====")

# 1. Prediction similarity
similarity = (preds_orig == preds_opt).mean()
print(f"Prediction match: {similarity:.4f}")

# 2. Unique predictions
print(f"Original unique classes: {np.unique(preds_orig)}")
print(f"Optimized unique classes: {np.unique(preds_opt)}")

# 3. Distribution
print("\nOriginal prediction distribution:")
print(pd.Series(preds_orig).value_counts(normalize=True))

print("\nOptimized prediction distribution:")
print(pd.Series(preds_opt).value_counts(normalize=True))
