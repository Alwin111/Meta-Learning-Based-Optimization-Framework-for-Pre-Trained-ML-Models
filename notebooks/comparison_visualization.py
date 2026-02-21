import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# Configuration
# ----------------------------------------
RESULTS_PATH = "experiments/results/results.csv"
OUTPUT_DIR = "results/plots"

sns.set(style="whitegrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
# Load Results
# ----------------------------------------
df = pd.read_csv(RESULTS_PATH)

print("Loaded Data:\n")
print(df)

# ----------------------------------------
# Keep Only Quantized Runs
# ----------------------------------------
df = df[df["optimization_type"] == "quantization"]

if df.empty:
    print("No quantized experiments found.")
    exit()

# ----------------------------------------
# Clean Columns
# ----------------------------------------
df["baseline_latency_sec"] = df["baseline_time_ms"] / 1000
df["optimized_latency_sec"] = df["optimized_time_ms"] / 1000
df["size_reduction_mb"] = (
    df["baseline_size_mb"] - df["optimized_size_mb"]
)

df["model"] = df["model_name"]

# ----------------------------------------
# 1️⃣ Latency Comparison
# ----------------------------------------
plt.figure()
df.set_index("model")[["baseline_latency_sec", "optimized_latency_sec"]].plot(kind="bar")
plt.title("Baseline vs Optimized Latency")
plt.ylabel("Inference Time (seconds)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/latency_comparison.png")
plt.close()

# ----------------------------------------
# 2️⃣ Speedup
# ----------------------------------------
plt.figure()
df.set_index("model")["speedup"].plot(kind="bar")
plt.title("Inference Speedup (x)")
plt.ylabel("Speedup Factor")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/speedup.png")
plt.close()

# ----------------------------------------
# 3️⃣ Model Size Reduction
# ----------------------------------------
plt.figure()
df.set_index("model")["size_reduction_mb"].plot(kind="bar")
plt.title("Model Size Reduction (MB)")
plt.ylabel("Size Reduced (MB)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/size_reduction.png")
plt.close()

print("\nPlots generated successfully.")
print(f"Saved to: {OUTPUT_DIR}")
