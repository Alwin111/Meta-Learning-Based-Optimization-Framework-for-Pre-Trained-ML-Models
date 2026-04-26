from utils.optimizer import (
    evaluate_model,
    simulate_quantization,
    simulate_pruning,
    simulate_distillation
)

from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# =========================
# UI
# =========================
st.set_page_config(page_title="Meta-Learning Framework", layout="centered")

st.title("🚀 Meta-Learning Optimization Framework")
st.markdown("Upload a model and dataset to analyze performance and optimization.")

model_file = st.file_uploader("📦 Upload Model (.pkl)", type=["pkl"])
data_file = st.file_uploader("📊 Upload Dataset (.csv)", type=["csv"])

# =========================
# Optimization Selection
# =========================
st.subheader("⚙️ Optimization Selection")

run_all = st.checkbox("🚀 Run All Optimizations", value=True)

if not run_all:
    selected_method = st.selectbox(
        "Select Optimization",
        ["Baseline", "Quantization", "Pruning", "Distillation"]
    )
else:
    selected_method = "ALL"

# =========================
# Weights
# =========================
st.subheader("🎯 Set Optimization Priorities")

w_latency = st.slider("Latency Importance", 0.0, 1.0, 0.4)
w_accuracy = st.slider("Accuracy Importance", 0.0, 1.0, 0.3)
w_size = st.slider("Model Size Importance", 0.0, 1.0, 0.2)
w_throughput = st.slider("Throughput Importance", 0.0, 1.0, 0.1)

# =========================
# RUN
# =========================
if st.button("▶️ Run Optimization"):

    if model_file and data_file:

        model = joblib.load(model_file)

        data = pd.read_csv(data_file)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Keep DataFrame to avoid sklearn warnings
        X_input = X.fillna(0)

        results_list = []
        model_map = {}

        def run_method(name, model_obj):
            preds, latency, throughput, size = evaluate_model(model_obj, X_input, y)

            results_list.append({
                "Method": name,
                "Latency": latency,
                "Accuracy": accuracy_score(y, preds),
                "Size (MB)": size,
                "Throughput": throughput
            })

            model_map[name] = model_obj

        # =========================
        # Run Methods
        # =========================
        if run_all or selected_method == "Baseline":
            run_method("Baseline", model)

        if run_all or selected_method == "Quantization":
            run_method("Quantization", simulate_quantization(model))

        if run_all or selected_method == "Pruning":
            run_method("Pruning", simulate_pruning(model))

        if run_all or selected_method == "Distillation":
            run_method("Distillation", simulate_distillation(model, X_input))

        results = pd.DataFrame(results_list)

        # =========================
        # LOG NORMALIZED SCORING
        # =========================
        norm = results.copy()

        norm["Latency"] = 1 / norm["Latency"]
        norm["Size (MB)"] = 1 / norm["Size (MB)"]

        for col in ["Latency", "Size (MB)", "Throughput"]:
            norm[col] = np.log1p(norm[col])

        metrics = ["Latency", "Accuracy", "Size (MB)", "Throughput"]

        norm[metrics] = (norm[metrics] - norm[metrics].min()) / (
            norm[metrics].max() - norm[metrics].min() + 1e-8
        )

        results["Score"] = (
            norm["Latency"] * w_latency +
            norm["Accuracy"] * w_accuracy +
            norm["Size (MB)"] * w_size +
            norm["Throughput"] * w_throughput
        )

        best_method = results.loc[results["Score"].idxmax()]["Method"]

        # =========================
        # OUTPUT
        # =========================
        st.success("✅ Optimization Complete!")
        st.dataframe(results)
        st.success(f"🏆 Best Method: {best_method}")

        # =========================
        # 📊 METRIC GRAPH
        # =========================
        st.subheader("📊 Metric Comparison")

        fig, ax = plt.subplots(figsize=(10, 5))
        norm.set_index("Method")[metrics].plot(kind="bar", ax=ax)

        plt.xticks(rotation=0)
        plt.ylabel("Normalized Score (0–1)")
        plt.tight_layout()
        st.pyplot(fig)

        # =========================
        # 📈 SCORE GRAPH
        # =========================
        st.subheader("📈 Overall Score Comparison")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        results.set_index("Method")["Score"].plot(kind="bar", ax=ax2)

        for i, v in enumerate(results["Score"]):
            ax2.text(i, v, f"{v:.2f}", ha='center')

        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig2)

        # =========================
        # DOWNLOAD FIX
        # =========================
        best_model = model_map[best_method]

        output_path = "optimized_model.pkl"
        joblib.dump(best_model, output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Optimized Model",
                data=f,
                file_name="optimized_model.pkl",
                mime="application/octet-stream"
            )

    else:
        st.error("❌ Please upload model and dataset.")
