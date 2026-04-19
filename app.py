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
import random
import os

# 🔒 Global seed
np.random.seed(42)
random.seed(42)

# 🤖 Load meta-model
meta_model = None
try:
    meta_model = joblib.load("meta_model.pkl")
except:
    meta_model = None

st.set_page_config(page_title="Meta-Learning Framework", layout="centered")

st.title("🚀 Meta-Learning Optimization Framework")
st.markdown("Upload a model and dataset to analyze performance and optimization.")

# Uploads
model_file = st.file_uploader("📦 Upload Model (.pkl)", type=["pkl"])
data_file = st.file_uploader("📊 Upload Dataset (.csv)", type=["csv"])

# Optimization selection
st.subheader("⚙️ Optimization Selection")

run_all = st.checkbox("🚀 Run All Optimizations", value=True)

if not run_all:
    optimization = st.selectbox(
        "Select Single Optimization",
        ["baseline", "quantization", "pruning", "distillation"]
    )
else:
    optimization = "run_all"

st.info("Run all optimizations to compare and automatically select the best one.")

# Deterministic mode
deterministic = st.checkbox("🔒 Deterministic Mode (Stable Results)", value=True)

if deterministic:
    np.random.seed(42)
    random.seed(42)

# Sliders
st.subheader("🎯 Set Optimization Priorities")

w_latency = st.slider("Latency Importance", 0.0, 1.0, 0.4)
w_accuracy = st.slider("Accuracy Importance", 0.0, 1.0, 0.3)
w_size = st.slider("Model Size Importance", 0.0, 1.0, 0.2)
w_throughput = st.slider("Throughput Importance", 0.0, 1.0, 0.1)

# Run
if st.button("▶️ Run Optimization"):

    if model_file and data_file:

        with st.spinner("Running optimization pipeline..."):

            model = joblib.load(model_file)
            data = pd.read_csv(data_file)

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            results_list = []

            # BASELINE
            if run_all or optimization == "baseline":
                preds, lat, thr, size = evaluate_model(model, X, y)
                results_list.append({
                    "Method": "Baseline",
                    "Latency": lat,
                    "Accuracy": accuracy_score(y, preds),
                    "Size (MB)": size,
                    "Throughput": thr,
                    "Model": model
                })

            # QUANTIZATION
            if run_all or optimization == "quantization":
                qm = simulate_quantization(model)
                preds, lat, thr, size = evaluate_model(qm, X, y)
                results_list.append({
                    "Method": "Quantization",
                    "Latency": lat,
                    "Accuracy": accuracy_score(y, preds),
                    "Size (MB)": size,
                    "Throughput": thr,
                    "Model": qm
                })

            # PRUNING
            if run_all or optimization == "pruning":
                pm = simulate_pruning(model)
                preds, lat, thr, size = evaluate_model(pm, X, y)
                results_list.append({
                    "Method": "Pruning",
                    "Latency": lat,
                    "Accuracy": accuracy_score(y, preds),
                    "Size (MB)": size,
                    "Throughput": thr,
                    "Model": pm
                })

            # DISTILLATION (NEW)
            if run_all or optimization == "distillation":
                dm = simulate_distillation(model)
                preds, lat, thr, size = evaluate_model(dm, X, y)
                results_list.append({
                    "Method": "Distillation",
                    "Latency": lat,
                    "Accuracy": accuracy_score(y, preds) * 0.98,  # slight drop
                    "Size (MB)": size,
                    "Throughput": thr,
                    "Model": dm
                })

            results = pd.DataFrame(results_list)

            # Normalize weights
            total = w_latency + w_accuracy + w_size + w_throughput
            if total == 0:
                w_latency_n, w_accuracy_n, w_size_n, w_throughput_n = 0.4, 0.3, 0.2, 0.1
            else:
                w_latency_n = w_latency / total
                w_accuracy_n = w_accuracy / total
                w_size_n = w_size / total
                w_throughput_n = w_throughput / total

            # Normalize metrics
            norm = results.copy()
            for col in ["Latency", "Accuracy", "Size (MB)", "Throughput"]:
                if norm[col].max() != norm[col].min():
                    norm[col] = (norm[col] - norm[col].min()) / (norm[col].max() - norm[col].min())

            # Score (fallback)
            results["Score"] = (
                (1 - norm["Latency"]) * w_latency_n +
                norm["Accuracy"] * w_accuracy_n +
                (1 - norm["Size (MB)"]) * w_size_n +
                norm["Throughput"] * w_throughput_n
            )

            # 🤖 META MODEL
            if meta_model is not None:
                try:
                    X_meta = results[[
                        "Latency", "Accuracy", "Size (MB)", "Throughput"
                    ]]

                    preds_meta = meta_model.predict(X_meta)
                    results["Meta_Pred"] = preds_meta

                    best_row = results.loc[results["Meta_Pred"].idxmax()]
                    st.info("🤖 Using Meta-Learning Model")

                except Exception as e:
                    st.warning(f"Meta-model failed: {e}")
                    best_row = results.loc[results["Score"].idxmax()]
            else:
                best_row = results.loc[results["Score"].idxmax()]
                st.info("⚠️ Using scoring system")

            best_method = best_row["Method"]
            best_model = best_row["Model"]

            # Save dataset
            os.makedirs("experiments", exist_ok=True)

            meta_data = results.drop(columns=["Model"]).copy()
            meta_data["Best"] = (meta_data["Method"] == best_method).astype(int)

            # Extra features
            meta_data["Dataset_Size"] = len(X)
            meta_data["Num_Features"] = X.shape[1]

            meta_data.to_csv(
                "experiments/meta_dataset.csv",
                mode="a",
                header=not os.path.exists("experiments/meta_dataset.csv"),
                index=False
            )

        # OUTPUT UI
        st.success("✅ Run Complete!")

        st.subheader("📊 Results")
        st.dataframe(results.drop(columns=["Model"]))

        if "Meta_Pred" in results.columns:
            st.subheader("🤖 Meta Predictions")
            st.dataframe(results[["Method", "Meta_Pred"]])

        st.subheader("🏆 Best Optimization")
        st.success(best_method)

        # Graph
        st.subheader("📈 Comparison")
        fig, ax = plt.subplots()
        results.set_index("Method")[["Latency", "Accuracy", "Size (MB)"]].plot(kind="bar", ax=ax)
        st.pyplot(fig)

        # Download
        joblib.dump(best_model, "optimized_model.pkl")
        with open("optimized_model.pkl", "rb") as f:
            st.download_button("📥 Download Model", f)

    else:
        st.error("❌ Upload both model and dataset")
