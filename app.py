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
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Meta-Learning Framework", layout="centered")

st.title("🚀 Meta-Learning Optimization Framework")
st.markdown("Upload a model and dataset to analyze performance and optimization.")

# =========================
# FILE UPLOAD
# =========================
model_file = st.file_uploader("📦 Upload Model (.pkl)", type=["pkl"])
data_file = st.file_uploader("📊 Upload Dataset (.csv)", type=["csv"])

# =========================
# OPTIMIZATION SELECTION
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
# PRIORITY SLIDERS
# =========================
st.subheader("🎯 Set Optimization Priorities")

w_latency = st.slider("Latency Importance", 0.0, 1.0, 0.4)
w_accuracy = st.slider("Accuracy Importance", 0.0, 1.0, 0.3)
w_size = st.slider("Model Size Importance", 0.0, 1.0, 0.2)
w_throughput = st.slider("Throughput Importance", 0.0, 1.0, 0.1)

# =========================
# RUN BUTTON
# =========================
if st.button("▶️ Run Optimization"):

    if model_file is not None and data_file is not None:

        with st.spinner("Running optimization..."):

            model = joblib.load(model_file)
            data = pd.read_csv(data_file)

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            results_list = []
            model_map = {}

            def run_method(name, model_obj):
                preds, latency, throughput, size = evaluate_model(model_obj, X, y)

                results_list.append({
                    "Method": name,
                    "Latency": latency,
                    "Accuracy": accuracy_score(y, preds),
                    "Size (MB)": size,
                    "Throughput": throughput
                })

                model_map[name] = model_obj

            # =========================
            # RUN METHODS
            # =========================
            if run_all or selected_method == "Baseline":
                run_method("Baseline", model)

            if run_all or selected_method == "Quantization":
                quant_model = simulate_quantization(joblib.load(model_file))
                run_method("Quantization", quant_model)

            if run_all or selected_method == "Pruning":
                prune_model = simulate_pruning(joblib.load(model_file))
                run_method("Pruning", prune_model)

            if run_all or selected_method == "Distillation":
                distill_model = simulate_distillation(joblib.load(model_file), X)
                run_method("Distillation", distill_model)

            results = pd.DataFrame(results_list)

            # =========================
            # NORMALIZE WEIGHTS
            # =========================
            total = w_latency + w_accuracy + w_size + w_throughput

            if total == 0:
                w_latency_n, w_accuracy_n, w_size_n, w_throughput_n = 0.4, 0.3, 0.2, 0.1
            else:
                w_latency_n = w_latency / total
                w_accuracy_n = w_accuracy / total
                w_size_n = w_size / total
                w_throughput_n = w_throughput / total

            # =========================
            # SCORING
            # =========================
            results["Score"] = (
                (1 / results["Latency"]) * w_latency_n +
                results["Accuracy"] * w_accuracy_n +
                (1 / results["Size (MB)"]) * w_size_n +
                results["Throughput"] * w_throughput_n
            )

            best_method = results.loc[results["Score"].idxmax()]["Method"]

            # =========================
            # META MODEL (FIXED)
            # =========================
            meta_prediction = None

            if os.path.exists("meta_model.pkl"):

                if len(results) < 2:
                    meta_prediction = "⚠️ Meta-model requires multiple methods to compare"
                else:
                    meta_model = joblib.load("meta_model.pkl")

                    try:
                        required_features = list(meta_model.feature_names_in_)
                        avg_row = results.mean(numeric_only=True).to_frame().T

                        # Add missing features dynamically
                        for col in required_features:
                            if col not in avg_row.columns:
                                if col == "Dataset_Size":
                                    avg_row[col] = len(X)
                                elif col == "Num_Features":
                                    avg_row[col] = X.shape[1]
                                else:
                                    avg_row[col] = 0

                        avg_features = avg_row[required_features]

                        meta_prediction = meta_model.predict(avg_features)[0]

                    except Exception as e:
                        meta_prediction = f"Error: {str(e)}"

            # =========================
            # EXPLANATION
            # =========================
            best_row = results.iloc[results["Score"].idxmax()]

            if len(results) == 1:
                explanation = f"""
                **{best_method} was selected because it was the only method executed.**

                Metrics:
                - Latency: {round(best_row['Latency'], 5)}
                - Accuracy: {round(best_row['Accuracy'], 4)}
                - Size: {round(best_row['Size (MB)'], 4)} MB
                - Throughput: {round(best_row['Throughput'], 2)}

                👉 Run multiple optimizations for better comparison.
                """
            else:
                explanation = f"""
                **{best_method} was selected because:**

                - It achieved the highest score: {round(best_row['Score'], 4)}
                - Latency: {round(best_row['Latency'], 5)}
                - Accuracy: {round(best_row['Accuracy'], 4)}
                - Size: {round(best_row['Size (MB)'], 4)} MB
                - Throughput: {round(best_row['Throughput'], 2)}

                Based on your priorities:
                - Latency weight: {round(w_latency_n, 2)}
                - Accuracy weight: {round(w_accuracy_n, 2)}
                - Size weight: {round(w_size_n, 2)}
                - Throughput weight: {round(w_throughput_n, 2)}

                👉 The system selected the best trade-off across all metrics.
                """

        # =========================
        # OUTPUT
        # =========================
        st.success("✅ Optimization Complete!")

        st.subheader("📊 Full Comparison")
        st.dataframe(results)

        st.subheader("🏆 Best Optimization")
        st.success(best_method)

        if meta_prediction is not None:
            st.subheader("🧠 Meta-Model Prediction")
            st.info(meta_prediction)

        st.subheader("📌 Why This Method Was Chosen")
        st.markdown(explanation)

        st.subheader("⚖️ Final Weights Used")
        st.write({
            "Latency": round(w_latency_n, 2),
            "Accuracy": round(w_accuracy_n, 2),
            "Size": round(w_size_n, 2),
            "Throughput": round(w_throughput_n, 2)
        })

        # =========================
        # GRAPH
        # =========================
        st.subheader("📈 Multi-Metric Comparison")

        fig, ax = plt.subplots()
        results.set_index("Method")[["Latency", "Accuracy", "Size (MB)"]].plot(kind="bar", ax=ax)
        st.pyplot(fig)

        # =========================
        # DOWNLOAD
        # =========================
        best_model = model_map[best_method]

        joblib.dump(best_model, "optimized_model.pkl")

        with open("optimized_model.pkl", "rb") as f:
            st.download_button(
                "📥 Download Optimized Model",
                f,
                file_name="optimized_model.pkl"
            )

    else:
        st.error("❌ Please upload both model and dataset.")
