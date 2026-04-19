import pandas as pd
from typing import Optional
RESULTS_PATH = "experiments/results/results.csv"
ACCURACY_DROP_THRESHOLD = 0.02  # 2% allowed drop


def load_results() -> pd.DataFrame:
    """Load experiment results CSV."""
    try:
        return pd.read_csv(RESULTS_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def filter_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Filter dataframe for given model."""
    return df[df["model_name"] == model_name]


def calculate_accuracy_drop(
    baseline_row: pd.Series,
    optimized_row: pd.Series
) -> Optional[float]:
    """Calculate accuracy drop if accuracy column exists."""
    if "accuracy" not in baseline_row or "accuracy" not in optimized_row:
        return None

    baseline_acc = baseline_row["accuracy"]
    optimized_acc = optimized_row["accuracy"]

    if pd.isna(baseline_acc) or pd.isna(optimized_acc):
        return None

    return baseline_acc - optimized_acc


def select_best_optimization(
    model_name: str,
    hardware_constraint: str,
    latency_requirement_ms: float
) -> str:
    """
    Select best optimization strategy based on constraints.
    """

    df = load_results()

    if df.empty:
        return "none"

    model_df = filter_model(df, model_name)

    if model_df.empty:
        return "none"

    baseline = model_df[model_df["optimization_type"] == "none"]
    quantized = model_df[model_df["optimization_type"] == "quantization"]

    if baseline.empty or quantized.empty:
        return "none"

    baseline_row = baseline.iloc[0]
    optimized_row = quantized.iloc[0]

    optimized_latency = optimized_row.get("optimized_time_ms", None)
    size_reduction = optimized_row.get("size_reduction_percent", 0)

    # Rule 1 — Accuracy protection
    accuracy_drop = calculate_accuracy_drop(baseline_row, optimized_row)
    if accuracy_drop is not None and accuracy_drop > ACCURACY_DROP_THRESHOLD:
        return "none"

    # Rule 2 — Latency requirement
    if optimized_latency is not None:
        if optimized_latency <= latency_requirement_ms:
            return "quantization"

    # Rule 3 — Edge hardware preference
    if hardware_constraint.lower() == "edge" and size_reduction > 0:
        return "quantization"

    return "none"
