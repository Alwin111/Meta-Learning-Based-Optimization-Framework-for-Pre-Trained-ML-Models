# src/utils/save_results.py

import os
import json
import pandas as pd


def save_experiment_results(model_name: str, results: dict):
    """
    Saves experiment results in both JSON and CSV formats.
    """

    os.makedirs("experiments", exist_ok=True)

    # ---------------------------
    # Save JSON (per experiment)
    # ---------------------------
    json_path = f"experiments/{model_name}_experiment.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # ---------------------------
    # Save CSV (aggregate log)
    # ---------------------------
    csv_path = "experiments/experiment_log.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved:")
    print(f"→ {json_path}")
    print(f"→ {csv_path}")
