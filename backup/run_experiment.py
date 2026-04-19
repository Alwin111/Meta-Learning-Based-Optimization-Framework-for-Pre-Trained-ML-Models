import sys
import yaml
import csv
import os

from src.models.random_forest import train_random_forest


def log_experiment(model_type, optimization, latency, accuracy, model_size):
    file_path = "experiments/experiment_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model_type", "optimization", "latency", "accuracy", "model_size"])

        writer.writerow([model_type, optimization, latency, accuracy, model_size])


def main():

    if len(sys.argv) < 2:
        print("Usage: python3 run_experiment.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]

    if model_name == "random_forest":
        metrics = train_random_forest(config)
    else:
        raise ValueError("Unsupported model")

    # Extract metrics
    latency = metrics.get("latency")
    accuracy = metrics.get("accuracy")
    model_size = metrics.get("model_size")

    model_type = model_name

    # ✅ CLEAN optimization extraction
    optimization_config = config.get("optimization", {})

    if isinstance(optimization_config, dict):
        optimization = optimization_config.get("type", "baseline")
    else:
        optimization = optimization_config

    # Log experiment
    log_experiment(model_type, optimization, latency, accuracy, model_size)

    return metrics


if __name__ == "__main__":
    main()
