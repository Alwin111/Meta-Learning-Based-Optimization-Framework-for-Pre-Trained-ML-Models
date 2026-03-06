import sys
import yaml

from src.models.random_forest import train_random_forest


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_experiment.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ✅ FIXED LINE
    model_name = config["model_name"]

    if model_name == "random_forest":
        train_random_forest(config)
    else:
        raise ValueError("Unsupported model")


if __name__ == "__main__":
    main()
