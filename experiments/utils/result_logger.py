import os
import csv
from datetime import datetime

RESULTS_DIR = "experiments/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")


def log_results(result_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(result_dict)
