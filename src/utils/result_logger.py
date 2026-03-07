import csv
import os
from datetime import datetime

def log_result(model_name, optimization, latency, speedup):

    # ensure experiments folder exists
    os.makedirs("experiments", exist_ok=True)

    file_path = "experiments/results.csv"

    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp","model_name","optimization","latency","speedup"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        writer.writerow([timestamp, model_name, optimization, latency, speedup])
