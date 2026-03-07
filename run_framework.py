from src.experiments.run_experiment import run_experiment
from config.config import MODEL_PATH, DATASET_PATH

run_experiment(MODEL_PATH, DATASET_PATH, "mobilenet")
