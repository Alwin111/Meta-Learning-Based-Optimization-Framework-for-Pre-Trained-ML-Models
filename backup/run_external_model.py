from src.model_loader.onnx_loader import load_onnx_model
from src.dataset.load_dataset import load_dataset
from src.inference.run_inference import run_inference

import numpy as np


MODEL_PATH = "user_input/model/model.onnx"
DATASET_PATH = "user_input/dataset/test.csv"


def main():

    print("Loading model...")
    session = load_onnx_model(MODEL_PATH)

    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)

    print("Running inference...")

    predictions, latency = run_inference(session, X.astype(np.float32))

    print("Inference latency:", latency)


if __name__ == "__main__":
    main()
