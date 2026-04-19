import os
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    print("Loading trained Random Forest...")
    model = joblib.load("models/random_forest_baseline.pkl")

    # 3072 features (32x32x3 flattened CIFAR)
    initial_type = [("float_input", FloatTensorType([None, 3072]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    os.makedirs("models", exist_ok=True)

    with open("models/random_forest.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("ONNX model saved at models/random_forest.onnx")


if __name__ == "__main__":
    main()
