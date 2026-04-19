import numpy as np
import joblib
import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_PATH = "models/random_forest_baseline.pkl"
ONNX_PATH = "models/random_forest_baseline.onnx"
DATA_PATH = "data/processed/cifar10"

def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Loading sample input for shape inference...")
    X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
    X_train = X_train.reshape(X_train.shape[0], -1)

    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    os.makedirs("models", exist_ok=True)
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("ONNX model saved at:", ONNX_PATH)

if __name__ == "__main__":
    main()
