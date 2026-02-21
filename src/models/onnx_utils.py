import time
import numpy as np
import os

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import torch


# ---------------------------------------------------
# SKLEARN → ONNX (Random Forest)
# ---------------------------------------------------
def convert_to_onnx(model, output_path):
    print("Converting model to ONNX...")

    initial_type = [("float_input", FloatTensorType([None, model.n_features_in_]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("ONNX model saved:", output_path)
    return output_path


# ---------------------------------------------------
# VERIFY ONNX
# ---------------------------------------------------
def verify_onnx(onnx_path, X_sample):
    print("Verifying ONNX model...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    _ = session.run(None, {input_name: X_sample.astype(np.float32)})
    print("ONNX inference successful.")


# ---------------------------------------------------
# BENCHMARK (OPTIONAL)
# ---------------------------------------------------
def benchmark_models(sklearn_model, onnx_path, X_sample):
    print("\nBenchmarking Inference Speed...")

    start = time.time()
    sklearn_model.predict(X_sample)
    sklearn_time = time.time() - start

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    start = time.time()
    session.run(None, {input_name: X_sample.astype(np.float32)})
    onnx_time = time.time() - start

    print(f"Sklearn Inference Time: {sklearn_time*1000:.4f} ms")
    print(f"ONNX Inference Time: {onnx_time*1000:.4f} ms")

    if onnx_time > 0:
        speedup = sklearn_time / onnx_time
        print(f"Speedup: {speedup:.2f}x faster")


# ---------------------------------------------------
# PYTORCH → ONNX (MobileNet)
# ---------------------------------------------------
def convert_pytorch_to_onnx(model, dummy_input, output_path):
    """
    Converts a PyTorch model to ONNX.
    output_path must be full path (including directory).
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )

    print("ONNX model saved:", output_path)
    return output_path
