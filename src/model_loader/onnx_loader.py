import onnxruntime as ort


def load_onnx_model(model_path):
    """
    Loads an ONNX model using ONNX Runtime.
    """
    session = ort.InferenceSession(model_path)
    return session
