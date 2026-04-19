# src/inference/quantization.py

import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(
    input_model_path: str,
    output_model_path: str,
    weight_type_str: str = "qint8",
):
    """
    Applies dynamic quantization to an ONNX model.

    Args:
        input_model_path (str): Path to original ONNX model
        output_model_path (str): Path to save quantized model
        weight_type_str (str): 'qint8' or 'quint8'

    Returns:
        str: Path to quantized model
    """

    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"ONNX model not found at {input_model_path}")

    # Map string to ONNX QuantType
    weight_type_map = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
    }

    weight_type = weight_type_map.get(weight_type_str.lower())

    if weight_type is None:
        raise ValueError(
            f"Unsupported weight_type '{weight_type_str}'. "
            f"Supported types: {list(weight_type_map.keys())}"
        )

    print("\n🔄 Applying Dynamic Quantization...")
    print(f"Weight Type: {weight_type_str}")

    # 🔥 KEY FIX: Load model first to avoid internal shape inference conflict
    model = onnx.load(input_model_path)

    quantize_dynamic(
        model_input=model,  # pass model object instead of file path
        model_output=output_model_path,
        weight_type=weight_type,
    )

    print(f"✅ Quantized model saved at: {output_model_path}")

    return output_model_path
