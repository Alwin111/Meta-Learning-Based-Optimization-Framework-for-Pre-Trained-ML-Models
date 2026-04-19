from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(input_model, output_model):

    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8
    )

    print("Quantized model saved to:", output_model)
