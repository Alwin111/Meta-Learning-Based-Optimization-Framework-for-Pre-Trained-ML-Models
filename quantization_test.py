from src.optimization.quantize_model import quantize_onnx_model

input_model = "user_input/model/model.onnx"
output_model = "user_input/model/model_quantized.onnx"

quantize_onnx_model(input_model, output_model)
