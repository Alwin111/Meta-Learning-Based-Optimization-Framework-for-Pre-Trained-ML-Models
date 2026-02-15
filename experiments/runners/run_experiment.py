import yaml
import sys
import torch
import time
import os
import onnxruntime as ort

from src.models.random_forest import train_random_forest
from src.models.mobilenet_model import get_model, benchmark_pytorch
from src.models.onnx_utils import (
    convert_to_onnx,
    verify_onnx,
    convert_pytorch_to_onnx
)

from src.inference.quantization import quantize_onnx_model


# =====================================================
# CONFIG LOADER
# =====================================================

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# =====================================================
# ONNX BENCHMARK
# =====================================================

def benchmark_onnx_model(onnx_path, input_data):
    import numpy as np

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    if isinstance(input_data, np.ndarray):
        input_data = input_data.astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: input_data})

    start = time.time()
    session.run(None, {input_name: input_data})
    return time.time() - start


# =====================================================
# MAIN RUNNER
# =====================================================

def run_experiment(config):

    print("\nLoaded Configuration:")
    print(config)

    model_name = config["model"]["name"]

    opt_cfg = config.get("optimization", {})
    optimization_enabled = opt_cfg.get("enable", False)
    optimization_type = opt_cfg.get("type", "none")

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    if model_name == "random_forest":

        model, X_test = train_random_forest(config)

        print("\nExporting Random Forest to ONNX...")
        onnx_path = convert_to_onnx(model, X_test)
        verify_onnx(onnx_path, X_test)

        print("\nBenchmarking Original ONNX Model...")
        original_time = benchmark_onnx_model(onnx_path, X_test[:1])
        print(f"Original ONNX Inference Time: {original_time*1000:.4f} ms")

        # ONNX Quantization (Correct for RF)
        if optimization_enabled and optimization_type == "quantization":

            print("\nApplying ONNX Dynamic Quantization...")

            suffix = opt_cfg.get("save_suffix", "_quantized")
            quantized_path = onnx_path.replace(".onnx", f"{suffix}.onnx")

            weight_type = opt_cfg.get("quantization", {}).get("weight_type", "qint8")

            quantize_onnx_model(
                input_model_path=onnx_path,
                output_model_path=quantized_path,
                weight_type_str=weight_type
            )

            print("\nBenchmarking Quantized ONNX Model...")
            quantized_time = benchmark_onnx_model(quantized_path, X_test[:1])
            print(f"Quantized ONNX Inference Time: {quantized_time*1000:.4f} ms")

            if quantized_time > 0:
                print(f"\n🚀 Speedup: {(original_time/quantized_time):.2f}x faster")

    # =====================================================
    # MOBILENET V2
    # =====================================================
    elif model_name == "mobilenet_v2":

        model = get_model(num_classes=10, pretrained=False)
        model.eval()

        dummy_input = torch.randn(1, 3, 32, 32)

        print("\nBenchmarking PyTorch Inference...")
        pytorch_time = benchmark_pytorch(model, dummy_input)
        print(f"PyTorch Inference Time: {pytorch_time*1000:.4f} ms")

        print("\nExporting MobileNetV2 to ONNX...")
        onnx_path = convert_pytorch_to_onnx(
            model,
            dummy_input,
            filename="mobilenet_v2.onnx"
        )

        print("\nBenchmarking Original ONNX Model...")
        original_time = benchmark_onnx_model(onnx_path, dummy_input.numpy())
        print(f"Original ONNX Inference Time: {original_time*1000:.4f} ms")

        # 🔥 USE PYTORCH QUANTIZATION FOR CNN (Correct Approach)
        if optimization_enabled and optimization_type == "quantization":

            print("\nApplying PyTorch Dynamic Quantization...")

            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},   # Only linear layers
                dtype=torch.qint8
            )

            print("\nBenchmarking Quantized PyTorch Model...")
            quantized_time = benchmark_pytorch(quantized_model, dummy_input)
            print(f"Quantized PyTorch Inference Time: {quantized_time*1000:.4f} ms")

            if quantized_time > 0:
                print(f"\n🚀 Speedup: {(pytorch_time/quantized_time):.2f}x faster")

    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python -m experiments.runners.run_experiment <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    run_experiment(config)
