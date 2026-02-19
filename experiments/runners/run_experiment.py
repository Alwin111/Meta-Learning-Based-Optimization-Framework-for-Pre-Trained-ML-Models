import yaml
import sys
import torch
import time
import os
import onnxruntime as ort

from src.train_random_forest import train_random_forest
from src.models.mobilenet_model import get_model, benchmark_pytorch
from src.models.onnx_utils import (
    convert_to_onnx,
    verify_onnx,
    convert_pytorch_to_onnx
)
from src.inference.quantization import quantize_onnx_model
from experiments.utils.result_logger import log_results


def benchmark_onnx(model_path, input_data):
    session = ort.InferenceSession(model_path)

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    session.run(None, {input_name: input_data})
    end = time.perf_counter()

    return end - start


def run_experiment(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    optimization_enabled = config["optimization"]["enabled"]
    optimization_type = config["optimization"]["type"]

    print(f"\nRunning experiment: {config['experiment_name']}")
    print(f"Model: {model_name}")
    print(f"Optimization: {optimization_enabled}")

    # -------------------------------
    # RANDOM FOREST PIPELINE
    # -------------------------------
    if model_name == "random_forest":

        model, X_test, accuracy_value = train_random_forest()

        # Save baseline ONNX
        baseline_onnx_path = "models/random_forest_baseline.onnx"
        convert_to_onnx(model, baseline_onnx_path)

        original_time = benchmark_onnx(baseline_onnx_path, X_test.astype("float32"))
        original_size = os.path.getsize(baseline_onnx_path) / (1024 * 1024)

        quantized_time = None
        quantized_size = None
        reduction = None

        if optimization_enabled:
            quantized_onnx_path = "models/random_forest_quantized.onnx"
            quantize_onnx_model(baseline_onnx_path, quantized_onnx_path)

            quantized_time = benchmark_onnx(
                quantized_onnx_path,
                X_test.astype("float32")
            )

            quantized_size = os.path.getsize(quantized_onnx_path) / (1024 * 1024)

            reduction = (
                (original_size - quantized_size) / original_size * 100
            )

    # -------------------------------
    # MOBILENET PIPELINE
    # -------------------------------
    elif model_name == "mobilenet_v2":

        model = get_model()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        baseline_onnx_path = "models/mobilenet_baseline.onnx"
        convert_pytorch_to_onnx(model, dummy_input, baseline_onnx_path)

        original_time = benchmark_onnx(
            baseline_onnx_path,
            dummy_input.numpy()
        )

        original_size = os.path.getsize(baseline_onnx_path) / (1024 * 1024)

        quantized_time = None
        quantized_size = None
        reduction = None
        accuracy_value = "N/A"

        if optimization_enabled:
            quantized_onnx_path = "models/mobilenet_quantized.onnx"
            quantize_onnx_model(baseline_onnx_path, quantized_onnx_path)

            quantized_time = benchmark_onnx(
                quantized_onnx_path,
                dummy_input.numpy()
            )

            quantized_size = os.path.getsize(quantized_onnx_path) / (1024 * 1024)

            reduction = (
                (original_size - quantized_size) / original_size * 100
            )

    else:
        print("Unsupported model")
        sys.exit(1)

    # -------------------------------
    # LOG RESULTS (Task 11)
    # -------------------------------
    result = {
        "experiment_name": config["experiment_name"],
        "model_name": model_name,
        "optimization_type": optimization_type if optimization_enabled else "none",
        "baseline_time_ms": original_time * 1000,
        "optimized_time_ms": quantized_time * 1000 if quantized_time else None,
        "speedup": (original_time / quantized_time) if quantized_time else None,
        "baseline_size_mb": original_size,
        "optimized_size_mb": quantized_size if quantized_size else None,
        "size_reduction_percent": reduction if reduction else None,
        "accuracy": accuracy_value
    }

    log_results(result)

    print("\nExperiment logged successfully.")
    print("Results saved to experiments/results/results.csv")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("Usage: python run_experiment.py --config <config_path>")
        sys.exit(1)

    run_experiment(sys.argv[2])
