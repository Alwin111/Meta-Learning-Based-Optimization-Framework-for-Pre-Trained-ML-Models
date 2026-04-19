from config.config import NUM_BENCHMARK_SAMPLES, QUANT_MODEL_PATH
import numpy as np
import cv2
import onnxruntime as ort

from src.benchmarking.benchmark_inference import benchmark_model
from src.optimization.quantize_model import quantize_onnx_model
from src.utils.result_logger import log_result


def preprocess_image(img):

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def run_experiment(model_path, dataset_path, model_name):

    # load dataset
    X_test = np.load(dataset_path)

    imgs = X_test[:NUM_BENCHMARK_SAMPLES] 
    # load baseline model
    session = ort.InferenceSession(model_path)

    latencies = []

    for img in imgs:

        input_data = preprocess_image(img)

        latency = benchmark_model(session, input_data)

        latencies.append(latency)

    baseline_latency = sum(latencies) / len(latencies)

    print("Baseline latency:", baseline_latency)

    log_result(model_name, "baseline", baseline_latency, 1.0)

    # quantize model
    quant_model = QUANT_MODEL_PATH
    quantize_onnx_model(model_path, quant_model)

    session_quant = ort.InferenceSession(quant_model)

    latencies_quant = []

    for img in imgs:

        input_data = preprocess_image(img)

        latency = benchmark_model(session_quant, input_data)

        latencies_quant.append(latency)

    quant_latency = sum(latencies_quant) / len(latencies_quant)

    speedup = baseline_latency / quant_latency

    print("Quantized latency:", quant_latency)
    print("Speedup:", speedup)

    log_result(model_name, "quantization", quant_latency, speedup)
