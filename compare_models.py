import numpy as np
import cv2
import onnxruntime as ort

from src.benchmarking.benchmark_inference import benchmark_model

# load dataset
X_test = np.load("data/processed/cifar10/X_test.npy")

img = X_test[0]
img = cv2.resize(img,(224,224))
img = img.astype(np.float32)/255.0
img = np.transpose(img,(2,0,1))
img = np.expand_dims(img,axis=0)

# original model
session_original = ort.InferenceSession("user_input/model/model.onnx")

latency_original = benchmark_model(session_original,img)

# quantized model
session_quant = ort.InferenceSession("user_input/model/model_quantized.onnx")

latency_quant = benchmark_model(session_quant,img)

print("\nMODEL COMPARISON\n")

print("Original model latency:",latency_original)

print("Quantized model latency:",latency_quant)

speedup = latency_original/latency_quant

print("Speedup:",speedup,"x")	
