import numpy as np
import cv2
import onnxruntime as ort

from src.benchmarking.benchmark_inference import benchmark_model

# load cifar sample
X_test = np.load("data/processed/cifar10/X_test.npy")

img = X_test[0]
img = cv2.resize(img,(224,224))
img = img.astype(np.float32)/255.0
img = np.transpose(img,(2,0,1))
img = np.expand_dims(img,axis=0)

session = ort.InferenceSession("user_input/model/model.onnx")

latency = benchmark_model(session,img)

print("Average inference latency:",latency)
