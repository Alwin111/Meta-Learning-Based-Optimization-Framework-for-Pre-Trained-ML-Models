import numpy as np
import cv2
import onnxruntime as ort

# Load dataset
X_test = np.load("data/processed/cifar10/X_test.npy")

# Resize first image to 224x224
img = X_test[0]

img_resized = cv2.resize(img, (224,224))

# Normalize
img_resized = img_resized.astype(np.float32) / 255.0

# Convert to NCHW format
img_resized = np.transpose(img_resized, (2,0,1))

# Add batch dimension
img_resized = np.expand_dims(img_resized, axis=0)

# Load ONNX model
session = ort.InferenceSession("user_input/model/model.onnx")

input_name = session.get_inputs()[0].name

output = session.run(None, {input_name: img_resized})

print("Prediction shape:", output[0].shape)
