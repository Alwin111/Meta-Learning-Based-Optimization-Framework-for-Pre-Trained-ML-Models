import os
import numpy as np
from torchvision import datasets
from torchvision import transforms

SAVE_PATH = "data/processed/cifar10"

os.makedirs(SAVE_PATH, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

X_train = np.array([img.numpy() for img, _ in train_dataset])
y_train = np.array([label for _, label in train_dataset])

X_test = np.array([img.numpy() for img, _ in test_dataset])
y_test = np.array([label for _, label in test_dataset])

np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)

print("CIFAR10 dataset prepared and saved successfully.")
