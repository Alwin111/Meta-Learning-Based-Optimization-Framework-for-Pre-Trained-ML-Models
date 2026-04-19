import onnxruntime as ort
import numpy as np
import torchvision
import torchvision.transforms as transforms


def evaluate():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    session = ort.InferenceSession("mobilenet.onnx")

    correct = 0
    total = 0

    for image, label in testset:
        image = image.unsqueeze(0).numpy()
        outputs = session.run(None, {"input": image})
        pred = np.argmax(outputs[0])

        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total
    print("ONNX Accuracy:", accuracy)


if __name__ == "__main__":
    evaluate()
