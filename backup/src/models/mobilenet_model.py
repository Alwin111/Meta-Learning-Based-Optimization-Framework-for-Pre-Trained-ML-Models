import torch
import torch.nn as nn
import time
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def get_model(num_classes=10, pretrained=False):
    """
    Returns a configured MobileNetV2 model.
    Makes the model reusable across the framework.
    Uses modern torchvision weights API (no deprecated warnings).
    """

    # Handle pretrained weights correctly (new API)
    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
    else:
        weights = None

    model = mobilenet_v2(weights=weights)

    # Replace classifier for CIFAR-10
    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model


def benchmark_pytorch(model, dummy_input, runs=50):
    """
    Benchmarks PyTorch inference with warmup + averaging
    for stable measurement.
    """

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Timed runs
    with torch.no_grad():
        start = time.time()
        for _ in range(runs):
            _ = model(dummy_input)
        pytorch_time = (time.time() - start) / runs

    print(f"PyTorch Inference Time (avg over {runs} runs): {pytorch_time*1000:.4f} ms")
    return pytorch_time
