import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


def get_model(num_classes=10, pretrained=False):
    """
    Returns a configured MobileNetV2 model.
    Makes the model reusable across the framework.
    """

    model = mobilenet_v2(pretrained=pretrained)

    # Replace classifier for CIFAR-10
    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model
