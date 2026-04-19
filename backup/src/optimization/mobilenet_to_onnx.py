import torch
import os
from src.models.mobilenet_model import get_model


def convert_mobilenet_to_onnx():
    print("Loading trained MobileNet...")

    # Correct checkpoint path
    checkpoint_path = "models/mobilenet.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} not found")

    model = get_model(num_classes=10, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs("models", exist_ok=True)

    dummy_input = torch.randn(1, 3, 32, 32)

    print("Exporting to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        "models/mobilenet.onnx",
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print("ONNX model exported successfully at models/mobilenet.onnx")


if __name__ == "__main__":
    convert_mobilenet_to_onnx()
