import torch
import os


# -----------------------------
# Random Forest Optimization
# -----------------------------
def optimize_random_forest():
    print("Applying quantization to Random Forest (placeholder)...")
    print("Random Forest optimization complete.")


# -----------------------------
# MobileNet Optimization
# -----------------------------
def optimize_mobilenet():
    print("Applying dynamic quantization to MobileNet...")

    model_path = "models/mobilenet.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError("MobileNet model not found. Train first.")

    from src.models.mobilenet_model import get_model

    device = torch.device("cpu")  # Quantization must run on CPU
    model = get_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Apply dynamic quantization (Linear layers only)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    os.makedirs("models", exist_ok=True)
    torch.save(quantized_model.state_dict(), "models/mobilenet_quantized.pth")

    print("MobileNet quantization complete.")
    print("Saved to models/mobilenet_quantized.pth")


# -----------------------------
# Unified Interface
# -----------------------------
def optimize_model(model_name: str):

    print(f"Optimization step for {model_name}...")

    if model_name == "random_forest":
        optimize_random_forest()

    elif model_name == "mobilenet":
        optimize_mobilenet()

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print("Optimization completed.")
