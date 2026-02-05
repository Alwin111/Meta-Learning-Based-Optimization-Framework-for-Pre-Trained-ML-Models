import time
import torch

def main():
    print("Baseline environment test started...")

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )

    model.eval()
    x = torch.randn(1, 128)

    start = time.time()
    with torch.no_grad():
        y = model(x)
    end = time.time()

    print("Output shape:", y.shape)
    print("Inference time (ms):", (end - start) * 1000)
    print("Environment OK ✅")

if __name__ == "__main__":
    main()
