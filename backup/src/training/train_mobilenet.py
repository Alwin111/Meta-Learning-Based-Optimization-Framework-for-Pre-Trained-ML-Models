import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.models.mobilenet_model import get_model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Data Preparation
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Small subset for faster experimentation
    trainset = torch.utils.data.Subset(full_trainset, range(1000))
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # -----------------------------
    # Model (Reusable)
    # -----------------------------
    model = get_model(num_classes=10, pretrained=False).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # Training Loop
    # -----------------------------
    model.train()

    for epoch in range(2):
        running_loss = 0.0

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Completed | Total Loss: {running_loss:.4f}")

    # -----------------------------
    # Save Model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/mobilenet.pth"
    torch.save(model.state_dict(), model_path)

    print(f"\nModel saved successfully at {model_path}")


if __name__ == "__main__":
    train()
