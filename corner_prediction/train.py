from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import CornerDataset
from model import CornerPredictor


def main():
    torch.manual_seed(42)
    project_dir = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")
    num_epochs = 100
    batch_size = 2
    learning_rate = 1e-4

    # Dataset / DataLoader
    dataset = CornerDataset(project_dir / "datasets/action4/train_pbr/000000")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # nn.Module / loss / optimizer
    model = CornerPredictor(num_corners=8, weights=None).to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)  # (N, 3, 480, 640)
            y = y.to(device)  # (N, 8, 480, 640)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]

        print(f"Epoch {epoch + 1}/{num_epochs}, loss: {total_loss / len(dataset):.6f}", flush=True)

    torch.save(model.state_dict(), project_dir / "corner_prediction/corner_predictor.pt")


if __name__ == "__main__":
    main()
