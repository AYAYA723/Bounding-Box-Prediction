from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CornerDataset
from model import CornerPredictor


def main():
    torch.manual_seed(42)
    project_dir = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")

    dataset = CornerDataset(project_dir / "datasets/action4/train_pbr/000000")
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, y = next(iter(data_loader))
    x = x.to(device)  # (N, 3, 480, 640)
    y = y.to(device)  # (N, 8, 480, 640)

    model = CornerPredictor(num_corners=8, weights=None).to(device)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    pred = model(x)
    loss = loss_fn(pred, y)
    model.zero_grad()
    loss.backward()

    print("dataset size:", len(dataset))
    print("input:", x.shape)
    print("target:", y.shape)
    print("output:", pred.shape)
    print("loss:", loss.item())
    print("encoder gradient:", model.encoder[0].weight.grad.norm().item())
    print("head gradient:", model.heatmap_head[0].weight.grad.norm().item())

    model.eval()
    with torch.no_grad():
        pred = model(x)
        N, C, H, W = pred.shape
        indices = pred.flatten(start_dim=2).argmax(dim=2)  # (N, 8)
        corners_xy = torch.stack((indices % W, indices // W), dim=-1)  # (N, 8, 2)
    print("corners_xy:", corners_xy.shape)


if __name__ == "__main__":
    main()
