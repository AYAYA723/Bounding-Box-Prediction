from pathlib import Path

import torch
from torch.utils.data import DataLoader

from predict_corner.dataset import CornerDataset
from predict_corner.model import CornerPredictor
from predict_corner.supervision import BoxDreamerLoss


def main():
    torch.manual_seed(42)
    project_dir = Path(__file__).resolve().parents[2]
    device = torch.device("cuda")

    dataset = CornerDataset(project_dir / "datasets/action4/train_pbr/000000")
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, y, target_corners = next(iter(data_loader))
    x = x.to(device)  # (N, 3, 480, 640)
    y = y.to(device)  # (N, 8, 480, 640)
    target_corners = target_corners.to(device)

    model = CornerPredictor(num_corners=8, weights=None).to(device)
    loss_fn = BoxDreamerLoss()

    model.train()
    pred, pred_corners = model(x)
    loss, coarse, fine = loss_fn(pred, pred_corners, y, target_corners)
    fine_gradient = torch.autograd.grad(fine, model.encoder[0].weight, retain_graph=True)[0]
    model.zero_grad()
    loss.backward()

    print("dataset size:", len(dataset))
    print("input:", x.shape)
    print("target:", y.shape)
    print("output:", pred.shape)
    print("loss:", loss.item())
    print("coarse:", coarse.item(), "fine:", fine.item())
    print("fine-only encoder gradient:", fine_gradient.norm().item())
    print("encoder gradient:", model.encoder[0].weight.grad.norm().item())
    print("head gradient:", model.heatmap_head[0].weight.grad.norm().item())

    model.eval()
    with torch.no_grad():
        pred, corners_xy = model(x)
    print("corners_xy:", corners_xy.shape)


if __name__ == "__main__":
    main()
