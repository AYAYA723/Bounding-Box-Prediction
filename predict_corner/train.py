import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from predict_corner.dataset import CornerDataset
from predict_corner.model import CornerPredictor
from predict_corner.supervision import BoxDreamerLoss


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
    loss_fn = BoxDreamerLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    run_dir = project_dir / "results/paper_train" / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "seed": 42, "train_images": len(dataset), "loss": "coarse + 2*fine",
        "coordinate_decoder": "pre-Sigmoid logit soft-argmax, temperature=0.25",
        "decoder_temperature": model.decoder_temperature,
        "normalization": "2*xy/[W,H]-1"}, indent=2) + "\n", encoding="utf-8")
    log = (run_dir / "history.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(log)
    writer.writerow(["epoch", "total_loss", "coarse_loss", "fine_loss"])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_coarse, total_fine = 0.0, 0.0
        for x, y, corners_xy in train_loader:
            x = x.to(device)  # (N, 3, 480, 640)
            y = y.to(device)  # (N, 8, 480, 640)
            corners_xy = corners_xy.to(device)
            pred, pred_corners = model(x)
            loss, coarse, fine = loss_fn(pred, pred_corners, y, corners_xy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
            total_coarse += coarse.item() * x.shape[0]
            total_fine += fine.item() * x.shape[0]

        print(f"Epoch {epoch + 1}/{num_epochs}, total: {total_loss / len(dataset):.6f}, "
              f"coarse: {total_coarse / len(dataset):.6f}, fine: {total_fine / len(dataset):.6f}", flush=True)
        writer.writerow([epoch + 1, total_loss / len(dataset), total_coarse / len(dataset), total_fine / len(dataset)])
        log.flush()

    log.close()
    torch.save(model.state_dict(), run_dir / "last.pt")
    checkpoint_dir = project_dir / "predict_corner/checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "corner_predictor_paper.pt")


if __name__ == "__main__":
    main()
