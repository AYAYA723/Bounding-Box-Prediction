import argparse
import csv
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from dataset import CornerDataset
from evaluate import evaluate, save_preview
from model import CornerPredictor
from transformer_model import TransformerCornerPredictor


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["resnet", "vit"], default="resnet")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=root / "results/tuning_20261025")
    args = parser.parse_args()
    args.output = args.output.resolve()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dataset = CornerDataset(root / "datasets/action4/train_pbr/000000")
    model_classes = {"resnet": CornerPredictor, "vit": TransformerCornerPredictor}
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices, val_indices = indices[:16], indices[16:]
    args.output.mkdir(parents=True, exist_ok=True)
    split = {"seed": args.seed, "train": [dataset.label_paths[i].stem for i in train_indices],
             "validation": [dataset.label_paths[i].stem for i in val_indices],
             "video_policy": "test only; video frames are not training or tuning labels"}
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    results = []
    for learning_rate in args.learning_rates:
        torch.manual_seed(args.seed)
        run_dir = args.output / f"lr_{learning_rate:.0e}"
        run_dir.mkdir(exist_ok=False)
        model = model_classes[args.architecture]().to(device)
        config = {"started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  "learning_rate": learning_rate, "epochs": args.epochs,
                  "batch_size": args.batch_size, "seed": args.seed,
                  "train_images": 16, "validation_images": 4,
                  "architecture": args.architecture,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "initialization": "random initialization; no pretrained weights", "optimizer": "AdamW",
                  "transformer_settings": {"patch_size": 32, "hidden_dim": 384, "layers": 6,
                                           "heads": 6, "mlp_dim": 1536, "dropout": 0.1}
                                          if args.architecture == "vit" else None,
                  "weight_decay": 0.01, "loss": "SmoothL1 heatmap only", "device": args.device,
                  "torch": torch.__version__, "torchvision": torchvision.__version__,
                  "checkpoint_selection": "lowest validation mean corner error, evaluated every 10 epochs"}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size,
                            shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed + 1))
        loss_fn = nn.SmoothL1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        best_error = float("inf")
        best_epoch = 0
        start = time.perf_counter()
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "train_heatmap_loss", "validation_heatmap_loss", "validation_corner_error_px", "elapsed_seconds"])
            for epoch in range(1, args.epochs + 1):
                model.train()
                total = 0.0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * x.shape[0]
                train_loss = total / len(train_indices)
                val_loss, val_error = "", ""
                if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
                    metrics = evaluate(model, dataset, val_indices, device, args.batch_size)
                    val_loss, val_error = metrics["heatmap_loss"], metrics["mean_corner_error_px"]
                    if val_error < best_error:
                        best_error, best_epoch = val_error, epoch
                        torch.save(model.state_dict(), run_dir / "best.pt")
                    print(f"lr={learning_rate:g} epoch={epoch}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f} corner={val_error:.2f}px", flush=True)
                writer.writerow([epoch, train_loss, val_loss, val_error, time.perf_counter() - start])
                stream.flush()
        torch.save(model.state_dict(), run_dir / "last.pt")
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
        train_metrics = evaluate(model, dataset, train_indices, device, args.batch_size)
        val_metrics = evaluate(model, dataset, val_indices, device, args.batch_size)
        save_preview(train_metrics, dataset.scene_dir, run_dir / "train_preview")
        save_preview(val_metrics, dataset.scene_dir, run_dir / "validation_preview")
        summary = {"learning_rate": learning_rate, "best_epoch": best_epoch,
                   "seconds": time.perf_counter() - start,
                   "train": train_metrics, "validation": val_metrics}
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        results.append({"learning_rate": learning_rate, "best_epoch": best_epoch,
                        "train_error_px": train_metrics["mean_corner_error_px"],
                        "validation_error_px": val_metrics["mean_corner_error_px"],
                        "validation_within_10px_percent": val_metrics["within_10px_percent"],
                        "best_weights": (run_dir / "best.pt").relative_to(root).as_posix()})
        (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print("RUN COMPLETE: " + json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
