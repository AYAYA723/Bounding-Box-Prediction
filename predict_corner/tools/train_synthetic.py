import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import optim
from torch.utils.data import DataLoader

from predict_corner.dataset import RenderedCornerDataset
from predict_corner.model import CornerPredictor
from predict_corner.supervision import BoxDreamerLoss, make_target_heatmaps


def evaluate(model, loader, device, loss_fn):
    model.eval()
    errors, records, losses = [], [], [0., 0., 0.]
    offset = 0
    with torch.no_grad():
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            heatmaps, xy = model(x)
            truth = make_target_heatmaps(target, heatmaps.shape[-2:])
            total, coarse, fine = loss_fn(heatmaps, xy, truth, target)
            count = x.shape[0]
            for index, value in enumerate((total, coarse, fine)):
                losses[index] += value.item() * count
            distances = (xy - target).norm(dim=-1).cpu()
            for item in range(count):
                records.append({"image": f"{int(loader.dataset.ids[offset + item]):06d}.png",
                                "gt_xy": target[item].cpu().tolist(), "pred_xy": xy[item].cpu().tolist(),
                                "errors_px": distances[item].tolist()})
            errors.extend(distances.flatten().tolist())
            offset += count
    values = torch.tensor(errors)
    return {"image_count": offset, "mean_corner_error_px": values.mean().item(),
            "median_corner_error_px": values.median().item(),
            "within_5px_percent": (values <= 5).float().mean().item() * 100,
            "within_10px_percent": (values <= 10).float().mean().item() * 100,
            "total_loss": losses[0] / offset, "coarse_loss": losses[1] / offset,
            "fine_loss": losses[2] / offset, "per_image": records}


def save_previews(report, dataset, output):
    output.mkdir(exist_ok=True)
    edges = json.loads((dataset.scene_dir / "keypoints_3d.json").read_text())["edges"]
    sheet = Image.new("RGB", (1600, 6 * 324), "white")
    for index, record in enumerate(report["per_image"][:24]):
        with Image.open(dataset.scene_dir / "rgb" / record["image"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for key, color in (("gt_xy", "#30ef70"), ("pred_xy", "#ff5050")):
            points = record[key]
            for a, b in edges:
                draw.line((tuple(points[a]), tuple(points[b])), fill=color, width=2)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        image.save(output / record["image"])
        x, y = index % 4 * 400, index // 4 * 324
        sheet.paste(image.resize((400, 300)), (x, y))
        ImageDraw.Draw(sheet).text((x + 5, y + 304), f"{record['image']} / {sum(record['errors_px']) / 8:.1f}px", fill="black")
    sheet.save(output / "gt_green_pred_red.jpg", quality=94)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=root / "datasets/action4/synthetic_v2/train")
    parser.add_argument("--validation", type=Path, default=root / "datasets/action4/synthetic_v2/val")
    parser.add_argument("--init", type=Path, default=root / "predict_corner/checkpoints/corner_predictor_paper.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    train_set, val_set = RenderedCornerDataset(args.train), RenderedCornerDataset(args.validation)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    model = CornerPredictor(decoder_temperature=.25).to(device)
    model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    loss_fn = BoxDreamerLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"train_images": len(train_set), "validation_images": len(val_set), "seed": 42,
                   "loss": "coarse + 2*fine", "temperature": .25, "precision": "BF16 training; float32 coordinate decode and evaluation",
                   "architecture": "unchanged ResNet-18 + five-stage transpose-convolution decoder + BatchNorm",
                   "init_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest(),
                   "started_at": datetime.now().astimezone().isoformat(),
                   "source_sha256": {name: hashlib.sha256((root / "predict_corner" / name).read_bytes()).hexdigest()
                                     for name in ("model.py", "dataset.py", "supervision.py", "tools/train_synthetic.py")}})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    baseline = evaluate(model, val_loader, device, loss_fn)
    (args.output / "baseline_validation.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"BASELINE validation={baseline['mean_corner_error_px']:.3f}px", flush=True)
    best_error, best_epoch = float("inf"), 0
    history = []
    started = time.perf_counter()
    with (args.output / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "learning_rate", "total", "coarse", "fine", "train_error_px",
                         "validation_error_px", "validation_5px_percent", "validation_10px_percent", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = [0., 0., 0., 0.]
            count = 0
            for batch_index, (x, target) in enumerate(train_loader):
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
                truth = make_target_heatmaps(target, x.shape[-2:])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    heatmaps, xy = model(x)
                    total, coarse, fine = loss_fn(heatmaps, xy, truth, target)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                N = x.shape[0]
                values = (total.item(), coarse.item(), fine.item(), (xy.detach() - target).norm(dim=-1).mean().item())
                for i, value in enumerate(values):
                    sums[i] += value * N
                count += N
                if (batch_index + 1) % 25 == 0:
                    print(f"epoch={epoch} batch={batch_index + 1}/{len(train_loader)} loss={total.item():.5f}", flush=True)
            metrics = evaluate(model, val_loader, device, loss_fn)
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()
            row = [epoch, learning_rate, *(value / count for value in sums), metrics["mean_corner_error_px"],
                   metrics["within_5px_percent"], metrics["within_10px_percent"], time.perf_counter() - started]
            writer.writerow(row)
            stream.flush()
            history.append(dict(zip(["epoch", "learning_rate", "total", "coarse", "fine", "train_error_px",
                                      "validation_error_px", "validation_5px_percent", "validation_10px_percent", "seconds"], row)))
            if metrics["mean_corner_error_px"] < best_error:
                best_error, best_epoch = metrics["mean_corner_error_px"], epoch
                torch.save(model.state_dict(), args.output / "best.pt")
                (args.output / "best_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            torch.save(model.state_dict(), args.output / "last.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "epoch": epoch, "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state()},
                       args.output / "training_state.pt")
            print(f"EPOCH {epoch}/{args.epochs} train={sums[3] / count:.2f}px validation={metrics['mean_corner_error_px']:.2f}px "
                  f"best={best_error:.2f}px elapsed={time.perf_counter() - started:.1f}s", flush=True)
            if epoch - best_epoch >= args.patience:
                break
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True))
    final_metrics = evaluate(model, val_loader, device, loss_fn)
    save_previews(final_metrics, val_set, args.output / "validation_preview")
    summary = {"best_epoch": best_epoch, "epochs_completed": epoch, "seconds": time.perf_counter() - started,
               "baseline_validation_error_px": baseline["mean_corner_error_px"], "validation": final_metrics}
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("total", "coarse", "fine"):
        axes[0].plot([r["epoch"] for r in history], [r[name] for r in history], label=name)
    for name in ("train_error_px", "validation_error_px"):
        axes[1].plot([r["epoch"] for r in history], [r[name] for r in history], label=name)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.output / "training_curves.png", dpi=160)
    plt.close(fig)
    print(f"COMPLETE best_epoch={best_epoch}, validation={best_error:.3f}px", flush=True)


if __name__ == "__main__":
    main()
