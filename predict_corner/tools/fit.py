import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from predict_corner.dataset import CornerDataset, AppearanceDataset
from predict_corner.model import CornerPredictor
from predict_corner.supervision import BoxDreamerLoss
from predict_corner.tools.evaluate import evaluate, save_preview


def replace_batch_norm(module):
    """Keep normalization comparisons in the experiment script."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            normalization = torch.nn.GroupNorm(8, child.num_features, eps=child.eps)
            normalization.weight = child.weight
            normalization.bias = child.bias
            setattr(module, name, normalization)
        else:
            replace_batch_norm(child)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, choices=[1, 16], default=1)
    parser.add_argument("--image-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--target-error", type=float, default=1.0)
    parser.add_argument("--max-error", type=float, default=3.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--normalization", choices=["batch", "group"], default="batch")
    parser.add_argument("--init", type=Path)
    parser.add_argument("--warmup-temperature", type=float)
    parser.add_argument("--anneal-steps", type=int, default=1000)
    parser.add_argument("--selection", choices=["train", "validation"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.hub.set_dir(str(root / ".cache/torch"))
    device = torch.device("cuda")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = CornerDataset(root / "datasets/action4/train_pbr/000000")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = [args.image_id] if args.images == 1 else indices[:16]
    val_indices = indices[16:]
    samples = [dataset[i] for i in train_indices]
    if args.augment:
        masks = []
        for index in train_indices:
            path = dataset.scene_dir / "mask_visib" / f"{dataset.label_paths[index].stem}_000000.png"
            with Image.open(path) as mask:
                masks.append(mask.convert("L"))
        samples = AppearanceDataset(samples, masks)
    loader = DataLoader(samples, batch_size=min(2, len(samples)), shuffle=True,
                        generator=torch.Generator().manual_seed(43))
    weights = ResNet18_Weights.DEFAULT if args.pretrained else None
    model = CornerPredictor(weights=weights, decoder_temperature=args.temperature)
    if args.normalization == "group":
        replace_batch_norm(model)
    model = model.to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    loss_fn = BoxDreamerLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    config = vars(args).copy()
    config.update({"output": str(output), "init": str(args.init) if args.init else None,
                   "initialization_source": str(args.init) if args.init else ("ImageNet1K" if args.pretrained else "random"),
                   "seed": 42, "batch_size": min(2, len(samples)),
                   "loss": "coarse + 2*fine", "scheduler": "cosine, eta_min=1e-6",
                   "train_images": [dataset.label_paths[i].stem for i in train_indices],
                   "validation_images": [dataset.label_paths[i].stem for i in val_indices],
                   "started_at": datetime.now().astimezone().isoformat(),
                   "source_sha256": {name: hashlib.sha256((root / "predict_corner" / name).read_bytes()).hexdigest()
                                     for name in ("model.py", "dataset.py", "supervision.py", "tools/fit.py")}})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best_error = float("inf")
    best_step = 0
    reached_target = False
    step = 0
    started = time.perf_counter()
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "epoch", "learning_rate", "temperature", "total", "coarse", "fine",
                         "train_error_px", "train_max_error_px", "validation_error_px", "seconds"])
        for epoch in range(math.ceil(args.steps / len(loader))):
            for x, y, target_xy in loader:
                model.train()
                if args.freeze_bn:
                    for module in model.modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            module.eval()
                initial_temperature = args.warmup_temperature or args.temperature
                fraction = max(0.0, 1 - step / args.anneal_steps)
                model.decoder_temperature = args.temperature + (initial_temperature - args.temperature) * fraction
                training_temperature = model.decoder_temperature
                pred, xy = model(x.to(device))
                total, coarse, fine = loss_fn(pred, xy, y.to(device), target_xy.to(device))
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                learning_rate = optimizer.param_groups[0]["lr"]
                scheduler.step()
                step += 1
                train_error, train_max, val_error = "", "", ""
                if step == 1 or step % args.eval_every == 0 or step == args.steps:
                    model.decoder_temperature = args.temperature
                    train_metrics = evaluate(model, dataset, train_indices, device)
                    val_metrics = evaluate(model, dataset, val_indices, device)
                    train_error = train_metrics["mean_corner_error_px"]
                    train_max = max(value for row in train_metrics["per_image"] for value in row["errors_px"])
                    val_error = val_metrics["mean_corner_error_px"]
                    selection_error = train_error if args.selection == "train" else val_error
                    if selection_error < best_error:
                        best_error, best_step = selection_error, step
                        torch.save(model.state_dict(), output / "best.pt")
                    reached_target = train_error <= args.target_error and train_max <= args.max_error
                    print(f"step={step}/{args.steps} total={total.item():.6f} coarse={coarse.item():.6f} "
                          f"fine={fine.item():.6f} train={train_error:.3f}px max={train_max:.3f}px "
                          f"validation={val_error:.3f}px", flush=True)
                writer.writerow([step, epoch + 1, learning_rate, training_temperature,
                                 total.item(), coarse.item(), fine.item(),
                                 train_error, train_max, val_error, time.perf_counter() - started])
                stream.flush()
                if step == args.steps or reached_target:
                    break
            if step == args.steps or reached_target:
                break
    torch.save(model.state_dict(), output / "last.pt")
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    train_metrics = evaluate(model, dataset, train_indices, device)
    val_metrics = evaluate(model, dataset, val_indices, device)
    save_preview(train_metrics, dataset.scene_dir, output / "train_preview")
    save_preview(val_metrics, dataset.scene_dir, output / "validation_preview")
    summary = {"best_step": best_step, "steps_completed": step, "reached_target": reached_target,
               "seconds": time.perf_counter() - started, "train": train_metrics, "validation": val_metrics}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE best_step={best_step}, train={train_metrics['mean_corner_error_px']:.3f}px, "
          f"validation={val_metrics['mean_corner_error_px']:.3f}px, reached_target={reached_target}", flush=True)


if __name__ == "__main__":
    main()
