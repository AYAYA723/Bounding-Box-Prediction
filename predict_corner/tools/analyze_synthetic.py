import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from predict_corner.dataset import CornerDataset, RenderedCornerDataset
from predict_corner.model import CornerPredictor
from predict_corner.supervision import BoxDreamerLoss
from predict_corner.tools.evaluate import evaluate as evaluate_original
from predict_corner.tools.train_synthetic import evaluate, save_previews


def group_metrics(metrics, annotations):
    groups = {"all": [], "unoccluded": [], "occluded": [], "blurred": [], "no_blur": [],
              "dark_exposure": [], "normal_exposure": [], "bright_exposure": []}
    for row in metrics["per_image"]:
        index = int(Path(row["image"]).stem)
        aug = annotations[index]
        names = ["all", "unoccluded" if aug["occluder"] == "none" else "occluded",
                 "no_blur" if aug["blur"] == "none" else "blurred"]
        names.append("dark_exposure" if aug["exposure_ev"] < -.3 else
                     "bright_exposure" if aug["exposure_ev"] > .3 else "normal_exposure")
        for name in names:
            groups[name].extend(row["errors_px"])
    return {name: {"images": len(values) // 8, "mean_corner_error_px": sum(values) / len(values),
                    "within_10px_percent": sum(v <= 10 for v in values) / len(values) * 100}
            for name, values in groups.items()}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    config = json.loads((args.run / "config.json").read_text())
    model = CornerPredictor().to(device)
    model.load_state_dict(torch.load(args.run / "best.pt", map_location=device, weights_only=True))
    train_set = RenderedCornerDataset(config["train"])
    loader = DataLoader(train_set, batch_size=8, shuffle=False, num_workers=2)
    training = evaluate(model, loader, device, BoxDreamerLoss())
    (args.run / "train_evaluation.json").write_text(json.dumps(training, indent=2) + "\n")
    original = CornerDataset(root / "datasets/action4/train_pbr/000000")
    indices = torch.randperm(len(original), generator=torch.Generator().manual_seed(42)).tolist()[16:]
    old_holdout_after = evaluate_original(model, original, indices, device)
    model.load_state_dict(torch.load(config["init"], map_location=device, weights_only=True))
    old_holdout_before = evaluate_original(model, original, indices, device)
    baseline = json.loads((args.run / "baseline_validation.json").read_text())
    best = json.loads((args.run / "best_validation.json").read_text())
    annotations = json.loads((Path(config["validation"]) / "augmentation_manifest.json").read_text())["frames"]
    comparison = {"synthetic_validation_before": group_metrics(baseline, annotations),
                  "synthetic_validation_after": group_metrics(best, annotations),
                  "train_eval_mean_corner_error_px": training["mean_corner_error_px"],
                  "original_holdout_ids": [original.label_paths[i].stem for i in indices],
                  "original_holdout_before_px": old_holdout_before["mean_corner_error_px"],
                  "original_holdout_after_px": old_holdout_after["mean_corner_error_px"]}
    val_set = RenderedCornerDataset(config["validation"])
    sample = dict(best)
    rows = best["per_image"]
    sample["per_image"] = [rows[round(i * (len(rows) - 1) / 23)] for i in range(24)]
    save_previews(sample, val_set, args.run / "validation_all_views")
    (args.run / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
