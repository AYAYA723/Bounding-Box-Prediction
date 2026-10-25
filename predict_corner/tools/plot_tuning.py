import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=root / "results/tuning_20261025")
    args = parser.parse_args()
    comparison = json.loads((args.runs / "comparison.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for result in comparison:
        run = args.runs / f"lr_{result['learning_rate']:.0e}"
        with (run / "history.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        label = f"lr={result['learning_rate']:g}"
        line = axes[0].plot([int(row["epoch"]) for row in rows],
                            [float(row["train_heatmap_loss"]) for row in rows], label=label)[0]
        validation = [row for row in rows if row["validation_corner_error_px"]]
        axes[1].plot([int(row["epoch"]) for row in validation],
                     [float(row["validation_corner_error_px"]) for row in validation],
                     marker="o", markersize=3, color=line.get_color(), label=label)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Training heatmap SmoothL1 loss")
    axes[1].set_ylabel("Validation mean corner error (px)")
    figure.savefig(args.runs / "training_curves.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.2), layout="constrained")
    x = list(range(len(comparison)))
    axis.bar([v - 0.18 for v in x], [row["train_error_px"] for row in comparison],
             width=0.36, label="Train: 16 images")
    axis.bar([v + 0.18 for v in x], [row["validation_error_px"] for row in comparison],
             width=0.36, label="Validation: 4 images")
    axis.set_xticks(x, [f"lr={row['learning_rate']:g}\nbest epoch {row['best_epoch']}" for row in comparison])
    axis.set_ylabel("Mean corner error (px)")
    axis.set_title("Checkpoint selected by validation corner error")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(args.runs / "best_checkpoints.png", dpi=180)
    print(args.runs / "training_curves.png")


if __name__ == "__main__":
    main()
