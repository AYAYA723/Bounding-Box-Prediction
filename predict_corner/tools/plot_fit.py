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
    parser.add_argument("--runs", type=Path, default=root / "results/decoder_fix_20261025")
    args = parser.parse_args()
    names = [("fit_16", "Random initialization"),
             ("fit_16_pretrained_original", "Pretrained"),
             ("fit_16_pretrained_appearance", "Pretrained + appearance")]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    fields = ["total", "coarse", "fine", "train_error_px", "train_max_error_px", "validation_error_px"]
    titles = ["Training total", "Training coarse", "Training fine", "Train mean error (px)",
              "Train maximum error (px)", "Validation mean error (px)"]
    summary = []
    for name, label in names:
        folder = args.runs / name
        with (folder / "history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for axis, key, title in zip(axes.flat, fields, titles):
            selected = [row for row in rows if row[key]]
            axis.plot([int(row["step"]) for row in selected], [float(row[key]) for row in selected], label=label)
            axis.set_title(title)
            axis.set_xlabel("Optimizer step")
            axis.grid(alpha=0.2)
        metrics = json.loads((folder / "metrics.json").read_text())
        summary.append({"run": name, "label": label, "steps": metrics["steps_completed"],
                        "selected_step": metrics["best_step"], "selection": "lowest training mean corner error",
                        "train_error_px": metrics["train"]["mean_corner_error_px"],
                        "validation_error_px": metrics["validation"]["mean_corner_error_px"],
                        "validation_within_10px_percent": metrics["validation"]["within_10px_percent"]})
    axes[0, 0].legend(frameon=False)
    figure.savefig(args.runs / "repair_curves.png", dpi=160)
    plt.close(figure)
    (args.runs / "controlled_comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
