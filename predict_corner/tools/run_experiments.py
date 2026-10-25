import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=root / "results" / datetime.now().strftime("paper_supervision_%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    cache = root / ".cache/corner_prediction"
    os.environ["CUDA_CACHE_PATH"] = str(cache / "cuda")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    report = {
        "started_at": datetime.now().astimezone().isoformat(), "status": "running",
        "loss": "coarse + 2*fine", "seed": 42, "batch_size": 2, "epochs": args.epochs,
        "resnet_learning_rates": [1e-4, 3e-4, 1e-3],
        "transformer_learning_rate": "selected ResNet learning rate",
        "split": "16 train / 4 validation; video frames are test only",
        "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((root / "predict_corner").rglob("*.py"))},
        "stages": [],
    }
    save_json(output / "pipeline.json", report)

    def run(name, module, *arguments):
        command = [sys.executable, "-B", "-m", module, *map(str, arguments)]
        stage = {"name": name, "command": command, "status": "running",
                 "started_at": datetime.now().astimezone().isoformat(),
                 "log": f"logs/{name}.log"}
        report["stages"].append(stage)
        save_json(output / "pipeline.json", report)
        print(f"START {name}: {stage['started_at']}", flush=True)
        started = time.perf_counter()
        with (output / stage["log"]).open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        stage.update({"finished_at": datetime.now().astimezone().isoformat(),
                      "elapsed_seconds": time.perf_counter() - started,
                      "returncode": process.returncode,
                      "status": "complete" if process.returncode == 0 else "failed"})
        if process.returncode:
            report["status"] = "failed"
        save_json(output / "pipeline.json", report)
        process.check_returncode()
        print(f"DONE {name}: {stage['elapsed_seconds']:.1f}s", flush=True)

    run("unit_tests", "unittest", "discover", "-s", "predict_corner/tests", "-v")
    run("resnet_tuning", "predict_corner.tools.tune", "--architecture", "resnet",
        "--epochs", args.epochs, "--device", args.device, "--output", output / "resnet")
    run("resnet_curves", "predict_corner.tools.plot_tuning", "--runs", output / "resnet")
    candidates = json.loads((output / "resnet/comparison.json").read_text(encoding="utf-8"))
    best_resnet = min(candidates, key=lambda row: row["validation_error_px"])
    learning_rate = best_resnet["learning_rate"]
    checkpoints = root / "predict_corner/checkpoints"
    checkpoints.mkdir(exist_ok=True)
    resnet_weights = checkpoints / "corner_predictor_paper.pt"
    shutil.copy2(root / best_resnet["best_weights"], resnet_weights)
    report["selected_resnet"] = best_resnet
    save_json(output / "pipeline.json", report)
    run("resnet_24_frame_crops", "predict_corner.tools.predict", "--device", args.device,
        "--weights", resnet_weights, "--output", output / "resnet_video_test_24")
    run("resnet_full_video", "predict_corner.tools.test_video", "--device", args.device,
        "--weights", resnet_weights, "--output", output / "resnet_full_video")

    # Complete the ResNet full-video test before training the Transformer.
    run("vit_training", "predict_corner.tools.tune", "--architecture", "vit",
        "--epochs", args.epochs, "--learning-rates", learning_rate,
        "--device", args.device, "--output", output / "vit")
    run("vit_curves", "predict_corner.tools.plot_tuning", "--runs", output / "vit")
    best_vit = json.loads((output / "vit/comparison.json").read_text(encoding="utf-8"))[0]
    vit_weights = checkpoints / "corner_predictor_transformer_paper.pt"
    shutil.copy2(root / best_vit["best_weights"], vit_weights)
    run("vit_24_frame_crops", "predict_corner.tools.predict", "--architecture", "vit",
        "--device", args.device, "--weights", vit_weights, "--output", output / "vit_video_test_24")
    run("vit_full_video", "predict_corner.tools.test_video", "--architecture", "vit",
        "--device", args.device, "--weights", vit_weights, "--output", output / "vit_full_video")

    summary = {"loss": report["loss"], "split": report["split"],
               "resnet18": best_resnet, "small_vit": best_vit,
               "comparison": "Same selected learning rate, batch size, seed, epoch budget and decoder",
               "checkpoints": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (resnet_weights, vit_weights)}}
    save_json(output / "architecture_comparison.json", summary)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, key, title in zip(axes, ("train_error_px", "validation_error_px"), ("Train", "Validation")):
        axis.bar(["ResNet-18", "Small ViT"], [best_resnet[key], best_vit[key]], color=["#377eb8", "#e68732"])
        axis.set_title(title)
        axis.set_ylabel("Mean corner error (px)")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "architecture_comparison.png", dpi=180)
    plt.close(figure)
    report.update({"status": "complete", "finished_at": datetime.now().astimezone().isoformat(),
                   "selected_vit": best_vit, "summary": "architecture_comparison.json"})
    save_json(output / "pipeline.json", report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"All experiments complete: {output}", flush=True)


if __name__ == "__main__":
    main()
