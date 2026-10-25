import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from predict_corner.dataset import CornerDataset
from predict_corner.model import CornerPredictor
from predict_corner.transformer_model import TransformerCornerPredictor
from predict_corner.supervision import BoxDreamerLoss


def evaluate(model, dataset, indices, device, batch_size=2):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_coarse, total_fine = 0.0, 0.0
    loss_fn = BoxDreamerLoss()
    records = []
    errors = []
    offset = 0
    with torch.no_grad():
        for x, y, target_corners in loader:
            pred, corners_xy = model(x.to(device))
            loss, coarse, fine = loss_fn(pred, corners_xy, y.to(device), target_corners.to(device))
            N = x.shape[0]
            total_loss += loss.item() * N
            total_coarse += coarse.item() * N
            total_fine += fine.item() * N
            corners_xy = corners_xy.cpu()
            for item in range(N):
                index = indices[offset + item]
                name = dataset.label_paths[index].stem
                truth = target_corners[item]
                distance = (corners_xy[item] - truth).norm(dim=1)
                errors.extend(distance.tolist())
                records.append({"image": name + ".png", "gt_xy": truth.tolist(),
                                "pred_xy": corners_xy[item].tolist(), "errors_px": distance.tolist()})
            offset += N
    values = torch.tensor(errors)
    return {
        "image_count": len(indices), "corner_count": len(errors),
        "total_loss": total_loss / len(indices),
        "heatmap_loss": total_coarse / len(indices),
        "fine_loss": total_fine / len(indices),
        "coordinate_decoder": f"spatial soft-argmax over pre-Sigmoid logits, temperature={model.decoder_temperature:g}",
        "mean_corner_error_px": values.mean().item(),
        "median_corner_error_px": values.median().item(),
        "within_5px_percent": (values <= 5).float().mean().item() * 100,
        "within_10px_percent": (values <= 10).float().mean().item() * 100,
        "per_image": records,
    }


def save_preview(report, scene_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    edges = json.loads((scene_dir.parents[1] / "keypoints_3d.json").read_text())["edges"]
    rows = (len(report["per_image"]) + 3) // 4
    sheet = Image.new("RGB", (1600, rows * 324), "#eeeeee")
    for index, record in enumerate(report["per_image"]):
        with Image.open(scene_dir / "rgb" / record["image"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for key, color in (("gt_xy", "#30ef70"), ("pred_xy", "#ff5050")):
            points = record[key]
            for a, b in edges:
                draw.line((tuple(points[a]), tuple(points[b])), fill=color, width=2)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        image.save(output / record["image"], quality=94)
        left, top = index % 4 * 400, index // 4 * 324
        sheet.paste(image.resize((400, 300)), (left, top))
        error = sum(record["errors_px"]) / 8
        ImageDraw.Draw(sheet).text((left + 5, top + 304), f"{record['image']} / mean {error:.1f} px", fill="black")
    sheet.save(output / "gt_green_pred_red.jpg", quality=94)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["resnet", "vit"], default="resnet")
    parser.add_argument("--weights", type=Path, default=root / "predict_corner/checkpoints/corner_predictor_paper.pt")
    parser.add_argument("--output", type=Path, default=root / "results/paper_evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.25)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dataset = CornerDataset(root / "datasets/action4/train_pbr/000000")
    model_classes = {"resnet": CornerPredictor, "vit": TransformerCornerPredictor}
    model = model_classes[args.architecture](decoder_temperature=args.temperature).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    report = evaluate(model, dataset, list(range(len(dataset))), device)
    save_preview(report, dataset.scene_dir, args.output)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
