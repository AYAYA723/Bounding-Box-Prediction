import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from predict_corner.model import CornerPredictor
from predict_corner.transformer_model import TransformerCornerPredictor


def main():
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["resnet", "vit"], default="resnet")
    parser.add_argument("--input", type=Path,
                        default=project_dir / "datasets/action4/video_test_24/crops")
    parser.add_argument("--output", type=Path,
                        default=project_dir / "results/paper_video_test_24")
    parser.add_argument("--weights", type=Path,
                        default=project_dir / "predict_corner/checkpoints/corner_predictor_paper.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_classes = {"resnet": CornerPredictor, "vit": TransformerCornerPredictor}
    model = model_classes[args.architecture](decoder_temperature=args.temperature).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    edges = json.loads((project_dir / "datasets/action4/keypoints_3d.json").read_text())["edges"]
    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted(args.input.glob("*.jpg")) + sorted(args.input.glob("*.png"))
    results = []
    previews = []

    with torch.no_grad():
        for path in images:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            heatmaps, points = model(transform(rgb).unsqueeze(0).to(device))
            pred, points = heatmaps[0], points[0]
            # Map resized pixel centers back to the original image coordinates.
            x = (points[:, 0] + 0.5) * rgb.width / 640 - 0.5
            y = (points[:, 1] + 0.5) * rgb.height / 480 - 0.5
            corners = torch.stack((x, y), dim=-1).cpu().tolist()
            preview = rgb.copy()
            draw = ImageDraw.Draw(preview)
            for a, b in edges:
                draw.line((tuple(corners[a]), tuple(corners[b])), fill="#ff5050", width=3)
            for i, (px, py) in enumerate(corners):
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#ffe040", outline="black")
                draw.text((px + 5, py + 3), str(i), fill="white", stroke_width=1, stroke_fill="black")
            preview.save(args.output / f"{path.stem}_pred.jpg", quality=94)
            previews.append((path.stem, preview))
            results.append({"image": path.name, "corners_xy": corners,
                            "peak_values": pred.flatten(start_dim=1).max(dim=1).values.cpu().tolist()})
    cols = 4
    rows = (len(previews) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 400, rows * 322), "#eeeeee")
    for index, (name, preview) in enumerate(previews):
        left, top = index % cols * 400, index // cols * 322
        sheet.paste(preview.resize((400, 300)), (left, top))
        ImageDraw.Draw(sheet).text((left + 4, top + 304), name, fill="black")
    sheet.save(args.output / "predictions_contact_sheet.jpg", quality=94)
    (args.output / "predictions.json").write_text(json.dumps({
        "weights": str(args.weights), "device": args.device,
        "coordinate_decoder": f"spatial soft-argmax over logits, temperature={model.decoder_temperature:g}",
        "note": "Qualitative predictions; heatmap peaks are responses, not calibrated confidence. Manual video crops are not corner ground truth.",
        "predictions": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Predicted {len(results)} images. Results: {args.output}")


if __name__ == "__main__":
    main()
