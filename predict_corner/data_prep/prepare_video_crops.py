import json
from pathlib import Path

from PIL import Image


# Approximate left/right camera centers read manually from the sampled frames.
CENTERS = [
    [(0.435, 0.47), (0.74, 0.48)], [(0.445, 0.71), (0.66, 0.68)],
    [(0.43, 0.70), (0.64, 0.82)], [(0.44, 0.75), (0.62, 0.76)],
    [(0.49, 0.74), (0.65, 0.76)], [(0.45, 0.72), (0.61, 0.66)],
    [(0.48, 0.735), (0.65, 0.73)], [(0.47, 0.73), (0.63, 0.715)],
    [(0.43, 0.72), (0.57, 0.77)], [(0.48, 0.755), (0.63, 0.765)],
    [(0.48, 0.73), (0.55, 0.77)], [(0.51, 0.73), (0.64, 0.71)],
    [(0.46, 0.745), (0.64, 0.67)], [(0.46, 0.67), (0.62, 0.715)],
    [(0.46, 0.64), (0.61, 0.73)], [(0.48, 0.63), (0.62, 0.68)],
    [(0.46, 0.64), (0.56, 0.72)], [(0.49, 0.65), (0.63, 0.68)],
    [(0.44, 0.57), (0.65, 0.63)], [(0.32, 0.65), (0.73, 0.71)],
    [(0.35, 0.665), (0.61, 0.78)], [(0.34, 0.815), (0.80, 0.82)],
    [(0.43, 0.775), (0.59, 0.69)], [(0.42, 0.88), (0.67, 0.92)]
]


def main():
    root = Path(__file__).resolve().parents[2] / "datasets/action4/video_test_24"
    frames = json.loads((root / "frames.json").read_text())["frames"]
    (root / "crops").mkdir(exist_ok=True)
    records = []
    for entry, centers in zip(frames, CENTERS):
        with Image.open(root / entry["file"]) as image:
            width, height = image.size
            crop_width = round(width * (0.40 if entry["index"] == 0 else 0.22))
            crop_height = round(crop_width * 3 / 4)
            for side, (cx, cy) in zip(("L", "R"), centers):
                left = max(0, min(width - crop_width, round(cx * width - crop_width / 2)))
                top = max(0, min(height - crop_height, round(cy * height - crop_height / 2)))
                bbox = [left, top, left + crop_width, top + crop_height]
                filename = f"{entry['index']:02d}_{side}.jpg"
                image.crop(bbox).resize((640, 480), Image.Resampling.LANCZOS).save(
                    root / "crops" / filename, quality=95)
                records.append({"image": entry["file"], "side": side,
                                "bbox_xyxy": bbox, "crop": f"crops/{filename}"})
    (root / "manual_rois.json").write_text(json.dumps({
        "source": "manual approximate crop selection for qualitative testing",
        "note": "Not detector outputs or corner ground truth; inspect partial visibility and crop boundaries.",
        "split": "video test only", "rois": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} manual target crops")


if __name__ == "__main__":
    main()
