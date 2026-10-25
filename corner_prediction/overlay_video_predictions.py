import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1]
    video_dir = root / "datasets/action4/video_test_24"
    results_dir = root / "results/paper_video_test_24"
    output = results_dir / "full_frames"
    output.mkdir(parents=True, exist_ok=True)
    frames = json.loads((video_dir / "frames.json").read_text())["frames"]
    rois = json.loads((video_dir / "manual_rois.json").read_text())["rois"]
    records = json.loads((results_dir / "predictions.json").read_text())["predictions"]
    predictions = {record["image"]: record for record in records}
    edges = json.loads((root / "datasets/action4/keypoints_3d.json").read_text())["edges"]
    tile_w, tile_h = 600, 455
    sheet = Image.new("RGB", (2400, 6 * (tile_h + 24)), "#eeeeee")
    for entry in frames:
        with Image.open(video_dir / entry["file"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for roi in [row for row in rois if row["image"] == entry["file"]]:
            x0, y0, x1, y1 = roi["bbox_xyxy"]
            record = predictions[Path(roi["crop"]).name]
            points = [((x + 0.5) * (x1 - x0) / 640 - 0.5 + x0,
                       (y + 0.5) * (y1 - y0) / 480 - 0.5 + y0) for x, y in record["corners_xy"]]
            draw.rectangle((x0, y0, x1, y1), outline="#40bbff", width=3)
            for a, b in edges:
                draw.line((points[a], points[b]), fill="#ff5050", width=5)
            for x, y in points:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#ffe040", outline="black")
        name = f"{entry['index']:02d}_prediction.jpg"
        image.save(output / name, quality=94)
        left, top = entry["index"] % 4 * tile_w, entry["index"] // 4 * (tile_h + 24)
        sheet.paste(image.resize((tile_w, tile_h)), (left, top))
        ImageDraw.Draw(sheet).text((left + 5, top + tile_h + 5),
            f"{entry['index']:02d} / {entry['time_seconds']:.2f}s / blue: manual crop, red: prediction", fill="black")
    sheet.save(results_dir / "full_frames_contact_sheet.jpg", quality=94)
    print(f"Saved {len(frames)} full-frame overlays to {output}")


if __name__ == "__main__":
    main()
