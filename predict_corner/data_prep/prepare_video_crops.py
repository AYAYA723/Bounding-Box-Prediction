import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ANNOTATIONS = Path(__file__).with_name("video_roi_annotations.json")


def load_annotations(path=ANNOTATIONS):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def camera_body_boxes(annotations, frame_id):
    keyframes = annotations["keyframes"]
    anchors = np.array([row[0] for row in keyframes])
    scale = np.array(annotations["video_size_wh"]) / annotations["annotation_size_wh"]
    boxes = []
    for side in (1, 2):
        values = np.array([row[side] for row in keyframes])
        box = np.array([np.interp(frame_id, anchors, values[:, axis]) for axis in range(4)])
        boxes.append(box * np.tile(scale, 2))  # Restore original video coordinates.
    return np.array(boxes)


def camera_rois(annotations, frame_id):
    """Interpolate body boxes, add context and preserve the 640:480 input ratio."""
    if annotations.get('roi_mode') == 'adaptive':
        return adaptive_camera_rois(annotations, frame_id)
    width, height = annotations["video_size_wh"]
    minimum = annotations["minimum_crop_width"] * width / annotations["annotation_size_wh"][0]
    rois = []
    for x0, y0, x1, y1 in camera_body_boxes(annotations, frame_id):
        crop_width = max(x1 - x0, (y1 - y0) * 4 / 3) * annotations["padding"]
        crop_width = int(np.ceil(max(crop_width, minimum) / 4)) * 4
        crop_height = crop_width * 3 // 4
        left = int(np.clip(round((x0 + x1 - crop_width) / 2), 0, width - crop_width))
        top = int(np.clip(round((y0 + y1 - crop_height) / 2), 0, height - crop_height))
        rois.append([left, top, left + crop_width, top + crop_height])
    return np.array(rois)


def fit_body_crop(body, image_size, padding):
    width, height = image_size
    x0, y0, x1, y1 = body
    crop_width = int(np.ceil(max(x1 - x0, (y1 - y0) * 4 / 3) * padding / 4)) * 4
    crop_width = min(crop_width, int(min(width, height * 4 / 3) // 4) * 4)
    crop_height = crop_width * 3 // 4  # Keep the 4:3 input ratio.
    left = int(np.clip(round((x0 + x1 - crop_width) / 2), 0, width - crop_width))
    top = int(np.clip(round((y0 + y1 - crop_height) / 2), 0, height - crop_height))
    return np.array([left, top, left + crop_width, top + crop_height])


def overlap_fraction(crop, body):
    intersection = np.maximum(np.minimum(crop[2:], body[2:]) - np.maximum(crop[:2], body[:2]), 0)
    return float(np.prod(intersection) / np.prod(body[2:] - body[:2]))


def adaptive_camera_rois(annotations, frame_id):
    """Fit each body and reduce its margin when the other camera enters the crop."""
    bodies = camera_body_boxes(annotations, frame_id)
    size = annotations['video_size_wh']
    minimum = annotations.get('adaptive_min_padding', 1.12)
    maximum = annotations.get('adaptive_max_padding', 1.30)
    limit = annotations.get('adaptive_other_fraction', .10)
    rois = []
    for index, body in enumerate(bodies):
        other = bodies[1 - index]
        low, high = minimum, maximum
        crop = fit_body_crop(body, size, high)
        if overlap_fraction(crop, other) > limit:
            # Shrink context while retaining the current body.
            for _ in range(12):
                middle = (low + high) / 2
                candidate = fit_body_crop(body, size, middle)
                if overlap_fraction(candidate, other) <= limit:
                    low = middle
                else:
                    high = middle
            crop = fit_body_crop(body, size, low)
        rois.append(crop)
    return np.array(rois)


def main():
    root = Path(__file__).resolve().parents[2] / "datasets/action4/video_test_24"
    annotations = load_annotations()
    frames = json.loads((root / "frames.json").read_text())["frames"]
    (root / "crops").mkdir(exist_ok=True)
    (root / "roi_previews").mkdir(exist_ok=True)
    records = []
    crop_sheet = Image.new("RGB", (1600, 12 * 322), "white")
    full_sheet = Image.new("RGB", (1920, 6 * 389), "white")
    for entry in frames:
        with Image.open(root / entry["file"]) as image:
            image = image.convert("RGB")
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        for side_index, (side, bbox) in enumerate(zip(("L", "R"), camera_rois(annotations, entry["frame_id"]))):
            bbox = bbox.tolist()
            filename = f"{entry['index']:02d}_{side}.jpg"
            crop = image.crop(bbox).resize((640, 480), Image.Resampling.LANCZOS)
            crop.save(root / "crops" / filename, quality=95)
            records.append({"image": entry["file"], "frame_id": entry["frame_id"],
                            "side": side, "bbox_xyxy": bbox, "crop": f"crops/{filename}"})
            draw.rectangle(bbox, outline="#40bbff", width=8)
            draw.text((bbox[0] + 10, bbox[1] + 10), side, fill="white", stroke_width=2, stroke_fill="black")
            index = entry["index"] * 2 + side_index
            left, top = index % 4 * 400, index // 4 * 322
            crop_sheet.paste(crop.resize((400, 300)), (left, top))
            ImageDraw.Draw(crop_sheet).text((left + 5, top + 304), f"{filename} / frame {entry['frame_id']}", fill="black")
        preview.save(root / "roi_previews" / f"{entry['index']:02d}.jpg", quality=94)
        left, top = entry["index"] % 4 * 480, entry["index"] // 4 * 389
        full_sheet.paste(preview.resize((480, 365)), (left, top))
        ImageDraw.Draw(full_sheet).text((left + 5, top + 368), f"{entry['index']:02d} / {entry['time_seconds']:.2f}s", fill="black")
    (root / "manual_rois.json").write_text(json.dumps({
        "source": "camera-body annotations with dense temporal anchors and padded 4:3 crops",
        "annotations": "predict_corner/data_prep/video_roi_annotations.json",
        "anchor_count": len(annotations["keyframes"]), "padding": annotations["padding"],
        "note": annotations["note"], "split": "video test only", "rois": records}, indent=2) + "\n", encoding="utf-8")
    crop_sheet.save(root / "manual_crops_contact_sheet.jpg", quality=94)
    full_sheet.save(root / "roi_contact_sheet.jpg", quality=94)
    print(f"Saved {len(records)} camera crops using {len(annotations['keyframes'])} temporal anchors")


if __name__ == "__main__":
    main()
