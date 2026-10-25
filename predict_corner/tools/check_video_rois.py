import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from predict_corner.data_prep.prepare_video_crops import camera_rois, load_annotations


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "results/roi_fix_20261025")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations()
    anchors = [row[0] for row in annotations["keyframes"]]
    midpoints = [(a + b) // 2 for a, b in zip(anchors, anchors[1:])]
    review_ids = set(anchors + midpoints)
    capture = cv2.VideoCapture(str(root / annotations["source_video"]))
    width, height = annotations["video_size_wh"]
    fps = capture.get(cv2.CAP_PROP_FPS)
    out_height = round(height / width * 1280 / 2) * 2
    writer = cv2.VideoWriter(str(args.output / "roi_only.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, out_height))
    assert writer.isOpened()
    frame_id = 0
    reviews = []
    tracks = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        boxes = camera_rois(annotations, frame_id)
        assert np.all(boxes[:, :2] >= 0)
        assert np.all(boxes[:, 2:] <= [width, height])
        assert np.all((boxes[:, 2] - boxes[:, 0]) * 3 == (boxes[:, 3] - boxes[:, 1]) * 4)
        tracks.append({"frame_id": frame_id, "rois_xyxy": boxes.tolist()})
        if frame_id in review_ids:
            for side, (x0, y0, x1, y1) in zip(("L", "R"), boxes):
                crop = cv2.resize(frame[y0:y1, x0:x1], (320, 240))
                reviews.append((frame_id, side, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        display = cv2.resize(frame, (1280, out_height))
        for side, box in zip(("L", "R"), boxes):
            rectangle = np.rint(box * np.array([1280 / width, out_height / height] * 2)).astype(int)
            cv2.rectangle(display, tuple(rectangle[:2]), tuple(rectangle[2:]), (255, 180, 30), 2)
            cv2.putText(display, side, tuple(rectangle[:2] + [5, 22]), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.putText(display, f"frame {frame_id} / {frame_id / fps:.2f}s / manual ROI", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        writer.write(display)
        frame_id += 1
    capture.release()
    writer.release()
    for start in range(0, len(reviews), 24):
        sheet = Image.new("RGB", (1920, 4 * 264), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (fid, side, crop) in enumerate(reviews[start:start + 24]):
            x, y = index % 6 * 320, index // 6 * 264
            sheet.paste(crop, (x, y))
            draw.text((x + 5, y + 243), f"frame {fid} / {side}", fill="black")
        sheet.save(args.output / f"review_{start // 24:02d}.jpg", quality=94)
    (args.output / "roi_tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    report = {"processed_frames": frame_id, "anchor_count": len(anchors), "review_frames": len(review_ids),
              "crop_count": len(reviews), "bounds_and_aspect_checks": "passed",
              "note": "Visual crop review is required; these annotations are not corner ground truth. Source-image truncation remains."}
    (args.output / "roi_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
