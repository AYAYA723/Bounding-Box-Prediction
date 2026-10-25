import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from model import CornerPredictor
from transformer_model import TransformerCornerPredictor


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["resnet", "vit"], default="resnet")
    parser.add_argument("--weights", type=Path, default=root / "corner_prediction/corner_predictor_paper.pt")
    parser.add_argument("--output", type=Path, default=root / "results/paper_full_video")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model_classes = {"resnet": CornerPredictor, "vit": TransformerCornerPredictor}
    model = model_classes[args.architecture]().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    output = args.output
    (output / "frames").mkdir(parents=True, exist_ok=True)
    video_dir = root / "datasets/action4/video_test_24"
    sampled = json.loads((video_dir / "frames.json").read_text())["frames"]
    rois = json.loads((video_dir / "manual_rois.json").read_text())["rois"]
    anchors = np.array([entry["frame_id"] for entry in sampled])
    sample_lookup = {entry["frame_id"]: entry["index"] for entry in sampled}
    boxes = {side: np.array([row["bbox_xyxy"] for row in rois if row["side"] == side]) for side in ("L", "R")}
    edges = json.loads((root / "datasets/action4/keypoints_3d.json").read_text())["edges"]
    video_path = root / "head_left_rgb_raw.mp4"
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = 1280, round(height / width * 1280 / 2) * 2
    full_path, crops_path = output / "full_video_overlay.mp4", output / "camera_crops.mp4"
    full_writer = cv2.VideoWriter(str(full_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    crop_writer = cv2.VideoWriter(str(crops_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    assert full_writer.isOpened() and crop_writer.isOpened()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    previews = []
    start = time.perf_counter()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream, torch.no_grad():
        for frame_id in range(count):
            success, frame = capture.read()
            assert success, f"Failed to decode frame {frame_id}"
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_boxes = [np.rint([np.interp(frame_id, anchors, boxes[side][:, axis])
                                     for axis in range(4)]).astype(int) for side in ("L", "R")]
            crops = [np.asarray(rgb.crop(tuple(box)).resize((640, 480), Image.Resampling.LANCZOS))
                     for box in current_boxes]
            x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(device).float() / 255
            pred, corners_xy = model((x - mean) / std)
            points = corners_xy.cpu().numpy()
            display = cv2.resize(frame, (out_w, out_h))
            crop_displays = [cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) for crop in crops]
            record = {"frame_id": frame_id, "time_seconds": frame_id / fps, "targets": []}
            for side_index, (side, box, crop_points) in enumerate(zip(("L", "R"), current_boxes, points)):
                x0, y0, x1, y1 = box
                source_points = (crop_points + 0.5) * np.array([(x1 - x0) / 640, (y1 - y0) / 480]) - 0.5 + box[:2]
                full_points = np.rint(source_points * np.array([out_w / width, out_h / height])).astype(int)
                display_box = np.rint(box * np.array([out_w / width, out_h / height] * 2)).astype(int)
                cv2.rectangle(display, tuple(display_box[:2]), tuple(display_box[2:]), (255, 180, 30), 1)
                for image, corners in ((display, full_points), (crop_displays[side_index], np.rint(crop_points).astype(int))):
                    for a, b in edges:
                        cv2.line(image, tuple(corners[a]), tuple(corners[b]), (60, 60, 255), 2)
                    for point in corners:
                        cv2.circle(image, tuple(point), 3, (30, 235, 255), -1)
                cv2.putText(crop_displays[side_index], side + " / manual ROI", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                record["targets"].append({"side": side, "manual_roi_xyxy": box.tolist(),
                                          "predicted_corners_original_xy": source_points.round(3).tolist()})
            cv2.putText(display, f"{frame_id / fps:.2f}s / manual ROI interpolation / predicted corners", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            full_writer.write(display)
            crop_writer.write(np.concatenate(crop_displays, axis=1))
            stream.write(json.dumps(record) + "\n")
            if frame_id in sample_lookup:
                preview = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                preview.save(output / "frames" / f"{sample_lookup[frame_id]:02d}.jpg", quality=94)
                previews.append((frame_id / fps, preview))
            if (frame_id + 1) % 250 == 0:
                stream.flush()
                print(f"Video: {frame_id + 1}/{count} frames, elapsed {time.perf_counter() - start:.1f}s", flush=True)
    capture.release()
    full_writer.release()
    crop_writer.release()
    encoded_counts = {}
    for path in (full_path, crops_path):
        check = cv2.VideoCapture(str(path))
        encoded_counts[path.name] = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        assert encoded_counts[path.name] == count
        check.release()
    sheet = Image.new("RGB", (2400, 6 * 480), "#eeeeee")
    for index, (seconds, preview) in enumerate(previews):
        left, top = index % 4 * 600, index // 4 * 480
        sheet.paste(preview.resize((600, 455)), (left, top))
        ImageDraw.Draw(sheet).text((left + 5, top + 459), f"{index:02d} / {seconds:.2f}s", fill="black")
    sheet.save(output / "contact_sheet.jpg", quality=94)
    report = {"architecture": args.architecture, "weights": str(args.weights), "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "source_video": video_path.name, "input_frames": count, "processed_frames": count,
              "fps": fps, "duration_seconds": count / fps, "target_inferences": count * 2,
              "output_video_frame_counts": encoded_counts, "elapsed_seconds": time.perf_counter() - start,
              "roi_method": "linear interpolation of 24 manually selected left/right crop rectangles; no automatic detector",
              "coordinate_decoder": "spatial soft-argmax over logits, temperature=1",
              "video_ground_truth_available": False}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
