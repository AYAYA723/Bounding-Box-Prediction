import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from predict_corner.model import CornerPredictor
from predict_corner.transformer_model import TransformerCornerPredictor
from predict_corner.dino_model import DinoCornerPredictor
from predict_corner.pose import crop_to_image, image_to_crop, load_camera, solve_box_pose
from predict_corner.data_prep.prepare_video_crops import ANNOTATIONS, camera_rois, load_annotations


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["resnet", "vit", "dino"], default="resnet")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=root / "results/roi_fixed_full_video")
    parser.add_argument("--rois", type=Path, default=ANNOTATIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--camera", type=Path, help="Pinhole calibration for the original video resolution")
    parser.add_argument("--sample-only", action="store_true", help="Evaluate the 24 fixed sample frames at 2 fps")
    parser.add_argument("--allow-estimated-camera", action="store_true", help="Use video-estimated intrinsics and label the output")
    parser.add_argument("--roi-mode", choices=['padded', 'adaptive'], default='padded')
    args = parser.parse_args()
    args.weights = args.weights or root / 'predict_corner/checkpoints' / (
        'corner_predictor_dino.pt' if args.architecture == 'dino' else 'corner_predictor_paper.pt')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model_classes = {"resnet": CornerPredictor, "vit": TransformerCornerPredictor}
    if args.architecture == 'dino':
        model = DinoCornerPredictor(decoder_temperature=args.temperature, pretrained=False).to(device)
    else:
        model = model_classes[args.architecture](decoder_temperature=args.temperature).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    output = args.output
    (output / "frames").mkdir(parents=True, exist_ok=True)
    video_dir = root / "datasets/action4/video_test_24"
    sampled = json.loads((video_dir / "frames.json").read_text())["frames"]
    annotations = load_annotations(args.rois)
    annotations['roi_mode'] = args.roi_mode
    sample_lookup = {entry["frame_id"]: entry["index"] for entry in sampled}
    geometry = json.loads((root / "datasets/action4/keypoints_3d.json").read_text())
    edges = geometry["edges"]
    video_path = root / "head_left_rgb_raw.mp4"
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = [entry['frame_id'] for entry in sampled] if args.sample_only else range(count)
    output_fps = 2 if args.sample_only else fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera = None
    if args.camera:
        K, distortion, camera = load_camera(args.camera, (width, height))
        if camera['calibration_kind'] != 'measured' and not (
                camera['calibration_kind'] == 'estimated' and args.allow_estimated_camera):
            parser.error('Use measured intrinsics, or explicitly select --allow-estimated-camera for estimated intrinsics.')
    out_w, out_h = 1280, round(height / width * 1280 / 2) * 2
    full_path, crops_path = output / "full_video_overlay.mp4", output / "camera_crops.mp4"
    full_writer = cv2.VideoWriter(str(full_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (out_w, out_h))
    crop_writer = cv2.VideoWriter(str(crops_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (1280, 480))
    assert full_writer.isOpened() and crop_writer.isOpened()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    previews = []
    start = time.perf_counter()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream, torch.no_grad():
        for frame_id in frame_ids:
            if args.sample_only:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = capture.read()
            assert success, f"Failed to decode frame {frame_id}"
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_boxes = camera_rois(annotations, frame_id)
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
                source_points = crop_to_image(crop_points, box)  # Restore coordinates before PnP.
                target_record = {"side": side, "manual_roi_xyxy": box.tolist(),
                                 "predicted_corners_original_xy": source_points.round(3).tolist()}
                display_points, crop_display_points = source_points, crop_points
                if camera is not None:
                    pose = solve_box_pose(geometry['corners'], source_points, K, distortion)
                    pose['translation_units'] = geometry['units']
                    target_record['pose'] = pose
                    if pose['success']:
                        display_points = np.asarray(pose['reprojected_corners_xy'])
                        crop_display_points = image_to_crop(display_points, box)
                full_points = np.rint(display_points * np.array([out_w / width, out_h / height])).astype(int)
                display_box = np.rint(box * np.array([out_w / width, out_h / height] * 2)).astype(int)
                cv2.rectangle(display, tuple(display_box[:2]), tuple(display_box[2:]), (255, 180, 30), 1)
                images_and_points = [] if camera is not None and not pose['success'] else [
                    (display, full_points), (crop_displays[side_index], np.rint(crop_display_points).astype(int))]
                for image, corners in images_and_points:
                    for a, b in edges:
                        cv2.line(image, tuple(corners[a]), tuple(corners[b]), (60, 60, 255), 2)
                    for point in corners:
                        cv2.circle(image, tuple(point), 3, (30, 235, 255), -1)
                cv2.putText(crop_displays[side_index], side + " / manual ROI", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                record["targets"].append(target_record)
            mode = 'PnP reprojection' if camera is not None else 'predicted corners'
            if camera is not None and camera['calibration_kind'] == 'estimated':
                mode += ' / estimated K'
            roi_label = 'adaptive ROI from manual boxes' if args.roi_mode == 'adaptive' else 'manual ROI interpolation'
            cv2.putText(display, f"{frame_id / fps:.2f}s / {roi_label} / {mode}", (12, 30),
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
        assert encoded_counts[path.name] == len(frame_ids)
        check.release()
    sheet = Image.new("RGB", (2400, 6 * 480), "#eeeeee")
    for index, (seconds, preview) in enumerate(previews):
        left, top = index % 4 * 600, index // 4 * 480
        sheet.paste(preview.resize((600, 455)), (left, top))
        ImageDraw.Draw(sheet).text((left + 5, top + 459), f"{index:02d} / {seconds:.2f}s", fill="black")
    sheet.save(output / "contact_sheet.jpg", quality=94)
    report = {"architecture": args.architecture, "weights": str(args.weights), "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "source_video": video_path.name, "input_frames": count, "processed_frames": len(frame_ids),
              "sample_only": args.sample_only, "output_fps": output_fps,
              "fps": fps, "duration_seconds": count / fps, "target_inferences": len(frame_ids) * 2,
              "output_video_frame_counts": encoded_counts, "elapsed_seconds": time.perf_counter() - start,
              "roi_method": "dense manual camera-body anchors; " + args.roi_mode + " 4:3 crops; no automatic detector",
              "roi_mode": args.roi_mode,
              "roi_parameters": ({'min_padding': annotations.get('adaptive_min_padding', 1.12),
                                  'max_padding': annotations.get('adaptive_max_padding', 1.30),
                                  'other_body_fraction_limit': annotations.get('adaptive_other_fraction', .10)}
                                 if args.roi_mode == 'adaptive' else
                                 {'padding': annotations['padding'], 'minimum_crop_width': annotations['minimum_crop_width']}),
              "roi_anchor_count": len(annotations["keyframes"]),
              "roi_annotations_sha256": hashlib.sha256(args.rois.read_bytes()).hexdigest(),
              "coordinate_decoder": f"spatial soft-argmax over logits, temperature={model.decoder_temperature:g}",
              "camera": camera, "pose_solver": 'OpenCV SOLVEPNP_SQPNP + LM' if camera else None,
              "camera_sha256": hashlib.sha256(args.camera.read_bytes()).hexdigest() if args.camera else None,
              "video_ground_truth_available": False}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
