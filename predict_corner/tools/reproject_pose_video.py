import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from predict_corner.pose import load_camera, project_box


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses', type=Path, required=True)
    parser.add_argument('--camera', type=Path, default=root / 'datasets/action4/video_camera_estimated.json')
    parser.add_argument('--geometry', type=Path, default=root / 'datasets/action4/keypoints_3d.json')
    parser.add_argument('--video', type=Path, default=root / 'head_left_rgb_raw.mp4')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'frames').mkdir()
    with np.load(args.poses, allow_pickle=False) as saved:
        transforms, valid = saved['T_object_to_camera'], saved['valid']
        frame_ids, times, sides = saved['frame_ids'], saved['times_seconds'], saved['sides']
    geometry = json.loads(args.geometry.read_text())
    capture = cv2.VideoCapture(str(args.video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    full_sequence = len(frame_ids) == source_frames
    fps = capture.get(cv2.CAP_PROP_FPS) if full_sequence else 2
    K, distortion, camera = load_camera(args.camera, (width, height))
    out_w, out_h = 1280, round(height / width * 1280 / 2) * 2
    scale = np.array([out_w / width, out_h / height])
    output_video = args.output / 'box_from_poses.mp4'
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    assert writer.isOpened()
    sample_rows = json.loads((root / 'datasets/action4/video_test_24/frames.json').read_text())['frames']
    sample_lookup = {row['frame_id']: row['index'] for row in sample_rows}
    previews = []
    fields = ['frame_id', 'time_seconds', 'side', 'valid', 'xmin', 'ymin', 'xmax', 'ymax',
              *[f'{axis}{corner}' for corner in range(8) for axis in ['x', 'y']]]
    started = time.perf_counter()
    count = 0
    with (args.output / 'boxes.jsonl').open('w', encoding='utf-8') as stream, \
         (args.output / 'boxes.csv').open('w', newline='', encoding='utf-8') as csv_stream:
        csv_writer = csv.DictWriter(csv_stream, fieldnames=fields)
        csv_writer.writeheader()
        for index, frame_id in enumerate(frame_ids):
            if not full_sequence:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            success, frame = capture.read()
            assert success
            display = cv2.resize(frame, (out_w, out_h))
            record = {'frame_id': int(frame_id), 'time_seconds': float(times[index]), 'targets': []}
            for target, side in enumerate(sides):
                row = {'frame_id': int(frame_id), 'time_seconds': float(times[index]),
                       'side': str(side), 'valid': bool(valid[index, target])}
                if row['valid']:
                    T = transforms[index, target]
                    points = project_box(geometry['corners'], T[:3, :3], T[:3, 3], K, distortion)
                    assert np.isfinite(points).all()
                    box = np.concatenate([points.min(0), points.max(0)])
                    row.update(dict(zip(['xmin', 'ymin', 'xmax', 'ymax'], box)))
                    row.update({f'{axis}{corner}': points[corner, channel]
                                for corner in range(8) for channel, axis in enumerate(['x', 'y'])})
                    record['targets'].append({'side': str(side), 'valid': True,
                                              'box_corners_original_xy': points.tolist(),
                                              'bbox2d_xyxy': box.tolist()})
                    pixels = np.rint(points * scale).astype(int)
                    for a, b in geometry['edges']:
                        cv2.line(display, tuple(pixels[a]), tuple(pixels[b]), (60, 60, 255), 2)
                    for point in pixels:
                        cv2.circle(display, tuple(point), 3, (30, 235, 255), -1)
                    label_position = tuple(np.rint(points.mean(0) * scale).astype(int))
                    cv2.putText(display, str(side), label_position, cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                    count += 1
                else:
                    record['targets'].append({'side': str(side), 'valid': False})
                csv_writer.writerow(row)
            label = 'Box from saved poses'
            if camera['calibration_kind'] == 'estimated':
                label += ' / estimated K'
            cv2.putText(display, f'{times[index]:.2f}s / {label}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            writer.write(display)
            stream.write(json.dumps(record) + '\n')
            if int(frame_id) in sample_lookup:
                preview = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                preview.save(args.output / 'frames' / f'{sample_lookup[int(frame_id)]:02d}.jpg', quality=94)
                previews.append((float(times[index]), preview))
            if (index + 1) % 500 == 0:
                print(f'Reprojection {index + 1}/{len(frame_ids)} frames', flush=True)
    capture.release()
    writer.release()
    sheet = Image.new('RGB', (2400, 6 * 480), '#eeeeee')
    for index, (seconds, preview) in enumerate(previews):
        left, top = index % 4 * 600, index // 4 * 480
        sheet.paste(preview.resize((600, 455)), (left, top))
        ImageDraw.Draw(sheet).text((left + 5, top + 459), f'{index:02d} / {seconds:.2f}s', fill='black')
    sheet.save(args.output / 'contact_sheet.jpg', quality=94)
    report = {'frames': len(frame_ids), 'boxes': count, 'neural_network_executed': False,
              'pose_source': str(args.poses), 'pose_sha256': hashlib.sha256(args.poses.read_bytes()).hexdigest(),
              'camera_source': str(args.camera), 'camera_kind': camera['calibration_kind'],
              'camera_sha256': hashlib.sha256(args.camera.read_bytes()).hexdigest(),
              'geometry_sha256': hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
              'output_fps': fps, 'elapsed_seconds': time.perf_counter() - started}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
