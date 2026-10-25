import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from predict_corner.pose import image_to_crop, solve_box_pose


def percentiles(values):
    values = np.asarray(values)
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'maximum': float(values.max())}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=root / 'results/video_pnp_estimated_k_full_20261025')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in (args.run / 'predictions.jsonl').read_text().splitlines()]
    camera = json.loads((args.run / 'camera.json').read_text())
    geometry = json.loads((args.run / 'keypoints_3d.json').read_text())
    K = np.array(camera['cam_K'])
    residual_crop, residual_relative, rotation_steps, translation_steps, examples = [], [], [], [], []
    initial_rotation_steps = []
    previous = {}
    jumps = []
    for record in records:
        for target in record['targets']:
            pose = target['pose']
            if not pose['success']:
                previous.pop(target['side'], None)
                continue
            raw = image_to_crop(target['predicted_corners_original_xy'], target['manual_roi_xyxy'])
            fitted = image_to_crop(pose['reprojected_corners_xy'], target['manual_roi_xyxy'])
            residual_crop.append(float(np.linalg.norm(raw - fitted, axis=1).mean()))
            extent = np.linalg.norm(np.ptp(fitted, axis=0))
            residual_relative.append(float(np.linalg.norm(raw - fitted, axis=1).mean() / extent))
            side = target['side']
            if side in previous:
                prev = previous[side]
                R, old_R = np.asarray(pose['R_object_to_camera']), np.asarray(prev['pose']['R_object_to_camera'])
                angle = float(np.degrees(np.arccos(np.clip((np.trace(R @ old_R.T) - 1) / 2, -1, 1))))
                translation = float(np.linalg.norm(np.asarray(pose['t_object_to_camera']) - prev['pose']['t_object_to_camera']))
                rotation_steps.append(angle)
                translation_steps.append(translation)
                if record['time_seconds'] < 3.5:
                    initial_rotation_steps.append(angle)
                jumps.append({'frame_id': record['frame_id'], 'side': side, 'rotation_step_deg': angle,
                              'translation_step_mm': translation})
            previous[side] = target
    sampled = json.loads((root / 'datasets/action4/video_test_24/frames.json').read_text())['frames']
    sensitivities = {}
    for focal in [1200., 1356., 1448.9096202758942, 1599., 1792., 1900.]:
        trial_K = K.copy()
        trial_K[0, 0] = trial_K[1, 1] = focal
        changes = []
        fits = []
        rotations = []
        for frame in sampled:
            record = records[frame['frame_id']]
            for target in record['targets']:
                original = target['pose']
                if not original['success']:
                    continue
                pose = solve_box_pose(geometry['corners'], target['predicted_corners_original_xy'], trial_K)
                if not pose['success']:
                    continue
                baseline = image_to_crop(original['reprojected_corners_xy'], target['manual_roi_xyxy'])
                trial = image_to_crop(pose['reprojected_corners_xy'], target['manual_roi_xyxy'])
                raw = image_to_crop(target['predicted_corners_original_xy'], target['manual_roi_xyxy'])
                changes.append(float(np.linalg.norm(trial - baseline, axis=1).mean()))
                fits.append(float(np.linalg.norm(trial - raw, axis=1).mean()))
                R = np.asarray(pose['R_object_to_camera']) @ np.asarray(original['R_object_to_camera']).T
                rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
        sensitivities[str(focal)] = {'solved': len(changes), 'box_change_crop_px': percentiles(changes),
                                     'fit_residual_crop_px': percentiles(fits), 'pose_rotation_change_deg': percentiles(rotations)}
    report = {'crop_coordinate_size_wh': [640, 480], 'fit_residual_crop_px': percentiles(residual_crop),
              'fit_residual_divided_by_box_diagonal': percentiles(residual_relative),
              'consecutive_rotation_step_deg': percentiles(rotation_steps),
              'consecutive_translation_step_mm': percentiles(translation_steps),
              'first_3p5_seconds_rotation_step_deg': percentiles(initial_rotation_steps),
              'rotation_steps_over_45deg': int(np.sum(np.asarray(rotation_steps) > 45)),
              'rotation_steps_count': len(rotation_steps),
              'focal_sensitivity_fixed_network_predictions': sensitivities,
              'largest_rotation_steps': sorted(jumps, key=lambda row: row['rotation_step_deg'], reverse=True)[:20]}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    capture = cv2.VideoCapture(str(root / 'head_left_rgb_raw.mp4'))
    for sample_index in [0, 4, 9, 12]:
        frame_id = sampled[sample_index]['frame_id']
        record = records[frame_id]
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = capture.read()
        assert success
        source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        panel = Image.new('RGB', (1280, 2 * 516), 'white')
        for row, target in enumerate(record['targets']):
            roi = target['manual_roi_xyxy']
            crop = source.crop(roi).resize((640, 480), Image.Resampling.LANCZOS)
            for column, (name, points) in enumerate([
                    ('Raw network corners', target['predicted_corners_original_xy']),
                    ('PnP reprojection', target['pose']['reprojected_corners_xy'])]):
                image = crop.copy()
                draw = ImageDraw.Draw(image)
                points = image_to_crop(points, roi)
                for a, b in geometry['edges']:
                    draw.line((tuple(points[a]), tuple(points[b])), fill='#ff3030', width=3)
                for index, point in enumerate(points):
                    x, y = point
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='#ffe030')
                    draw.text((x + 4, y + 4), str(index), fill='white', stroke_width=1, stroke_fill='black')
                left, top = column * 640, row * 516
                ImageDraw.Draw(panel).text((left + 5, top + 8), f'{record["time_seconds"]:.2f}s / {target["side"]} / {name}', fill='black')
                panel.paste(image, (left, top + 36))
        panel.save(args.output / f'raw_vs_pnp_{sample_index:02d}.jpg', quality=94)
    capture.release()
    print(json.dumps({key: value for key, value in report.items() if key != 'largest_rotation_steps'}, indent=2))


if __name__ == '__main__':
    main()
