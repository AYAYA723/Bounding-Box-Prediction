import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from predict_corner.pose import solve_box_pose


def rotation_error_degrees(R, truth):
    cosine = (np.trace(np.asarray(R) @ truth.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def evaluate_pose(predictions, scene, output):
    source = json.loads(predictions.read_text())
    geometry = json.loads((scene / 'keypoints_3d.json').read_text())
    cameras = json.loads((scene / 'scene_camera.json').read_text())
    ground_truth = json.loads((scene / 'scene_gt.json').read_text())
    object_points = np.asarray(geometry['corners'], dtype=np.float64)
    records, gt_checks = [], []
    for row in source['per_image']:
        key = str(int(Path(row['image']).stem))
        K = np.array(cameras[key]['cam_K']).reshape(3, 3)
        truth = ground_truth[key][0]
        R = np.array(truth['cam_R_m2c']).reshape(3, 3)
        t = np.array(truth['cam_t_m2c'])
        gt_xy = np.asarray(row['gt_xy'])
        check = solve_box_pose(object_points, gt_xy, K)
        assert check['success'], row['image']
        gt_checks.append([check['mean_reprojection_residual_px'],
                          rotation_error_degrees(check['R_object_to_camera'], R),
                          np.linalg.norm(np.asarray(check['t_object_to_camera']) - t)])
        pose = solve_box_pose(object_points, row['pred_xy'], K)
        record = dict(row)
        record['pose'] = pose
        if pose['success']:
            projected = np.asarray(pose['reprojected_corners_xy'])
            record['pose_corner_error_px'] = float(np.linalg.norm(projected - gt_xy, axis=1).mean())
            record['rotation_error_deg'] = rotation_error_degrees(pose['R_object_to_camera'], R)
            record['translation_error_mm'] = float(np.linalg.norm(np.asarray(pose['t_object_to_camera']) - t))
        records.append(record)
    gt_checks = np.asarray(gt_checks)
    assert (gt_checks.max(0) < np.array([.01, .01, .01])).all(), gt_checks.max(0)
    solved = [row for row in records if row['pose']['success']]
    raw = lambda row: float(np.linalg.norm(np.asarray(row['pred_xy']) - row['gt_xy'], axis=1).mean())
    summary = {'images': len(records), 'solved_images': len(solved), 'failed_images': len(records) - len(solved),
               'solver': 'OpenCV SOLVEPNP_SQPNP + LM, all eight ordered corners',
               'camera_source': 'synthetic scene_camera.json, zero rendering distortion',
               'translation_units': geometry['units'], 'coordinate_frame': 'object to camera',
               'raw_mean_corner_error_all_images_px': float(np.mean([raw(row) for row in records])),
               'gt_roundtrip_max_reprojection_px_rotation_deg_translation_mm': gt_checks.max(0).tolist(),
               'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
               'scene': str(scene), 'predictions': str(predictions)}
    if solved:
        projected_errors = np.concatenate([
            np.linalg.norm(np.asarray(row['pose']['reprojected_corners_xy']) - row['gt_xy'], axis=1) for row in solved])
        raw_errors = np.concatenate([
            np.linalg.norm(np.asarray(row['pred_xy']) - row['gt_xy'], axis=1) for row in solved])
        summary.update({'raw_mean_corner_error_solved_images_px': float(np.mean([raw(row) for row in solved])),
                        'pnp_mean_corner_error_solved_images_px': float(np.mean([row['pose_corner_error_px'] for row in solved])),
                        'median_rotation_error_deg': float(np.median([row['rotation_error_deg'] for row in solved])),
                        'median_translation_error_mm': float(np.median([row['translation_error_mm'] for row in solved])),
                        'mean_fit_residual_px': float(np.mean([row['pose']['mean_reprojection_residual_px'] for row in solved])),
                        'raw_within_10px_solved_percent': float(np.mean(raw_errors <= 10) * 100),
                        'pnp_within_10px_solved_percent': float(np.mean(projected_errors <= 10) * 100),
                        'pnp_p95_corner_error_px': float(np.percentile(projected_errors, 95)),
                        'pose_improved_images': sum(row['pose_corner_error_px'] < raw(row) for row in solved)})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'predictions.json').write_text(json.dumps(records, indent=2) + '\n')
    sheet = Image.new('RGB', (1920, 6 * 386), 'white')
    for index in range(min(24, len(records))):
        row = records[round(index * (len(records) - 1) / max(1, min(23, len(records) - 1)))]
        with Image.open(scene / 'rgb' / row['image']) as source_image:
            image = source_image.convert('RGB')
        draw = ImageDraw.Draw(image)
        layers = [(row['gt_xy'], '#30ef70'), (row['pred_xy'], '#ff5050')]
        if row['pose']['success']:
            layers.append((row['pose']['reprojected_corners_xy'], '#30dfff'))
        for points, color in layers:
            for a, b in geometry['edges']:
                draw.line((tuple(points[a]), tuple(points[b])), fill=color, width=2)
        left, top = index % 4 * 480, index // 4 * 386
        sheet.paste(image.resize((480, 360)), (left, top))
        label = f"{row['image']} raw {raw(row):.1f}px / "
        label += f"PnP {row['pose_corner_error_px']:.1f}px" if row['pose']['success'] else 'PnP failed'
        ImageDraw.Draw(sheet).text((left + 4, top + 364), label, fill='black')
    sheet.save(output / 'gt_green_raw_red_pnp_cyan.jpg', quality=94)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_pose(args.predictions, args.scene, args.output), indent=2))


if __name__ == '__main__':
    main()
