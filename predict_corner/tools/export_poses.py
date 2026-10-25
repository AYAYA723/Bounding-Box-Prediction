import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def export_poses(run, geometry):
    records = [json.loads(line) for line in (run / 'predictions.jsonl').read_text().splitlines()]
    config = json.loads((run / 'run.json').read_text())
    K = np.asarray(config['camera']['cam_K'])
    distortion = np.asarray(config['camera']['dist_coeffs'])
    points = np.asarray(json.loads(geometry.read_text())['corners'])
    count = len(records)
    transforms = np.full((count, 2, 4, 4), np.nan)
    valid = np.zeros((count, 2), dtype=bool)
    residuals = np.full((count, 2), np.nan)
    maximum_projection_difference = 0.
    statuses = {}
    fields = ['frame_id', 'time_seconds', 'side', 'success', 'status', 'tx_mm', 'ty_mm', 'tz_mm',
              *[f'r{i}{j}' for i in range(3) for j in range(3)],
              'rvec_x', 'rvec_y', 'rvec_z', 'mean_fit_residual_px', 'min_corner_depth_mm']
    with (run / 'poses.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            if not config['sample_only']:
                assert record['frame_id'] == index
            assert len(record['targets']) == 2
            for target_index, target in enumerate(record['targets']):
                assert target['side'] == ['L', 'R'][target_index]
                pose = target['pose']
                status = pose['status']
                statuses[status] = statuses.get(status, 0) + 1
                row = {'frame_id': record['frame_id'], 'time_seconds': record['time_seconds'],
                       'side': target['side'], 'success': pose['success'], 'status': status}
                if pose['success']:
                    T = np.asarray(pose['T_object_to_camera'])
                    assert T.shape == (4, 4) and np.isfinite(T).all()
                    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-9)
                    np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-7)
                    assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-7
                    depths = (points @ T[:3, :3].T + T[:3, 3])[:, 2]
                    assert (depths > 0).all()
                    projected = cv2.projectPoints(points, np.asarray(pose['rvec']), T[:3, 3], K, distortion)[0].reshape(8, 2)
                    difference = np.abs(projected - pose['reprojected_corners_xy']).max()
                    maximum_projection_difference = max(maximum_projection_difference, float(difference))
                    transforms[index, target_index] = T
                    valid[index, target_index] = True
                    residuals[index, target_index] = pose['mean_reprojection_residual_px']
                    row.update(dict(zip(['tx_mm', 'ty_mm', 'tz_mm'], T[:3, 3])))
                    row.update({f'r{i}{j}': T[i, j] for i in range(3) for j in range(3)})
                    row.update(dict(zip(['rvec_x', 'rvec_y', 'rvec_z'], pose['rvec'])))
                    row.update({'mean_fit_residual_px': pose['mean_reprojection_residual_px'],
                                'min_corner_depth_mm': pose['min_corner_depth']})
                writer.writerow(row)
    assert maximum_projection_difference < 1e-6
    frame_ids = np.array([record['frame_id'] for record in records], dtype=np.int32)
    times = np.array([record['time_seconds'] for record in records])
    np.savez_compressed(run / 'poses.npz', T_object_to_camera=transforms, valid=valid,
                        mean_fit_residual_px=residuals, frame_ids=frame_ids, times_seconds=times,
                        sides=np.array(['L', 'R']))
    with np.load(run / 'poses.npz', allow_pickle=False) as saved:
        np.testing.assert_equal(saved['valid'], valid)
        np.testing.assert_allclose(saved['T_object_to_camera'], transforms, equal_nan=True)
    video_counts = {}
    for name in ['full_video_overlay.mp4', 'camera_crops.mp4']:
        capture = cv2.VideoCapture(str(run / name))
        video_counts[name] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert video_counts[name] == count
        assert capture.read()[0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
        assert capture.read()[0]
        capture.release()
    summary = {'status': 'PASS', 'processed_frames': count, 'target_poses': int(valid.size),
               'solved_poses': int(valid.sum()), 'failed_poses': int((~valid).sum()), 'solver_statuses': statuses,
               'coordinate_convention': 'X_camera = R_object_to_camera @ X_object + t_object_to_camera',
               'translation_units': 'mm', 'camera_axes': 'x right, y down, z forward',
               'pose_array_shape': list(transforms.shape), 'target_axis_order': ['L', 'R'],
               'target_identity': 'L/R follow the manually annotated image-side ROIs, not persistent camera IDs',
               'camera_kind': config['camera']['calibration_kind'], 'camera_sha256': config['camera_sha256'],
               'weights_sha256': config['weights_sha256'],
               'maximum_saved_pose_reprojection_difference_px': maximum_projection_difference,
               'median_fit_residual_original_image_px': float(np.nanmedian(residuals)),
               'p95_fit_residual_original_image_px': float(np.nanpercentile(residuals, 95)),
               'fit_residual_is_not_ground_truth_pose_accuracy': True,
               'video_frame_counts': video_counts, 'temporal_smoothing': False}
    (run / 'pose_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, default=root / 'datasets/action4/keypoints_3d.json')
    args = parser.parse_args()
    print(json.dumps(export_poses(args.run, args.geometry), indent=2))


if __name__ == '__main__':
    main()
