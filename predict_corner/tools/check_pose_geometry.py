import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from predict_corner.pose import solve_box_pose
from predict_corner.tools.evaluate_pose import rotation_error_degrees


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, default=root / 'datasets/action4/synthetic_v2/val')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cameras = json.loads((args.scene / 'scene_camera.json').read_text())
    poses = json.loads((args.scene / 'scene_gt.json').read_text())
    geometry = json.loads((args.scene / 'keypoints_3d.json').read_text())
    points = np.asarray(geometry['corners'], dtype=np.float64)
    generator = np.random.default_rng(42)
    perturbations = {key: generator.normal(0, 1, (8, 2)) for key in cameras}
    configurations = [('exact', 0., 1.), ('noise_1px', 1., 1.), ('noise_3px', 3., 1.),
                      ('noise_10px', 10., 1.), ('focal_minus_10pct', 0., .9), ('focal_plus_10pct', 0., 1.1)]
    results = {}
    for name, noise, focal_scale in configurations:
        rows = []
        for key, camera in cameras.items():
            K = np.array(camera['cam_K']).reshape(3, 3)
            R = np.array(poses[key][0]['cam_R_m2c']).reshape(3, 3)
            t = np.array(poses[key][0]['cam_t_m2c'])
            gt_xy = cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, K, None)[0].reshape(8, 2)
            xy = gt_xy + perturbations[key] * noise
            estimated_K = K.copy()
            estimated_K[0, 0] *= focal_scale
            estimated_K[1, 1] *= focal_scale
            pose = solve_box_pose(points, xy, estimated_K)
            row = {'success': pose['success']}
            if pose['success']:
                row.update({'rotation_deg': rotation_error_degrees(pose['R_object_to_camera'], R),
                            'translation_mm': float(np.linalg.norm(np.asarray(pose['t_object_to_camera']) - t)),
                            'reprojection_residual_px': pose['mean_reprojection_residual_px'],
                            'corner_error_to_gt_px': float(np.linalg.norm(np.asarray(pose['reprojected_corners_xy']) - gt_xy, axis=1).mean())})
            rows.append(row)
        solved = [row for row in rows if row['success']]
        results[name] = {'images': len(rows), 'solved': len(solved), 'corner_noise_sigma_px': noise,
                         'focal_scale': focal_scale,
                         **{f'median_{key}': float(np.median([row[key] for row in solved]))
                            for key in ['rotation_deg', 'translation_mm', 'reprojection_residual_px', 'corner_error_to_gt_px']}}
    assert results['exact']['median_translation_mm'] < .001
    assert results['exact']['median_rotation_deg'] < .001
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
