import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from predict_corner.tools.calibrate_video_tags import heldout_error


def select_views(observations, minimum_side=40, minimum_motion=35):
    selected = []
    for row in sorted(observations, key=lambda value: value['minimum_side_px'], reverse=True):
        if row['minimum_side_px'] < minimum_side:
            continue
        points = np.asarray(row['corners_xy'])
        redundant = any(other['tag_id'] == row['tag_id'] and
                        np.linalg.norm(points - other['corners_xy'], axis=1).mean() < minimum_motion
                        for other in selected)
        if not redundant:
            selected.append(row)
    return sorted(selected, key=lambda value: (value['frame_id'], value['tag_id']))


def fit_camera(observations, size, model='pinhole', focals=None):
    """Estimate focal length with square pixels, centered principal point and zero skew."""
    width, height = size
    focals = focals if focals is not None else [width * .25, width * .5, width * .75]
    square = np.float32([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    objects = [square.copy() for _ in observations]
    images = [np.asarray(row['corners_xy'], dtype=np.float32) for row in observations]
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    if model == 'pinhole':
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    elif model == 'radial1':
        flags |= cv2.CALIB_FIX_K2
    candidates = []
    for focal in focals:
        K = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.]])
        result = cv2.calibrateCameraExtended(objects, images, size, K, None, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
        rms, K, distortion, _, _, std, _, per_view = result
        candidates.append({'rms_px': float(rms), 'cam_K': K.tolist(),
                           'dist_coeffs': distortion.reshape(-1).tolist(), 'views': len(observations),
                           'per_view_rms_px': per_view.reshape(-1).tolist(),
                           'intrinsics_std': std.reshape(-1).tolist(), 'initial_focal_px': float(focal)})
    return min(candidates, key=lambda value: value['rms_px'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=40)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    source = json.loads(args.observations.read_text())
    size = tuple(source['image_size_wh'])
    selected = select_views(source['observations'])
    assert len(selected) >= 15
    candidates = {name: fit_camera(selected, size, name) for name in ['pinhole', 'radial1', 'radial2']}
    # Keep the simpler zero-distortion model when radial terms are weakly constrained.
    first = candidates['pinhole']
    inliers = [row for row, error in zip(selected, first['per_view_rms_px']) if error <= 1.5]
    rejected = [row for row, error in zip(selected, first['per_view_rms_px']) if error > 1.5]
    final = fit_camera(inliers, size)
    focal = final['cam_K'][0][0]
    folds = []
    for fold in range(4):
        train = [row for row in inliers if row['frame_id'] // 150 % 4 != fold]
        test = [row for row in selected if row['frame_id'] // 150 % 4 == fold]
        if len(train) < 10 or not test:
            continue
        fit = fit_camera(train, size)
        median, mean = heldout_error(test, fit)
        folds.append({'fold': fold, 'train_views': len(train), 'test_views': len(test),
                      'focal_px': fit['cam_K'][0][0], 'heldout_median_rms_px': median,
                      'heldout_mean_rms_px': mean})
    random = np.random.default_rng(42)
    frame_ids = np.unique([row['frame_id'] for row in inliers])
    bootstrap = []
    for index in range(args.bootstrap):
        sampled_ids = random.choice(frame_ids, len(frame_ids), replace=True)
        sampled = [row for frame_id in sampled_ids for row in inliers if row['frame_id'] == frame_id]
        fit = fit_camera(sampled, size, focals=[focal * .8, focal * 1.2])
        bootstrap.append(fit['cam_K'][0][0])
        if (index + 1) % 10 == 0:
            print(f'Bootstrap {index + 1}/{args.bootstrap}', flush=True)
    thresholds = {}
    for minimum_side in [35, 40, 45, 50]:
        rows = select_views(source['observations'], minimum_side)
        fit = fit_camera(rows, size)
        thresholds[str(minimum_side)] = {'views': len(rows), 'focal_px': fit['cam_K'][0][0], 'rms_px': fit['rms_px']}
    camera = {'model': 'pinhole', 'calibration_kind': 'estimated', 'width': size[0], 'height': size[1],
              'cam_K': final['cam_K'], 'dist_coeffs': final['dist_coeffs'],
              'method': 'multiple AprilTag 36h11 views from the original video',
              'rms_reprojection_px': final['rms_px'], 'detected_views': len(source['observations']),
              'selected_views': len(selected), 'used_views': len(inliers), 'tag_ids': [18, 19],
              'principal_point': 'fixed at image center, not independently estimated',
              'pixel_aspect_ratio': 'fx=fy assumed', 'distortion': 'fixed at zero, not independently measured',
              'focal_bootstrap_5_95_percentile_px': np.percentile(bootstrap, [5, 95]).tolist(),
              'bootstrap_limitation': 'conditional on detected points and geometric assumptions; excludes systematic errors',
              'assumptions': ['physical markers are planar squares', 'intrinsics remain constant across the video',
                              'zero skew', 'square pixels', 'centered principal point', 'negligible lens distortion'],
              'not_ground_truth_calibration': True,
              'source_observations_sha256': hashlib.sha256(args.observations.read_bytes()).hexdigest(),
              'source_video_sha256': source['video_sha256']}
    report = {'camera': camera, 'candidate_models': candidates, 'final_fit': final,
              'temporal_folds': folds, 'bootstrap_focal_px': bootstrap, 'minimum_side_sensitivity': thresholds,
              'used_observations': inliers, 'rejected_observations': rejected,
              'focal_threshold_range_px': [min(row['focal_px'] for row in thresholds.values()),
                                           max(row['focal_px'] for row in thresholds.values())]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'camera.json').write_text(json.dumps(camera, indent=2) + '\n')
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'camera': camera, 'temporal_folds': folds, 'thresholds': thresholds}, indent=2), flush=True)


if __name__ == '__main__':
    main()
