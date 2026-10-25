import json
from pathlib import Path

import cv2
import numpy as np


def load_camera(path, image_size):
    """Read a pinhole calibration at its original image resolution."""
    camera = json.loads(Path(path).read_text(encoding='utf-8'))
    if camera['model'] != 'pinhole':
        raise ValueError('PnP expects a pinhole camera; fisheye images need their own rectification.')
    if [camera['width'], camera['height']] != list(image_size):
        raise ValueError('Calibration resolution must match the original image, before ROI cropping.')
    K = np.asarray(camera['cam_K'], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera['dist_coeffs'], dtype=np.float64)
    return K, distortion, camera


def crop_to_image(points, roi, crop_size=(640, 480)):
    roi = np.asarray(roi, dtype=np.float64)
    scale = (roi[2:] - roi[:2]) / np.asarray(crop_size)
    return (np.asarray(points) + .5) * scale - .5 + roi[:2]  # Match PIL pixel centers.


def image_to_crop(points, roi, crop_size=(640, 480)):
    roi = np.asarray(roi, dtype=np.float64)
    scale = (roi[2:] - roi[:2]) / np.asarray(crop_size)
    return (np.asarray(points) - roi[:2] + .5) / scale - .5


def project_box(corners_3d, R, t, K, distortion=None):
    """Project the fixed 3D bounding box using an object-to-camera pose."""
    points = np.asarray(corners_3d, dtype=np.float64).reshape(8, 3)
    rotation = np.asarray(R, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(t, dtype=np.float64).reshape(3)
    rvec = cv2.Rodrigues(rotation)[0]
    return cv2.projectPoints(points, rvec, translation, np.asarray(K, dtype=np.float64), distortion)[0].reshape(8, 2)


def solve_box_pose(corners_3d, corners_xy, K, distortion=None):
    """Fit one rigid pose to eight ordered corners, in object-to-camera convention."""
    object_points = np.ascontiguousarray(corners_3d, dtype=np.float64).reshape(8, 3)
    image_points = np.ascontiguousarray(corners_xy, dtype=np.float64).reshape(8, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(image_points).all():
        return {'success': False, 'status': 'nonfinite_corners'}
    if np.linalg.matrix_rank(image_points - image_points.mean(0), tol=1e-6) < 2:
        return {'success': False, 'status': 'degenerate_corners'}
    try:
        # Initialize with SQPnP, then refine reprojection error.
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, distortion, flags=cv2.SOLVEPNP_SQPNP)
        if success:
            rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, distortion, rvec, tvec)
    except cv2.error:
        return {'success': False, 'status': 'solver_failed'}
    if not success:
        return {'success': False, 'status': 'solver_failed'}
    R = cv2.Rodrigues(rvec)[0]
    camera_points = object_points @ R.T + tvec.reshape(3)
    if not np.isfinite(camera_points).all() or (camera_points[:, 2] <= 0).any():
        return {'success': False, 'status': 'invalid_depth'}
    projected = cv2.projectPoints(object_points, rvec, tvec, K, distortion)[0].reshape(8, 2)
    residuals = np.linalg.norm(projected - image_points, axis=1)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = R, tvec.reshape(3)
    return {'success': True, 'status': 'solved', 'rvec': rvec.reshape(3).tolist(),
            'R_object_to_camera': R.tolist(), 't_object_to_camera': tvec.reshape(3).tolist(),
            'T_object_to_camera': transform.tolist(), 'reprojected_corners_xy': projected.tolist(),
            'reprojection_residuals_px': residuals.tolist(),
            'mean_reprojection_residual_px': float(residuals.mean()),
            'min_corner_depth': float(camera_points[:, 2].min())}
