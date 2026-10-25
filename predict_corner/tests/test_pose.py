import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from predict_corner.pose import crop_to_image, image_to_crop, load_camera, solve_box_pose


class PoseTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.points = np.array(json.loads((root / 'datasets/action4/keypoints_3d.json').read_text())['corners'])
        self.K = np.array([[800., 0, 500], [0, 820., 350], [0, 0, 1]])
        self.rvec = np.array([.4, -.3, .2])
        self.tvec = np.array([25., -12., 400.])

    def project(self, distortion=None):
        return cv2.projectPoints(self.points, self.rvec, self.tvec, self.K, distortion)[0].reshape(8, 2)

    def test_exact_pose_and_units(self):
        pose = solve_box_pose(self.points, self.project(), self.K)
        self.assertTrue(pose['success'])
        np.testing.assert_allclose(pose['t_object_to_camera'], self.tvec, atol=1e-6)
        np.testing.assert_allclose(pose['R_object_to_camera'], cv2.Rodrigues(self.rvec)[0], atol=1e-7)
        self.assertLess(pose['mean_reprojection_residual_px'], 1e-6)

    def test_radial_and_tangential_distortion(self):
        distortion = np.array([-.15, .03, .002, -.001, .01])
        pose = solve_box_pose(self.points, self.project(distortion), self.K, distortion)
        self.assertTrue(pose['success'])
        np.testing.assert_allclose(pose['t_object_to_camera'], self.tvec, atol=1e-5)

    def test_crop_roundtrip_and_original_intrinsics(self):
        roi = [71, 98, 800, 645]
        original = self.project()
        cropped = image_to_crop(original, roi)
        restored = crop_to_image(cropped, roi)
        np.testing.assert_allclose(restored, original, atol=1e-10)
        pose = solve_box_pose(self.points, restored, self.K)
        np.testing.assert_allclose(pose['t_object_to_camera'], self.tvec, atol=1e-6)

    def test_noisy_corners_are_fitted_by_one_rigid_transform(self):
        noisy = self.project() + np.random.default_rng(42).normal(0, 1, (8, 2))
        pose = solve_box_pose(self.points, noisy, self.K)
        self.assertTrue(pose['success'])
        self.assertLess(pose['mean_reprojection_residual_px'], 2.)
        R = np.asarray(pose['R_object_to_camera'])
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(R), 1.)

    def test_degenerate_corners_do_not_create_a_pose(self):
        pose = solve_box_pose(self.points, np.ones((8, 2)), self.K)
        self.assertFalse(pose['success'])
        self.assertNotIn('T_object_to_camera', pose)

    def test_camera_resolution_and_model_are_explicit(self):
        camera = {'model': 'pinhole', 'width': 1000, 'height': 700,
                  'cam_K': self.K.tolist(), 'dist_coeffs': [0.] * 5, 'calibration_kind': 'synthetic'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera.json'
            path.write_text(json.dumps(camera))
            K, distortion, _ = load_camera(path, (1000, 700))
            np.testing.assert_equal(K, self.K)
            self.assertEqual(len(distortion), 5)
            with self.assertRaises(ValueError):
                load_camera(path, (640, 480))
            camera['model'] = 'fisheye'
            path.write_text(json.dumps(camera))
            with self.assertRaises(ValueError):
                load_camera(path, (1000, 700))


if __name__ == '__main__':
    unittest.main()
