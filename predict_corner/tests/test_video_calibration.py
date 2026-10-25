import unittest

import cv2
import numpy as np

from predict_corner.tools.refine_video_calibration import fit_camera, select_views


class VideoCalibrationTests(unittest.TestCase):
    def make_observations(self, noise=0.):
        random = np.random.default_rng(42)
        K = np.array([[1560., 0, 1623.5], [0, 1560., 1231.5], [0, 0, 1.]])
        square = np.float64([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        observations = []
        for index in range(35):
            rvec = random.uniform(-.8, .8, 3)
            tvec = np.array([random.uniform(-2, 1), random.uniform(-2, 1), random.uniform(6, 10)])
            points = cv2.projectPoints(square, rvec, tvec, K, None)[0].reshape(4, 2)
            points += random.normal(0, noise, points.shape)
            observations.append({'frame_id': index * 30, 'tag_id': 18, 'corners_xy': points.tolist(),
                                 'minimum_side_px': float(np.linalg.norm(points - np.roll(points, 1, 0), axis=1).min())})
        return observations, K

    def test_known_focal_recovered_from_multiple_single_square_views(self):
        observations, K = self.make_observations()
        fit = fit_camera(observations, (3248, 2464))
        np.testing.assert_allclose(fit['cam_K'], K, atol=.02, rtol=0)
        self.assertLess(fit['rms_px'], .001)

    def test_subpixel_noise_preserves_focal_accuracy(self):
        observations, K = self.make_observations(.2)
        fit = fit_camera(observations, (3248, 2464))
        self.assertLess(abs(fit['cam_K'][0][0] / K[0, 0] - 1), .05)

    def test_duplicate_poses_and_small_tags_are_removed(self):
        observations, _ = self.make_observations()
        original = observations[0]
        duplicate = dict(original, frame_id=1)
        small = dict(original, frame_id=2, minimum_side_px=10.)
        selected = select_views([original, duplicate, small])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['frame_id'], original['frame_id'])


if __name__ == '__main__':
    unittest.main()
