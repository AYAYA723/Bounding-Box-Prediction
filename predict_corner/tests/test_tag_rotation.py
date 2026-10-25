import unittest

import cv2
import numpy as np

from predict_corner.tools.check_tag_rotation import compare_rotation


class TagRotationTests(unittest.TestCase):
    def example(self, flip_last=False):
        K = np.array([[1000., 0, 640.], [0, 1000., 480.], [0, 0, 1.]])
        square = np.float64([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        Q = cv2.Rodrigues(np.array([-.3, .6, .2]))[0]
        records, observations = [], []
        for index in range(6):
            rvec = np.array([.2, -.1, .01 * index])
            tag_R = cv2.Rodrigues(rvec)[0]
            net_R = tag_R @ Q
            if flip_last and index == 5:
                net_R = net_R @ cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
            points = cv2.projectPoints(square, rvec, np.array([-.5, -.5, 4.]), K, None)[0].reshape(4, 2)
            observations.append({'frame_id': index, 'time_seconds': index * .2,
                                 'tag_id': 18, 'corners_xy': points.tolist()})
            records.append({'frame_id': index, 'targets': [
                {'side': 'L', 'pose': {'success': True, 'R_object_to_camera': net_R.tolist()}}]})
        return records, observations, {'cam_K': K.tolist(), 'dist_coeffs': [0.] * 5}

    def test_unknown_fixed_marker_to_body_rotation_cancels(self):
        report = compare_rotation(*self.example())
        self.assertEqual(report['summary']['rotation_pairs'], 5)
        self.assertLess(report['summary']['relative_rotation_disagreement_deg']['maximum'], .001)

    def test_correspondence_flip_is_not_counted_as_real_motion(self):
        report = compare_rotation(*self.example(flip_last=True))
        self.assertGreater(report['summary']['relative_rotation_disagreement_deg']['maximum'], 179.)


if __name__ == '__main__':
    unittest.main()
