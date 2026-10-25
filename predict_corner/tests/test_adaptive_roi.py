import json
from pathlib import Path
import unittest

import numpy as np

from predict_corner.data_prep.prepare_video_crops import camera_body_boxes, camera_rois, overlap_fraction
from predict_corner.pose import crop_to_image, image_to_crop


class AdaptiveRoiTests(unittest.TestCase):
    def annotations(self):
        path = Path(__file__).resolve().parents[1] / 'data_prep/video_roi_annotations.json'
        return dict(json.loads(path.read_text()), roi_mode='adaptive')

    def test_every_frame_contains_its_body_and_stays_inside_video(self):
        config = self.annotations()
        width, height = config['video_size_wh']
        for frame in range(config['frame_count']):
            boxes = camera_body_boxes(config, frame)
            crops = camera_rois(config, frame)
            for body, crop in zip(boxes, crops):
                self.assertGreaterEqual(crop[0], 0)
                self.assertGreaterEqual(crop[1], 0)
                self.assertLessEqual(crop[2], width)
                self.assertLessEqual(crop[3], height)
                self.assertEqual((crop[2] - crop[0]) * 3, (crop[3] - crop[1]) * 4)
                self.assertGreater(overlap_fraction(crop, body), .999)

    def test_crop_size_changes_with_target_size(self):
        config = self.annotations()
        widths = [camera_rois(config, frame)[0, 2] - camera_rois(config, frame)[0, 0]
                  for frame in [0, 300, 900, 1500, 2400]]
        self.assertGreater(max(widths), min(widths) * 1.5)

    def test_neighbor_reduces_margin_without_changing_coordinate_mapping(self):
        config = {'video_size_wh': [640, 480], 'annotation_size_wh': [640, 480],
                  'roi_mode': 'adaptive', 'keyframes': [[0, [200, 200, 300, 275], [303, 200, 403, 275]]]}
        config_wide = dict(config, adaptive_other_fraction=1.)
        tight = camera_rois(config, 0)[0]
        wide = camera_rois(config_wide, 0)[0]
        self.assertLess(tight[2] - tight[0], wide[2] - wide[0])
        points = np.array([[200., 200.], [300., 275.]])
        np.testing.assert_allclose(crop_to_image(image_to_crop(points, tight), tight), points, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
