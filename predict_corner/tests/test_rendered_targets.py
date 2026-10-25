import unittest

import numpy as np
import torch

from predict_corner.data_prep.generate_corner_heatmaps import make_corner_heatmaps
from predict_corner.supervision import make_target_heatmaps, spatial_soft_argmax


class RenderedTargetTests(unittest.TestCase):
    def test_batched_targets_match_the_existing_paper_formula(self):
        corners = np.array([[95.3, 80.2], [240.4, 67.8], [255.2, 201.1], [83.7, 210.9],
                            [118.3, 105.5], [229.8, 90.3], [241.2, 187.6], [107.3, 192.8]])
        batch = np.stack((corners, corners + [190.2, 150.7]))
        targets = make_target_heatmaps(torch.tensor(batch, dtype=torch.float32), (480, 640)).numpy()
        for index in range(2):
            expected = make_corner_heatmaps(batch[index], (640, 480))[0]
            np.testing.assert_allclose(targets[index], expected, atol=1e-5, rtol=1e-5)

    def test_low_precision_decode_preserves_large_pixel_coordinates(self):
        logits = torch.full((1, 1, 480, 640), -20., dtype=torch.bfloat16)
        logits[0, 0, 301, 399] = 20
        coordinates = spatial_soft_argmax(logits, .25)
        torch.testing.assert_close(coordinates, torch.tensor([[[399., 301.]]]), atol=.01, rtol=0)


if __name__ == "__main__":
    unittest.main()
