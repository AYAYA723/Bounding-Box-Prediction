import unittest

import torch

from predict_corner.model import CornerPredictor
from predict_corner.tools.fit import replace_batch_norm


class NormalizationTests(unittest.TestCase):
    def test_group_normalization_keeps_single_image_predictions_independent_of_batch(self):
        torch.manual_seed(1)
        model = CornerPredictor()
        replace_batch_norm(model)
        image = torch.randn(1, 3, 64, 64)
        companion = torch.randn_like(image)
        model.train()
        with torch.no_grad():
            single_maps, single_xy = model(image)
            batch_maps, batch_xy = model(torch.cat((image, companion)))
            model.eval()
            eval_maps, eval_xy = model(image)
        torch.testing.assert_close(single_maps, batch_maps[:1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(single_xy, batch_xy[:1], atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(single_maps, eval_maps)
        torch.testing.assert_close(single_xy, eval_xy)


if __name__ == "__main__":
    unittest.main()
