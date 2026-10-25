import unittest

import torch
from PIL import Image

from predict_corner.dataset import AppearanceDataset


class AppearanceTests(unittest.TestCase):
    def test_appearance_augmentation_preserves_heatmaps_and_corner_coordinates(self):
        torch.manual_seed(3)
        sample = (torch.zeros(3, 32, 48), torch.rand(8, 32, 48), torch.rand(8, 2))
        dataset = AppearanceDataset([sample], [Image.new("L", (48, 32), 0)])
        first, maps, xy = dataset[0]
        second, second_maps, second_xy = dataset[0]
        torch.testing.assert_close(maps, sample[1])
        torch.testing.assert_close(xy, sample[2])
        torch.testing.assert_close(second_maps, sample[1])
        torch.testing.assert_close(second_xy, sample[2])
        self.assertEqual(first.shape, sample[0].shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
