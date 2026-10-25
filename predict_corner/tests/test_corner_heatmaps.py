"""Numerical regression tests for the paper's unusual distance kernel."""

import unittest

import numpy as np

from predict_corner.data_prep.generate_corner_heatmaps import make_corner_heatmaps


class CornerHeatmapTests(unittest.TestCase):
    def setUp(self):
        # All corners are five pixels from center (5,5); denominator = 0.25.
        self.corners = np.array(
            [[2, 1], [8, 1], [8, 9], [2, 9], [1, 2], [9, 2], [9, 8], [1, 8]],
            dtype=np.float64,
        )

    def test_known_distances_and_xy_axis_order(self):
        heatmaps, center, denominators = make_corner_heatmaps(self.corners, (13, 11))
        self.assertEqual(heatmaps.shape, (8, 11, 13))
        self.assertEqual(heatmaps.dtype, np.float32)
        np.testing.assert_array_equal(center, [5, 5])
        np.testing.assert_array_equal(denominators, np.full(8, 0.25))
        self.assertEqual(float(heatmaps[0, 1, 2]), 1.0)
        # At distance 2: exp(-2/0.25), NOT exp(-4/0.25).
        self.assertAlmostEqual(float(heatmaps[0, 1, 4]), 0.00033546262790251185, places=10)
        self.assertAlmostEqual(float(heatmaps[0, 3, 2]), 0.00033546262790251185, places=10)

    def test_fractional_corners_not_rounded_or_peak_normalized(self):
        corners = self.corners + np.array([0.25, 0.125])
        heatmaps, _, _ = make_corner_heatmaps(corners, (13, 11))
        # At (2,1), distance=sqrt(.25^2+.125^2), denominator unchanged.
        self.assertAlmostEqual(float(heatmaps[0, 1, 2]), 0.3269218953517579, places=7)
        self.assertLess(float(heatmaps[0].max()), 1.0)
        self.assertEqual(np.unravel_index(heatmaps[0].argmax(), heatmaps[0].shape), (1, 2))

    def test_channel_permutation_preserves_corner_correspondence(self):
        original, _, _ = make_corner_heatmaps(self.corners, (13, 11))
        order = [6, 3, 0, 7, 4, 1, 5, 2]
        permuted, _, _ = make_corner_heatmaps(self.corners[order], (13, 11))
        np.testing.assert_array_equal(permuted, original[order])

    def test_scale_is_computed_separately_for_each_corner(self):
        corners = self.corners.copy()
        corners[0] += [0.5, 0]
        _, center, denominators = make_corner_heatmaps(corners, (13, 11))
        np.testing.assert_array_equal(center, [5.0625, 5])
        self.assertAlmostEqual(float(denominators[0]), 0.2256640625)
        self.assertNotEqual(float(denominators[0]), float(denominators[1]))

    def test_outside_corner_is_not_clamped(self):
        corners = self.corners + [-3, 0]
        heatmaps, _, _ = make_corner_heatmaps(corners, (13, 11))
        self.assertAlmostEqual(float(heatmaps[0, 1, 0]), 0.01831563888873418, places=8)
        self.assertLess(float(heatmaps[0].max()), 1.0)

    def test_invalid_or_degenerate_labels_fail(self):
        for corners in (np.zeros((8, 2)), np.zeros((7, 2)), np.full((8, 2), np.nan)):
            with self.assertRaises(ValueError):
                make_corner_heatmaps(corners, (13, 11))
        for size in ((0, 11), (13, -1), (13.0, 11), (True, 11)):
            with self.assertRaises(ValueError):
                make_corner_heatmaps(self.corners, size)


if __name__ == "__main__":
    unittest.main()
