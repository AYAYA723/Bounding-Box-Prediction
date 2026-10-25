import unittest

import torch

from supervision import BoxDreamerLoss, normalize_coordinates, spatial_soft_argmax


class SupervisionTests(unittest.TestCase):
    def test_uniform_map_center_on_non_square_image(self):
        logits = torch.zeros(2, 8, 5, 9)
        xy = spatial_soft_argmax(logits)
        torch.testing.assert_close(xy, torch.tensor([4.0, 2.0]).expand(2, 8, 2))

    def test_strong_peak_keeps_xy_order(self):
        logits = torch.zeros(1, 8, 5, 9)
        logits[:, :, 1, 7] = 50
        xy = spatial_soft_argmax(logits)
        torch.testing.assert_close(xy, torch.tensor([7.0, 1.0]).expand(1, 8, 2))

    def test_global_logit_offset_does_not_move_coordinates(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 8, 5, 9, dtype=torch.float64)
        torch.testing.assert_close(spatial_soft_argmax(logits), spatial_soft_argmax(logits + 35))

    def test_normalization_uses_width_for_x_height_for_y(self):
        points = torch.tensor([[[0., 0.], [4., 2.], [8., 4.]]])
        expected = torch.tensor([[[-1., -1.], [-1/9, -1/5], [7/9, 3/5]]])
        torch.testing.assert_close(normalize_coordinates(points, (5, 9)), expected)

    def test_loss_matches_paper_weight_and_mean_reduction(self):
        pred = torch.full((2, 8, 3, 5), .4)
        target = torch.full_like(pred, .2)
        pred_xy = torch.zeros(2, 8, 2)
        target_xy = torch.ones_like(pred_xy)
        loss, coarse, fine = BoxDreamerLoss()(pred, pred_xy, target, target_xy)
        expected_coarse = 0.5 * .2**2
        expected_fine = ((2/5)**2 + (2/3)**2) / 4
        self.assertAlmostEqual(coarse.item(), expected_coarse, places=7)
        self.assertAlmostEqual(fine.item(), expected_fine, places=7)
        self.assertAlmostEqual(loss.item(), expected_coarse + 2 * expected_fine, places=7)

    def test_perfect_predictions_have_zero_loss(self):
        maps = torch.rand(2, 8, 5, 9)
        points = torch.rand(2, 8, 2)
        loss, coarse, fine = BoxDreamerLoss()(maps, points, maps, points)
        self.assertEqual((loss.item(), coarse.item(), fine.item()), (0., 0., 0.))

    def test_fine_loss_alone_has_nonzero_logit_gradient(self):
        logits = torch.zeros(1, 8, 5, 9, requires_grad=True)
        xy = spatial_soft_argmax(logits)
        _, _, fine = BoxDreamerLoss()(logits.sigmoid(), xy, logits.sigmoid().detach(),
                                     torch.tensor([1., 1.]).expand(1, 8, 2))
        gradient = torch.autograd.grad(fine, logits)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)

    def test_coordinate_derivative_matches_finite_difference(self):
        torch.manual_seed(2)
        logits = torch.randn(1, 2, 3, 5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(spatial_soft_argmax, (logits,)))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
