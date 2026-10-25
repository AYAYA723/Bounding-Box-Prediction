import unittest
from unittest.mock import patch

import torch
from torch import nn

from predict_corner.dino_model import DinoCornerPredictor
from predict_corner.heatmap_decoder import HeatmapDecoder, position_embedding, unpatchify
from predict_corner.supervision import BoxDreamerLoss
from predict_corner.tools.train_synthetic import evaluate


class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 768, kernel_size=14, stride=14)

    def forward_features(self, x):
        return {'x_norm_patchtokens': self.patch(x).flatten(2).transpose(1, 2)}


class HeatmapDecoderTests(unittest.TestCase):
    def test_feature_validation_uses_output_image_resolution(self):
        decoder = HeatmapDecoder(image_size=(29, 43), hidden_dim=32, num_layers=1, num_heads=4)
        features = torch.randn(2, 12, 768)
        target = torch.tensor([[[8., 8.], [32., 8.], [36., 20.], [4., 20.],
                                [12., 12.], [28., 12.], [32., 24.], [8., 24.]]]).repeat(2, 1, 1)
        data = torch.utils.data.TensorDataset(features, target)
        data.ids = ['0', '1']
        report = evaluate(decoder, torch.utils.data.DataLoader(data, batch_size=2), torch.device('cpu'), BoxDreamerLoss())
        self.assertEqual(report['image_count'], 2)
        self.assertTrue(torch.isfinite(torch.tensor(report['total_loss'])))

    def test_paper_depth_and_linear_pixel_head(self):
        decoder = HeatmapDecoder(image_size=(28, 42), hidden_dim=32, num_heads=4)
        self.assertEqual(len(decoder.transformer), 12)
        self.assertEqual(decoder.heatmap_projection.out_features, 14 * 14 * 8)
        self.assertFalse(any(isinstance(module, nn.ConvTranspose2d) for module in decoder.modules()))

    def test_unpatchify_preserves_channel_and_xy_order(self):
        image = torch.arange(2 * 4 * 6).reshape(1, 2, 4, 6)
        patches = image.reshape(1, 2, 2, 2, 3, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 6, 8)
        torch.testing.assert_close(unpatchify(patches, (2, 3), 2, 2), image)

    def test_rectangular_position_embedding_varies_along_both_axes(self):
        embedding = position_embedding((2, 3), 32).reshape(2, 3, 32)
        torch.testing.assert_close(embedding[0, 0, :16], embedding[1, 0, :16])
        torch.testing.assert_close(embedding[0, 0, 16:], embedding[0, 1, 16:])
        self.assertFalse(torch.equal(embedding[0, 0, :16], embedding[0, 1, :16]))
        self.assertFalse(torch.equal(embedding[0, 0, 16:], embedding[1, 0, 16:]))

    def test_padding_is_removed_before_heatmap_and_coordinate_outputs(self):
        decoder = HeatmapDecoder(image_size=(29, 43), hidden_dim=32, num_layers=2, num_heads=4)
        heatmaps, xy = decoder(torch.randn(2, 12, 768))
        self.assertEqual(heatmaps.shape, (2, 8, 29, 43))
        self.assertEqual(xy.shape, (2, 8, 2))
        self.assertTrue(((xy[..., 0] >= 0) & (xy[..., 0] <= 42)).all())
        self.assertTrue(((xy[..., 1] >= 0) & (xy[..., 1] <= 28)).all())

    def test_attention_layers_have_independent_initial_weights(self):
        decoder = HeatmapDecoder(image_size=(28, 42), hidden_dim=32, num_layers=2, num_heads=4)
        self.assertFalse(torch.equal(decoder.transformer[0].qkv.weight, decoder.transformer[1].qkv.weight))

    @patch('torch.hub.load', side_effect=lambda *args, **kwargs: TinyDino())
    def test_frozen_encoder_and_decoder_gradient(self, load):
        model = DinoCornerPredictor(image_size=(29, 43), hidden_dim=32, num_layers=1, num_heads=4)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.decoder.training)
        before = model.encoder.patch.weight.detach().clone()
        image = torch.randn(2, 3, 29, 43)
        prediction, xy = model(image)
        target_xy = torch.tensor([10., 12.]).expand(2, 8, 2)
        total, _, fine = BoxDreamerLoss()(prediction, xy, torch.zeros_like(prediction), target_xy)
        fine_gradient = torch.autograd.grad(fine, model.decoder.transformer[0].qkv.weight, retain_graph=True)[0]
        self.assertGreater(fine_gradient.norm().item(), 0.)
        total.backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        torch.testing.assert_close(model.encoder.patch.weight, before)
        self.assertGreater(model.decoder.heatmap_projection.weight.grad.norm().item(), 0.)
        model.eval()
        with torch.no_grad():
            online = model(image)
            cached = model.decoder(model.encode(image))
        torch.testing.assert_close(online, cached)

    def test_bfloat16_attention_keeps_finite_outputs_and_gradients(self):
        decoder = HeatmapDecoder(image_size=(28, 42), hidden_dim=32, num_layers=1, num_heads=4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            heatmaps, xy = decoder(torch.randn(2, 6, 768))
            loss = heatmaps.float().square().mean() + xy.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(xy).all())
        self.assertTrue(torch.isfinite(decoder.heatmap_projection.weight.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
