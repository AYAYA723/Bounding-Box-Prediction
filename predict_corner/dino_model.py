from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from predict_corner.heatmap_decoder import HeatmapDecoder


class DinoCornerPredictor(nn.Module):
    def __init__(self, image_size=(480, 640), num_corners=8, hidden_dim=768,
                 num_layers=12, num_heads=8, decoder_temperature=.25, pretrained=True):
        super().__init__()
        self.patch_size = 14
        cache = Path(__file__).resolve().parents[1] / '.cache/torch'
        torch.hub.set_dir(str(cache))
        self.encoder = torch.hub.load(
            str(cache / 'facebookresearch_dinov2_main'), 'dinov2_vitb14_reg',
            source='local', pretrained=pretrained
        )
        self.encoder.eval()
        # Freeze the pretrained encoder.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.decoder = HeatmapDecoder(
            image_size=image_size, patch_size=self.patch_size, input_dim=768,
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            num_corners=num_corners, decoder_temperature=decoder_temperature
        )

    @property
    def decoder_temperature(self):
        return self.decoder.decoder_temperature

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Keep encoder statistics fixed.
        return self

    def encode(self, x):
        H, W = x.shape[-2:]
        x = F.pad(x, (0, -W % self.patch_size, 0, -H % self.patch_size))  # (N, 3, 490, 644)
        with torch.no_grad():
            # Patch tokens only: (N, 1610, 768).
            return self.encoder.forward_features(x)['x_norm_patchtokens']

    def decode(self, features):
        return self.decoder(features)

    def forward(self, x):
        return self.decode(self.encode(x))
