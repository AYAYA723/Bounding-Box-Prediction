import torch
from torch import nn


class TransformerCornerPredictor(nn.Module):
    def __init__(self, num_corners=8, hidden_dim=384, num_layers=6, num_heads=6):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=32, stride=32)
        self.position_embedding = nn.Parameter(torch.randn(1, 300, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False
        )
        self.projection = nn.Conv2d(hidden_dim, 512, kernel_size=1)

        # 与 ResNet 版本使用相同的 decoder 和 heatmap head。
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # (N, 256, 30, 40)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # (N, 128, 60, 80)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # (N, 64, 120, 160)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # (N, 32, 240, 320)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()  # (N, 16, 480, 640)
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(16, num_corners, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.patch_embed(x)  # (N, 384, 15, 20)
        N, C, H, W = features.shape
        tokens = features.flatten(start_dim=2).transpose(1, 2)  # (N, 300, 384)
        tokens = self.encoder(tokens + self.position_embedding)
        features = tokens.transpose(1, 2).reshape(N, C, H, W)
        features = self.projection(features)  # (N, 512, 15, 20)
        features = self.decoder(features)
        heatmaps = self.heatmap_head(features)  # (N, 8, 480, 640)
        return heatmaps
