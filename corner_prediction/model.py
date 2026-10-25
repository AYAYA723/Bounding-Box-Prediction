from torch import nn
from torchvision.models import resnet18

from supervision import spatial_soft_argmax


class CornerPredictor(nn.Module):
    def __init__(self, num_corners=8, weights=None):
        super().__init__()
        self.num_corners = num_corners

        resnet = resnet18(weights=weights)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )  # (N, 512, 15, 20)

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
            nn.Conv2d(16, self.num_corners, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)  # (N, 512, 15, 20)
        features = self.decoder(features)  # (N, 16, 480, 640)
        logits = self.heatmap_head[0](features)
        heatmaps = self.heatmap_head[1](logits)  # (N, 8, 480, 640)
        corners_xy = spatial_soft_argmax(logits)  # (N, 8, 2)
        return heatmaps, corners_xy
