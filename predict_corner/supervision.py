import torch
from torch import nn


def spatial_soft_argmax(logits, temperature=0.25):
    """Extract differentiable (x, y) pixel coordinates from pre-Sigmoid logits."""
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    N, C, H, W = logits.shape
    probabilities = (logits.flatten(start_dim=2) / temperature).softmax(dim=-1).reshape(N, C, H, W)
    xs = torch.arange(W, device=logits.device, dtype=logits.dtype)
    ys = torch.arange(H, device=logits.device, dtype=logits.dtype)
    x = (probabilities.sum(dim=2) * xs).sum(dim=-1)
    y = (probabilities.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack((x, y), dim=-1)  # (N, 8, 2)


def make_target_heatmaps(corners_xy, image_size):
    """Evaluate the same BoxDreamer kernel as the NumPy data preparation code."""
    H, W = image_size
    corners = corners_xy.float()
    center = corners.mean(dim=1, keepdim=True)
    denominators = ((corners - center) ** 2).sum(dim=-1) / 100
    xs = torch.arange(W, device=corners.device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.arange(H, device=corners.device, dtype=torch.float32).view(1, 1, H, 1)
    distance = torch.sqrt((xs - corners[..., 0, None, None]) ** 2 + (ys - corners[..., 1, None, None]) ** 2)
    return torch.exp(-distance / denominators[..., None, None])


def normalize_coordinates(corners_xy, image_size):
    H, W = image_size
    return 2 * corners_xy / corners_xy.new_tensor([W, H]) - 1


class BoxDreamerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, heatmaps, corners_xy, target_heatmaps, target_corners_xy):
        coarse_loss = self.smooth_l1(heatmaps, target_heatmaps)
        pred_normalized = normalize_coordinates(corners_xy, heatmaps.shape[-2:])
        target_normalized = normalize_coordinates(target_corners_xy, heatmaps.shape[-2:])
        fine_loss = self.smooth_l1(pred_normalized, target_normalized)
        loss = coarse_loss + 2.0 * fine_loss
        return loss, coarse_loss, fine_loss
