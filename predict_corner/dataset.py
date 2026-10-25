from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class CornerDataset(Dataset):
    def __init__(self, scene_dir):
        self.scene_dir = Path(scene_dir)
        self.label_paths = sorted((self.scene_dir / "heatmaps_boxdreamer").glob("*.npz"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        image_path = self.scene_dir / "rgb" / (label_path.stem + ".png")
        with Image.open(image_path) as image:
            x = self.transform(image.convert("RGB"))  # (3, 480, 640)
        with np.load(label_path, allow_pickle=False) as label:
            y = torch.from_numpy(label["heatmaps"])  # (8, 480, 640)
            corners_xy = torch.from_numpy(label["corners_xy"]).float()  # (8, 2)
        return x, y, corners_xy


class AppearanceDataset(Dataset):
    """Change backgrounds and appearance without moving the target corners."""
    def __init__(self, samples, masks):
        self.samples = samples
        self.masks = masks
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean.flatten().tolist(), self.std.flatten().tolist())
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, heatmaps, corners_xy = self.samples[index]
        image = TF.to_pil_image((x * self.std + self.mean).clamp(0, 1))
        noise = torch.rand(1, 3, 4, 4)
        background = TF.to_pil_image(F.interpolate(noise, size=x.shape[-2:], mode="bilinear", align_corners=False)[0])
        image = Image.composite(image, background, self.masks[index])
        return self.transform(image), heatmaps, corners_xy
