from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        return x, y
