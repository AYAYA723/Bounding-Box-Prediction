import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DinoFeatureDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        metadata = json.loads((self.cache_dir / 'metadata.json').read_text())
        self.scene_dir = Path(metadata['scene_dir'])
        self.ids = metadata['image_ids']
        self.features = np.load(self.cache_dir / 'features.npy', mmap_mode='r')
        labels = json.loads((self.scene_dir / 'keypoints_2d.json').read_text())['images']
        self.corners = [torch.tensor(labels[key]['corners_2d'], dtype=torch.float32) for key in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index].copy()), self.corners[index]
