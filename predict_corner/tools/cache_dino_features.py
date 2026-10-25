import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from predict_corner.dataset import RenderedCornerDataset
from predict_corner.dino_model import DinoCornerPredictor


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root / 'datasets/action4/dino_features_v1')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=False)
    model = DinoCornerPredictor().cuda().eval()
    weights = root / '.cache/torch/checkpoints/dinov2_vitb14_reg4_pretrain.pth'
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    for split in ['train', 'val']:
        dataset = RenderedCornerDataset(root / 'datasets/action4/synthetic_v2' / split)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)
        output = args.output / split
        output.mkdir()
        features = np.lib.format.open_memmap(
            output / 'features.npy', mode='w+', dtype=np.float32, shape=(len(dataset), 1610, 768))
        started = time.perf_counter()
        offset = 0
        for x, _ in loader:
            encoded = model.encode(x.cuda()).cpu().numpy()
            features[offset:offset + len(x)] = encoded
            offset += len(x)
            if offset % 100 < args.batch_size or offset == len(dataset):
                features.flush()
                print(f'{split}: {offset}/{len(dataset)}, elapsed={time.perf_counter() - started:.1f}s', flush=True)
        features.flush()
        metadata = {'scene_dir': str(dataset.scene_dir), 'image_ids': dataset.ids,
                    'shape': list(features.shape), 'dtype': 'float32', 'encoder_precision': 'float32',
                    'encoder': 'dinov2_vitb14_reg', 'encoder_weights_sha256': weights_sha,
                    'grid_size': [35, 46], 'image_size': [480, 640], 'padded_image_size': [490, 644],
                    'normalization': 'ImageNet mean/std, no additional crop or resize',
                    'labels_sha256': hashlib.sha256((dataset.scene_dir / 'keypoints_2d.json').read_bytes()).hexdigest(),
                    'seconds': time.perf_counter() - started}
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        del features
    print(f'Cached frozen float32 features: {args.output}', flush=True)


if __name__ == '__main__':
    main()
