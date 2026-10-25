import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from predict_corner.dataset import RenderedCornerDataset
from predict_corner.dino_model import DinoCornerPredictor
from predict_corner.supervision import BoxDreamerLoss
from predict_corner.tools.train_synthetic import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.run / 'config.json').read_text())
    metadata = json.loads((Path(config['features']) / 'val/metadata.json').read_text())
    model = DinoCornerPredictor(pretrained=False).cuda()
    model.load_state_dict(torch.load(args.run / 'best.pt', map_location='cuda', weights_only=True))
    data = RenderedCornerDataset(metadata['scene_dir'])
    report = evaluate(model, DataLoader(data, batch_size=config['batch_size']), torch.device('cuda'), BoxDreamerLoss())
    cached = json.loads((args.run / 'best_validation.json').read_text())
    online_xy = torch.tensor([row['pred_xy'] for row in report['per_image']])
    cached_xy = torch.tensor([row['pred_xy'] for row in cached['per_image']])
    difference = (online_xy - cached_xy).abs().max().item()
    error_difference = abs(report['mean_corner_error_px'] - cached['mean_corner_error_px'])
    assert difference < .5
    assert error_difference < .01
    (args.run / 'online_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    check = {'status': 'PASS', 'validation_images': len(data),
             'online_mean_corner_error_px': report['mean_corner_error_px'],
             'cached_mean_corner_error_px': cached['mean_corner_error_px'],
             'maximum_coordinate_difference_px': difference, 'mean_error_difference_px': error_difference}
    (args.run / 'cache_parity.json').write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    main()
