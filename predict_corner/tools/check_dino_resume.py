import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from predict_corner.dino_model import DinoCornerPredictor
from predict_corner.feature_dataset import DinoFeatureDataset
from predict_corner.supervision import BoxDreamerLoss, make_target_heatmaps
from predict_corner.tools.train_dino import encoder_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.run / 'config.json').read_text())
    full_model = DinoCornerPredictor(pretrained=False)
    full_model.load_state_dict(torch.load(args.run / 'last.pt', map_location='cpu', weights_only=True))
    state = torch.load(args.run / 'training_state.pt', map_location='cpu', weights_only=True)
    for name, value in full_model.decoder.state_dict().items():
        torch.testing.assert_close(value, state['decoder'][name], rtol=0, atol=0)
    initial_encoder = encoder_digest(full_model.encoder)
    model = full_model.decoder.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    optimizer.load_state_dict(state['optimizer'])
    parameter = model.heatmap_projection.weight
    before = parameter.detach().clone()
    previous_step = optimizer.state[parameter]['step'].item()
    generator = torch.Generator()
    generator.set_state(state['loader_rng_state'])
    data = DinoFeatureDataset(Path(config['features']) / 'train')
    features, target = next(iter(DataLoader(data, batch_size=config['batch_size'], shuffle=True, generator=generator)))
    features, target = features.cuda(), target.cuda()
    torch.set_rng_state(state['rng_state'])
    torch.cuda.set_rng_state(state['cuda_rng_state'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        heatmaps, xy = model(features)
        total, _, _ = BoxDreamerLoss()(heatmaps, xy, make_target_heatmaps(target, (480, 640)), target)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)
    assert not torch.equal(parameter, before)
    assert optimizer.state[parameter]['step'].item() == previous_step + 1
    assert encoder_digest(full_model.encoder) == initial_encoder
    report = {'status': 'PASS', 'weights_match_optimizer_checkpoint': True,
              'saved_epoch': state['epoch'], 'optimizer_step_before': previous_step,
              'optimizer_step_after': optimizer.state[parameter]['step'].item(),
              'decoder_weight_max_change': (parameter - before).abs().max().item(),
              'encoder_unchanged': True, 'original_checkpoint_files_modified': False,
              'loss_after_loading': total.item()}
    (args.run / 'resume_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
