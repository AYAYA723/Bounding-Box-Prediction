import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from predict_corner.dataset import RenderedCornerDataset
from predict_corner.dino_model import DinoCornerPredictor
from predict_corner.supervision import BoxDreamerLoss, make_target_heatmaps


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data = RenderedCornerDataset(root / 'datasets/action4/synthetic_v2/train')
    x, target = next(iter(DataLoader(data, batch_size=args.batch_size)))
    x, target = x.cuda(), target.cuda()
    model = DinoCornerPredictor().cuda()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    features = model.encode(x)
    torch.cuda.synchronize()
    encoding_seconds = time.perf_counter() - started
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4)
    loss_fn = BoxDreamerLoss()
    truth = make_target_heatmaps(target, x.shape[-2:])
    times = []
    for step in range(4):
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            heatmaps, xy = model.decoder(features)
            total, coarse, fine = loss_fn(heatmaps, xy, truth, target)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        assert torch.isfinite(total)
        assert torch.isfinite(model.decoder.heatmap_projection.weight.grad).all()
        assert model.decoder.transformer[0].qkv.weight.grad.norm() > 0
        assert all(p.grad is None for p in model.encoder.parameters())
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    model.eval()
    with torch.no_grad():
        online = model(x)
        cached = model.decoder(features)
    torch.testing.assert_close(online, cached, atol=.002, rtol=1e-5)
    report = {'status': 'PASS', 'input_shape': list(x.shape), 'feature_shape': list(features.shape),
              'heatmap_shape': list(heatmaps.shape), 'coordinate_shape': list(xy.shape),
              'encoder_training': model.encoder.training, 'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
              'batch_size': args.batch_size, 'float32_encoding_seconds': encoding_seconds,
              'decoder_update_seconds': times, 'bf16_training': True,
              'online_cached_max_coordinate_difference_px': (online[1] - cached[1]).abs().max().item()}
    output = root / 'results/dino_transformer_checks'
    output.mkdir(exist_ok=True)
    (output / f'model_check_batch{args.batch_size}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
