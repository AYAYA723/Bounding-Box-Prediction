import argparse
import csv
import json
from pathlib import Path
import time

import torch

from predict_corner.dino_model import DinoCornerPredictor
from predict_corner.feature_dataset import DinoFeatureDataset
from predict_corner.supervision import BoxDreamerLoss, make_target_heatmaps


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=False)
    full_model = DinoCornerPredictor()
    model = full_model.decoder.cuda()
    data = DinoFeatureDataset(root / 'datasets/action4/dino_features_v1/train')
    features, target = data[0]
    features, target = features[None].cuda(), target[None].cuda()
    truth = make_target_heatmaps(target, (480, 640))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    loss_fn = BoxDreamerLoss()
    best_error = float('inf')
    best_max = float('inf')
    reached = False
    passing_metrics = None
    started = time.perf_counter()
    with (args.output / 'history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'total', 'coarse', 'fine', 'eval_mean_error_px', 'eval_max_error_px'])
        for step in range(1, args.steps + 1):
            model.train()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, xy = model(features)
                total, coarse, fine = loss_fn(prediction, xy, truth, target)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    _, xy = model(features)
                    error = (xy - target).norm(dim=-1)
                mean, maximum = error.mean().item(), error.max().item()
                writer.writerow([step, total.item(), coarse.item(), fine.item(), mean, maximum])
                stream.flush()
                print(f'step={step} mean={mean:.3f}px max={maximum:.3f}px', flush=True)
                if mean < best_error:
                    best_error, best_max = mean, maximum
                    torch.save(model.state_dict(), args.output / 'best_decoder.pt')
                if mean < 1 and maximum < 2:
                    reached = True
                    passing_metrics = {'step': step, 'mean_error_px': mean, 'max_error_px': maximum}
                    torch.save(model.state_dict(), args.output / 'passed_decoder.pt')
                    break
    report = {'steps': step, 'maximum_steps': args.steps, 'image_id': data.ids[0], 'seed': 42,
              'learning_rate': args.learning_rate,
              'mean_error_px': best_error, 'max_error_px_at_best_mean': best_max,
              'passing_metrics': passing_metrics,
              'precision': 'BF16 training, float32 evaluation and coordinate decode',
              'pass_mean_below_1px_and_max_below_2px': reached,
              'seconds': time.perf_counter() - started}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
