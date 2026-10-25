import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from predict_corner.dataset import CornerDataset
from predict_corner.dino_model import DinoCornerPredictor
from predict_corner.feature_dataset import DinoFeatureDataset
from predict_corner.supervision import BoxDreamerLoss, make_target_heatmaps
from predict_corner.tools.evaluate import evaluate as evaluate_original
from predict_corner.tools.train_synthetic import evaluate, save_previews


def encoder_digest(encoder):
    digest = hashlib.sha256()
    for name, value in encoder.state_dict().items():
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, default=root / 'datasets/action4/dino_features_v1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=False)
    train_set = DinoFeatureDataset(args.features / 'train')
    val_set = DinoFeatureDataset(args.features / 'val')
    shuffle_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              generator=shuffle_generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    full_model = DinoCornerPredictor(pretrained=args.resume is None)
    previous_epochs = 0
    resume_state = None
    if args.resume:
        full_model.load_state_dict(torch.load(args.resume / 'last.pt', map_location='cpu', weights_only=True))
        resume_state = torch.load(args.resume / 'training_state.pt', map_location='cpu', weights_only=True)
        previous_epochs = resume_state['epoch']
        shuffle_generator.set_state(resume_state['loader_rng_state'])
    initial_encoder_hash = encoder_digest(full_model.encoder)
    model = full_model.decoder.cuda()
    device = torch.device('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
            group['initial_lr'] = args.learning_rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = BoxDreamerLoss()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'architecture': 'frozen DINOv2 ViT-B/14 reg + 12-layer 768-dim 8-head Transformer + Linear/unpatchify',
                   'encoder': 'dinov2_vitb14_reg', 'encoder_frozen': True,
                   'decoder_initialization': 'restored last checkpoint' if args.resume else 'random',
                   'previous_epochs': previous_epochs,
                   'optimizer_initialization': 'restored AdamW moments' if args.resume else 'new AdamW',
                   'scheduler_initialization': 'new cosine phase for the additional epochs',
                   'boxdreamer_task_weights_loaded': False, 'train_images': len(train_set), 'validation_images': len(val_set),
                   'seed': 42, 'loss': 'coarse + 2*fine', 'temperature': .25,
                   'image_size': [480, 640], 'padded_image_size': [490, 644], 'patch_grid': [35, 46],
                   'decoder_layers': 12, 'hidden_dim': 768, 'num_heads': 8, 'patch_size': 14,
                   'precision': 'float32 frozen features, BF16 decoder training, float32 evaluation and coordinates',
                   'trainable_parameters': sum(p.numel() for p in model.parameters()),
                   'encoder_initial_sha256': initial_encoder_hash,
                   'encoder_weights_sha256': hashlib.sha256((root / '.cache/torch/checkpoints/dinov2_vitb14_reg4_pretrain.pth').read_bytes()).hexdigest(),
                   'started_at': datetime.now().astimezone().isoformat(),
                   'source_sha256': {name: hashlib.sha256((root / 'predict_corner' / name).read_bytes()).hexdigest()
                                     for name in ['dino_model.py', 'heatmap_decoder.py', 'feature_dataset.py',
                                                  'supervision.py', 'tools/train_dino.py', 'tools/train_synthetic.py']}})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    baseline = evaluate(model, val_loader, device, loss_fn)
    (args.output / 'baseline_validation.json').write_text(json.dumps(baseline, indent=2) + '\n')
    if resume_state is not None:
        torch.set_rng_state(resume_state['rng_state'])
        torch.cuda.set_rng_state(resume_state['cuda_rng_state'])
    print(f"BASELINE validation={baseline['mean_corner_error_px']:.3f}px", flush=True)
    best_error, best_epoch = float('inf'), 0
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    fields = ['epoch', 'learning_rate', 'total', 'coarse', 'fine', 'train_error_px',
              'validation_error_px', 'validation_5px_percent', 'validation_10px_percent', 'seconds']
    with (args.output / 'history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for phase_epoch in range(1, args.epochs + 1):
            epoch = previous_epochs + phase_epoch
            model.train()
            sums = [0., 0., 0., 0.]
            count = 0
            for index, (features, target) in enumerate(train_loader):
                features, target = features.cuda(), target.cuda()
                truth = make_target_heatmaps(target, (480, 640))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction, xy = model(features)
                    total, coarse, fine = loss_fn(prediction, xy, truth, target)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                N = len(features)
                values = [total.item(), coarse.item(), fine.item(), (xy.detach() - target).norm(dim=-1).mean().item()]
                for i, value in enumerate(values):
                    sums[i] += value * N
                count += N
                if (index + 1) % 25 == 0:
                    print(f'epoch={epoch} batch={index + 1}/{len(train_loader)} loss={total.item():.5f}', flush=True)
            validation = evaluate(model, val_loader, device, loss_fn)
            learning_rate = optimizer.param_groups[0]['lr']
            scheduler.step()
            row = [epoch, learning_rate, *(value / count for value in sums), validation['mean_corner_error_px'],
                   validation['within_5px_percent'], validation['within_10px_percent'], time.perf_counter() - started]
            writer.writerow(row)
            stream.flush()
            history.append(dict(zip(fields, row)))
            if validation['mean_corner_error_px'] < best_error:
                best_error, best_epoch = validation['mean_corner_error_px'], epoch
                torch.save(full_model.state_dict(), args.output / 'best.pt')
                (args.output / 'best_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
            print(f'EPOCH {epoch}/{previous_epochs + args.epochs} train={sums[3] / count:.2f}px validation={validation["mean_corner_error_px"]:.2f}px '
                  f'best={best_error:.2f}px elapsed={time.perf_counter() - started:.1f}s', flush=True)
            if epoch - best_epoch >= args.patience:
                break
    torch.save(full_model.state_dict(), args.output / 'last.pt')
    torch.save({'decoder': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(),
                'loader_rng_state': shuffle_generator.get_state()},
               args.output / 'training_state.pt')
    optimizer.zero_grad(set_to_none=True)
    full_model.load_state_dict(torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True))
    final_encoder_hash = encoder_digest(full_model.encoder)
    assert initial_encoder_hash == final_encoder_hash
    final_metrics = evaluate(model, val_loader, device, loss_fn)
    save_previews(final_metrics, val_set, args.output / 'validation_preview')
    sample = dict(final_metrics)
    rows = final_metrics['per_image']
    sample['per_image'] = [rows[round(i * (len(rows) - 1) / 23)] for i in range(24)]
    save_previews(sample, val_set, args.output / 'validation_all_views')
    train_metrics = evaluate(model, DataLoader(train_set, batch_size=args.batch_size), device, loss_fn)
    (args.output / 'train_evaluation.json').write_text(json.dumps(train_metrics, indent=2) + '\n')
    old_data = CornerDataset(root / 'datasets/action4/train_pbr/000000')
    old_indices = torch.randperm(len(old_data), generator=torch.Generator().manual_seed(42)).tolist()[16:]
    original_metrics = evaluate_original(full_model.cuda(), old_data, old_indices, device)
    (args.output / 'original_holdout.json').write_text(json.dumps(original_metrics, indent=2) + '\n')
    summary = {'best_epoch': best_epoch, 'epochs_completed': epoch, 'seconds': time.perf_counter() - started,
               'previous_epochs': previous_epochs, 'phase_epochs_completed': phase_epoch,
               'validation': final_metrics, 'train_eval_mean_corner_error_px': train_metrics['mean_corner_error_px'],
               'original_holdout_mean_corner_error_px': original_metrics['mean_corner_error_px'],
               'encoder_initial_sha256': initial_encoder_hash, 'encoder_final_sha256': final_encoder_hash,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in ['total', 'coarse', 'fine']:
        axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    for key in ['train_error_px', 'validation_error_px']:
        axes[1].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    axes[1].axhline(47.77705, color='grey', linestyle='--', label='ResNet previous best')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.legend()
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output / 'training_curves.png', dpi=160)
    plt.close(fig)
    print(f'COMPLETE best_epoch={best_epoch}, validation={best_error:.3f}px', flush=True)


if __name__ == '__main__':
    main()
