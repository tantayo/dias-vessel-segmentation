"""Evaluate an existing DIAS VesselUNet checkpoint, without retraining.

Expected project layout:
  project/
    scripts/evaluate.py
    src/data/dataset.py
    src/models/model.py
    checkpoints/et-bce/best.pt

Run from any working directory:
  python scripts/evaluate.py --data-root data/DIAS

Outputs (under project/output/et-bce/evaluation/ by default):
  threshold_sweep.csv       Validation micro Dice, IoU, precision, recall per threshold
  per_sequence_metrics.csv  Per-sequence performance at 0.5 and best threshold
  threshold_sweep.png       Threshold curves (if matplotlib is installed)
  summary.json              Checkpoint, metrics, best threshold, split
  previews/sXX.png          Full-resolution six-panel previews
  probabilities/sXX.npy     Float32 probability arrays for later inspection

No training, augmentation, threshold tuning of weights, resizing, smoothing,
intensity transforms, or postprocessing. Uses the same DIASDataset input as
training and computes all binary metrics at original image resolution.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader

# Evaluate script lives in project/scripts/, while src/ lives at project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import DIASDataset
from src.models.model import VesselUNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=Path('data/DIAS'))
    p.add_argument('--checkpoint', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'et-bce' / 'best.pt')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--feature-mode', choices=['auto', 'raw', 'raw_et'], default='auto',
                   help='auto reads training mode from checkpoint; old checkpoints default to raw')
    p.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'output' / 'et-bce' / 'evaluation')
    p.add_argument('--min-threshold', type=float, default=0.05)
    p.add_argument('--max-threshold', type=float, default=0.95)
    p.add_argument('--threshold-step', type=float, default=0.05)
    p.add_argument('--preview-count', type=int, default=3,
                   help='Evenly spaced full-resolution previews; 0 saves all sequences')
    p.add_argument('--no-save-probs', action='store_true',
                   help='Do not save full-resolution float32 probability arrays')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--no-amp', action='store_true',
                   help='Disable CUDA FP16 autocast; enabled by default on CUDA')
    args = p.parse_args()

    if not (0 <= args.min_threshold <= args.max_threshold <= 1):
        p.error('Require 0 <= min-threshold <= max-threshold <= 1')
    if not (0 < args.threshold_step <= 1):
        p.error('Require 0 < threshold-step <= 1')
    if args.preview_count < 0 or args.num_workers < 0:
        p.error('preview-count and num-workers must be non-negative')
    if (args.max_threshold - args.min_threshold) / args.threshold_step > 1000:
        p.error('Threshold sweep would have more than 1000 steps; increase --threshold-step')
    return args


def thresholds_from_args(args):
    # Include the right endpoint when it lies on the grid. Always include 0.5
    # so the original checkpoint selection metric can be reproduced.
    n = int(np.floor((args.max_threshold - args.min_threshold) / args.threshold_step + 1e-8))
    values = args.min_threshold + args.threshold_step * np.arange(n + 1)
    values = np.concatenate((values, [0.5]))
    return np.unique(np.round(np.clip(values, 0.0, 1.0), 7))


def scores(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {
        'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
    }


def counts(prob, gt, threshold):
    pred = prob >= threshold
    return (np.count_nonzero(pred & gt),
            np.count_nonzero(pred & ~gt),
            np.count_nonzero(~pred & gt))


def make_overlay(gray, mask, color):
    """Overlay mask in one RGB color without changing any segmentation pixels."""
    out = np.repeat(gray[..., None], 3, axis=-1)
    if np.any(mask):
        color_arr = np.array(color, dtype=np.float32)
        out[mask] = np.round(0.5 * out[mask].astype(np.float32) + 0.5 * color_arr).astype(np.uint8)
    return out


def save_preview(frames, gt, prob, best_threshold, path):
    """Six panels at native H x W, with thresholding BEFORE any display operation."""
    gray = np.round(np.clip(frames[4], 0, 1) * 255).astype(np.uint8)  # Last raw input frame
    original = np.repeat(gray[..., None], 3, axis=-1)
    p05 = prob >= 0.5
    pbest = prob >= best_threshold
    gt_panel = make_overlay(gray, gt, (0, 255, 0))
    pred05_panel = make_overlay(gray, p05, (255, 0, 0))
    best_panel = make_overlay(gray, pbest, (255, 0, 0))
    probability_gray = np.round(np.clip(prob, 0, 1) * 255).astype(np.uint8)
    probability_panel = np.repeat(probability_gray[..., None], 3, axis=-1)

    # Error panel: green = TP, red = FP, blue = FN, background = original DSA.
    errors = original.copy()
    errors[pbest & gt] = (0, 255, 0)
    errors[pbest & ~gt] = (255, 0, 0)
    errors[~pbest & gt] = (0, 90, 255)

    labels = [
        'Last DSA frame', 'Ground truth (green)',
        'Prediction @ 0.50', f'Prediction @ {best_threshold:.2f}',
        'Probability (0..1)', 'Errors: TP green / FP red / FN blue',
    ]
    panels = [original, gt_panel, pred05_panel, best_panel, probability_panel, errors]
    height, width = gray.shape
    header = 32
    image = Image.new('RGB', (len(panels) * width, height + header), (12, 12, 12))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, panel) in enumerate(zip(labels, panels)):
        image.paste(Image.fromarray(panel), (i * width, header))
        draw.text((i * width + 8, 7), label, fill='white', font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows, path, best_threshold):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; threshold_sweep.csv was saved, skipping PNG plot.')
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = [row['threshold'] for row in rows]
    for key in ('dice', 'precision', 'recall', 'iou'):
        ax.plot(x, [row[key] for row in rows], marker='.', label=key.capitalize())
    ax.axvline(0.5, color='gray', linestyle=':', label='Original threshold = 0.50')
    ax.axvline(best_threshold, color='black', linestyle='--', label=f'Best Dice threshold = {best_threshold:.2f}')
    ax.set(xlabel='Probability threshold', ylabel='Micro score',
           title='DIAS segmentation: validation threshold sweep', ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    use_amp = device.type == 'cuda' and not args.no_amp

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    cfg = checkpoint.get('config', {}) if 'model_state_dict' in checkpoint else {}
    feature_mode = cfg.get('feature_mode', 'raw') if args.feature_mode == 'auto' else args.feature_mode
    if feature_mode not in ('raw', 'raw_et'):
        raise ValueError(f'Invalid checkpoint feature_mode: {feature_mode}')
    dataset = DIASDataset(args.data_root, split=args.split, augment=False,
                          feature_mode=feature_mode)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model = VesselUNet(in_channels=dataset.in_channels).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    thresholds = thresholds_from_args(args)
    total = np.zeros((len(thresholds), 3), dtype=np.int64)  # TP, FP, FN
    samples = []  # Retain CPU predictions for previews and per-sequence scores.
    prob_dir = args.output_dir / 'probabilities'
    if not args.no_save_probs:
        prob_dir.mkdir(parents=True, exist_ok=True)
    print(f'Checkpoint: {args.checkpoint} | feature_mode: {feature_mode}')
    print(f'Device: {device} | AMP: {use_amp} | {args.split} sequences: {len(dataset)}')
    print(f'Running inference once, evaluating {len(thresholds)} thresholds...')

    for index, (frames, mask) in enumerate(loader):
        seq_id = dataset.samples[index][0]
        frames_gpu = frames.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(frames_gpu)
        # Keep full precision probabilities for evaluation and saved arrays.
        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy().copy()
        gt = (mask[0, 0].numpy() >= 0.5)
        last_five = frames[0].numpy()

        if not args.no_save_probs:
            np.save(prob_dir / f's{seq_id}.npy', prob)
        samples.append({'id': seq_id, 'frames': last_five, 'mask': gt, 'prob': prob})
        for i, threshold in enumerate(thresholds):
            total[i] += counts(prob, gt, threshold)
        print(f'  [{index + 1:>2}/{len(dataset)}] sequence s{seq_id}', flush=True)

    rows = []
    for threshold, (tp, fp, fn) in zip(thresholds, total):
        rows.append({'threshold': float(threshold), **scores(tp, fp, fn),
                     'tp': int(tp), 'fp': int(fp), 'fn': int(fn)})
    best_idx = max(range(len(rows)),
                   key=lambda i: (rows[i]['dice'], -abs(rows[i]['threshold'] - 0.5)))
    best = rows[best_idx]
    base = next(row for row in rows if row['threshold'] == 0.5)
    best_threshold = best['threshold']

    # Save full-resolution native-grid previews, not bilinearly resized probability maps.
    if args.preview_count == 0:
        chosen = range(len(samples))
    else:
        chosen = np.unique(np.linspace(0, len(samples) - 1,
                                       num=min(args.preview_count, len(samples)),
                                       dtype=int))
    preview_dir = args.output_dir / 'previews'
    for index in chosen:
        item = samples[index]
        save_preview(item['frames'], item['mask'], item['prob'], best_threshold,
                     preview_dir / f"s{item['id']}.png")

    per_sequence = []
    for item in samples:
        tp0, fp0, fn0 = counts(item['prob'], item['mask'], 0.5)
        tp1, fp1, fn1 = counts(item['prob'], item['mask'], best_threshold)
        base_scores = scores(tp0, fp0, fn0)
        best_scores = scores(tp1, fp1, fn1)
        per_sequence.append({'sequence': f"s{item['id']}",
                             **{f'{k}_at_0.5': v for k, v in base_scores.items()},
                             **{f'{k}_at_best': v for k, v in best_scores.items()}})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'threshold_sweep.csv', rows,
              ['threshold', 'dice', 'iou', 'precision', 'recall', 'tp', 'fp', 'fn'])
    write_csv(args.output_dir / 'per_sequence_metrics.csv', per_sequence,
              ['sequence'] + [f'{k}_at_{s}' for s in ('0.5', 'best')
                              for k in ('dice', 'iou', 'precision', 'recall')])
    plot_sweep(rows, args.output_dir / 'threshold_sweep.png', best_threshold)

    summary = {
        'checkpoint': str(args.checkpoint), 'checkpoint_epoch': checkpoint.get('epoch') if 'model_state_dict' in checkpoint else None,
        'checkpoint_recorded_best_dice_at_0.5': checkpoint.get('best_val_dice') if 'model_state_dict' in checkpoint else None,
        'split': args.split, 'sequences': len(dataset), 'device': str(device),
        'feature_mode': feature_mode,
        'amp': use_amp, 'metrics': 'dataset-level micro over all segmentation pixels',
        'best_threshold_by_dice': best_threshold, 'original_threshold_metrics': base,
        'best_threshold_metrics': best,
        'note': 'Threshold chosen on this split; for unbiased assessment evaluate a held-out test set with a previously fixed threshold.',
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    print('\nOriginal threshold 0.50:',
          f'Dice {base["dice"]:.4f}, precision {base["precision"]:.4f}, recall {base["recall"]:.4f}')
    print(f'Best threshold {best_threshold:.2f}:',
          f'Dice {best["dice"]:.4f}, precision {best["precision"]:.4f}, recall {best["recall"]:.4f}')
    if args.split == 'validation' and summary['checkpoint_recorded_best_dice_at_0.5'] is not None:
        expected = float(summary['checkpoint_recorded_best_dice_at_0.5'])
        if abs(base['dice'] - expected) > 0.001:
            print(f'WARNING: Dice at 0.5 differs from checkpoint value ({expected:.4f}); '
                  'check dataset version, split, AMP, and checkpoint.')
    print(f'Outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
