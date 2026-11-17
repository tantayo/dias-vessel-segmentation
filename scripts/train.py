"""Train a small VesselUNet on DIAS sequences.

Use --feature-mode raw (five frames) or raw_et (five frames plus whole-sequence
spatial and temporal evidence). Default objective is BCE + soft Dice; optional
--cldice-weight adds a hybrid skeleton-based topology term.

Training artifacts are saved under runs/et-bce, checkpoints/et-bce and
output/et-bce by default. Override these directory flags for other experiments.
"""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from skimage.morphology import skeletonize
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

# scripts/train.py lives one level below the project src/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import DIASDataset
from src.models.model import VesselUNet


def parse_args():
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=Path('data/DIAS'))
    p.add_argument('--feature-mode', choices=['raw', 'raw_et'], default='raw_et',
                   help='raw: original 5 frames; raw_et: append quicklook E and T (7 channels)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=3,
                   help='Conservative default for GPUs with ~6 GB VRAM')
    p.add_argument('--accum-steps', type=int, default=1,
                   help='Gradient accumulation; OneCycleLR advances per optimizer update')
    p.add_argument('--max-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=0,
                   help='0 avoids extra worker processes and RAM usage')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-augment', action='store_true', help='Disable dataset RandomResizedCrop')
    p.add_argument('--no-amp', action='store_true', help='Disable CUDA mixed precision')
    p.add_argument('--grad-clip', type=float, default=1.0,
                   help='Gradient-norm clipping; 0 disables it')
    p.add_argument('--cldice-weight', type=float, default=0.0,
                   help='Weight of hybrid clDice loss; 0 turns it off')
    p.add_argument('--cldice-iters', type=int, default=10,
                   help='Differentiable skeletonization iterations (more = slower/more coverage)')
    p.add_argument('--no-cldice-checkpoint', action='store_true',
                   help='Disable activation checkpointing for predicted skeletonization (faster, more VRAM)')
    p.add_argument('--preview-count', type=int, default=3,
                   help='Fixed validation sequences logged at every epoch')
    p.add_argument('--preview-size', type=int, default=320,
                   help='Max display width of each preview panel (does not alter input)')
    p.add_argument('--runs-dir', type=Path, default=here / 'runs' / 'et-bce')
    p.add_argument('--checkpoints-dir', type=Path, default=here / 'checkpoints' / 'et-bce')
    p.add_argument('--output-dir', type=Path, default=here / 'output' / 'et-bce')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1 or args.num_workers < 0:
        p.error('epochs, batch-size and accum-steps must be >=1; num-workers must be >=0')
    if args.max_lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0 or args.preview_size < 16:
        p.error('Invalid optimizer or preview settings')
    if args.preview_count < 0:
        p.error('preview-count must be >=0')
    if args.cldice_weight < 0 or args.cldice_iters < 0:
        p.error('cldice-weight and cldice-iters must be >=0')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_skeletons(masks_cpu, device):
    """Generate GT skeletons after dataset augmentation."""
    skeletons = np.stack([
        skeletonize(mask[0].numpy() > 0.5)
        for mask in masks_cpu
    ])

    return torch.from_numpy(
        skeletons[:, None].astype(np.float32)
    ).to(device)


def soft_erode(x):
    """2D min-pooling along the vertical and horizontal directions."""
    vertical = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_open(x):
    """Differentiable approximation to morphological opening."""
    return F.max_pool2d(soft_erode(x), kernel_size=3, stride=1, padding=1)


def soft_skeleton(x, iterations):
    """Approximate predicted centerline using iterative min/max pooling.

    This is based on the clDice soft-skeletonization algorithm. Its finite
    number of iterations limits its ability to skeletonize thick structures.
    """
    skeleton = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def hybrid_cldice_loss(probabilities, targets, gt_skeletons, iterations=10,
                       checkpoint_skeleton=True, eps=1e-6):
    """Two-sided clDice approximation with a hard GT and soft predicted skeleton.

    Topological precision: fraction of predicted soft skeleton supported by GT.
    Topological sensitivity: fraction of GT skeleton covered by probabilities.
    Computed per sample before averaging to limit dominance by long vessel trees.
    """
    probabilities = probabilities.float()
    targets = targets.float()
    gt_skeletons = gt_skeletons.float()

    # Graph checkpointing recomputes morphology during backward to save VRAM.
    if checkpoint_skeleton and torch.is_grad_enabled() and probabilities.requires_grad:
        predicted_skeleton = checkpoint(
            soft_skeleton, probabilities, iterations, use_reentrant=False
        )
    else:
        predicted_skeleton = soft_skeleton(probabilities, iterations)

    spatial = (1, 2, 3)
    topo_precision = (
        (predicted_skeleton * targets).sum(spatial) + eps
    ) / (predicted_skeleton.sum(spatial) + eps)
    topo_sensitivity = (
        (gt_skeletons * probabilities).sum(spatial) + eps
    ) / (gt_skeletons.sum(spatial) + eps)
    cldice = (
        2.0 * topo_precision * topo_sensitivity + eps
    ) / (topo_precision + topo_sensitivity + eps)
    return 1.0 - cldice.mean(), topo_precision.mean(), topo_sensitivity.mean()


def compute_losses(logits, targets, gt_skeletons, bce_loss,
                   cldice_weight, cldice_iters, checkpoint_skeleton):
    """BCE + soft Dice + weighted hybrid clDice, over the current batch."""
    logits = logits.float()
    targets = targets.float()
    bce = bce_loss(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum()
    denominator = probabilities.sum() + targets.sum()
    soft_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - soft_dice

    if cldice_weight > 0:
        cldice_loss, topo_prec, topo_sens = hybrid_cldice_loss(
            probabilities, targets, gt_skeletons,
            iterations=cldice_iters,
            checkpoint_skeleton=checkpoint_skeleton,
        )
    else:
        # Allows direct BCE + Dice baseline with --cldice-weight 0.
        cldice_loss = logits.new_zeros(())
        topo_prec = logits.new_zeros(())
        topo_sens = logits.new_zeros(())
    total = bce + dice_loss + cldice_weight * cldice_loss
    return (total, bce.detach(), dice_loss.detach(), cldice_loss.detach(),
            topo_prec.detach(), topo_sens.detach())


def metric_scores(tp, fp, fn):
    """Dataset-level (micro) metrics over all validation pixels."""
    return {
        'dice': (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        'iou': (tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
        'precision': (tp / (tp + fp)) if (tp + fp) else 0.0,
        'recall': (tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def make_preview(frames, mask, probability, max_size=320):
    """Four columns: final DSA frame | GT green | predicted red | probability.

    Downsampling occurs ONLY for TensorBoard/PNG visualization, not training or evaluation.
    """
    import torch.nn.functional as F

    frame = frames[4:5].detach().float().cpu().unsqueeze(0)  # Last raw frame; E and T follow
    gt = mask.detach().float().cpu().reshape(1, 1, *mask.shape[-2:])
    prob = probability.detach().float().cpu().reshape(1, 1, *probability.shape[-2:])
    h, w = frame.shape[-2:]
    if max(h, w) > max_size:
        new_size = (max(16, round(h * max_size / max(h, w))),
                    max(16, round(w * max_size / max(h, w))))
        frame = F.interpolate(frame, size=new_size, mode='bilinear', align_corners=False)
        gt = F.interpolate(gt, size=new_size, mode='nearest')
        prob = F.interpolate(prob, size=new_size, mode='bilinear', align_corners=False)

    base = frame[0].clamp(0, 1).expand(3, -1, -1)
    gt_on = base.clone()
    pred_on = base.clone()
    gt_pixels = gt[0, 0] > 0.5
    pred_pixels = prob[0, 0] >= 0.5
    # Blends preserve anatomical context while highlighting foreground.
    gt_on[:, gt_pixels] = gt_on[:, gt_pixels] * 0.5 + torch.tensor([0.0, 0.5, 0.0]).view(3, 1)
    pred_on[:, pred_pixels] = pred_on[:, pred_pixels] * 0.5 + torch.tensor([0.5, 0.0, 0.0]).view(3, 1)
    prob_rgb = prob[0].expand(3, -1, -1).clamp(0, 1)
    return torch.cat([base, gt_on, pred_on, prob_rgb], dim=2).clamp(0, 1)


def train_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device,
                use_amp, accum_steps, grad_clip, cldice_weight, cldice_iters, checkpoint_skeleton):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {'loss': 0.0, 'bce': 0.0, 'dice_loss': 0.0, 'cldice': 0.0,
              'topo_precision': 0.0, 'topo_sensitivity': 0.0}
    updates = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    clipped_updates = 0
    tp = fp = fn = 0

    progress = tqdm(loader, desc="Training", leave=False)
    samples_seen = 0

    for batch_idx, (images, masks) in enumerate(progress):
        skeletons = make_skeletons(masks, device)

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # For a partial final accumulation group, divide by its actual batch count.
        group_start = (batch_idx // accum_steps) * accum_steps
        group_size = min(accum_steps, len(loader) - group_start)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss, bce, dice_loss, cl_loss, tprec, tsens = compute_losses(
                logits, masks, skeletons, loss_fn,
                cldice_weight, cldice_iters, checkpoint_skeleton
            )

        scaler.scale(loss / group_size).backward()
        with torch.no_grad():
            pred = torch.sigmoid(logits.float()) >= 0.5
            gt = masks >= 0.5
            tp += (pred & gt).sum().item()
            fp += (pred & ~gt).sum().item()
            fn += (~pred & gt).sum().item()
        n = images.shape[0]
        samples_seen += n
        totals['loss'] += loss.detach().item() * n
        totals['bce'] += bce.item() * n
        totals['dice_loss'] += dice_loss.item() * n
        totals['cldice'] += cl_loss.item() * n
        totals['topo_precision'] += tprec.item() * n
        totals['topo_sensitivity'] += tsens.item() * n

        progress.set_postfix(
            loss=f"{totals['loss'] / samples_seen:.4f}"
        )

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            # Unscale gradients before measuring or clipping.
            scaler.unscale_(optimizer)

            # Returns the total gradient norm BEFORE clipping.
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip if grad_clip > 0 else float('inf')
            )

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP may skip optimizer.step on inf/nan gradients. Don't advance LR then.
            did_step = (not scaler.is_enabled()) or (scaler.get_scale() >= old_scale)
            if did_step:
                norm = grad_norm.item()

                if math.isfinite(norm):
                    grad_norm_sum += norm
                    grad_norm_count += 1

                    if grad_clip > 0 and norm > grad_clip:
                        clipped_updates += 1

                scheduler.step()
                updates += 1
            optimizer.zero_grad(set_to_none=True)

    for key in totals:
        totals[key] /= len(loader.dataset)
    totals['updates'] = updates
    totals.update(metric_scores(tp, fp, fn))
    totals['grad_norm'] = (
        grad_norm_sum / grad_norm_count
        if grad_norm_count else float('nan')
    )

    totals['clip_fraction'] = (
        clipped_updates / grad_norm_count
        if grad_norm_count else 0.0
    )
    return totals


@torch.inference_mode()
def validate(model, loader, loss_fn, device, use_amp,
             selected_indices, preview_size, cldice_weight, cldice_iters):
    model.eval()
    totals = {'loss': 0.0, 'bce': 0.0, 'dice_loss': 0.0, 'cldice': 0.0,
              'topo_precision': 0.0, 'topo_sensitivity': 0.0}
    tp = fp = fn = 0
    previews = {}

    # Fixed batch_size=1 so dataset index corresponds to each validation sample.
    progress = tqdm(loader, desc="Validation", leave=False)

    for index, (images, masks) in enumerate(progress):
        skeletons = make_skeletons(masks, device)

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss, bce, dice_loss, cl_loss, tprec, tsens = compute_losses(
                logits, masks, skeletons, loss_fn,
                cldice_weight, cldice_iters, checkpoint_skeleton=False
            )

        p = torch.sigmoid(logits.float())
        pred = p >= 0.5
        gt = masks >= 0.5
        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()
        totals['loss'] += loss.item()
        totals['bce'] += bce.item()
        totals['dice_loss'] += dice_loss.item()
        totals['cldice'] += cl_loss.item()
        totals['topo_precision'] += tprec.item()
        totals['topo_sensitivity'] += tsens.item()

        progress.set_postfix(
            loss=f"{totals['loss'] / (index + 1):.4f}"
        )

        if index in selected_indices:
            seq_id = loader.dataset.samples[index][0]
            previews[f's{seq_id}'] = make_preview(
                images[0], masks[0], p[0], max_size=preview_size)

    for key in totals:
        totals[key] /= len(loader.dataset)
    totals.update(metric_scores(tp, fp, fn))
    return totals, previews


def save_best_preview_images(previews, output_dir):
    directory = output_dir / 'best_validation'
    directory.mkdir(parents=True, exist_ok=True)
    for sequence, rgb in previews.items():
        pixels = (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels, 'RGB').save(directory / f'{sequence}.png')


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not args.no_amp

    train_data = DIASDataset(args.data_root, split='training', augment=not args.no_augment,
                             feature_mode=args.feature_mode)
    val_data = DIASDataset(args.data_root, split='validation', augment=False,
                           feature_mode=args.feature_mode)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = VesselUNet(in_channels=train_data.in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                                  weight_decay=args.weight_decay)
    total_updates = math.ceil(len(train_loader) / args.accum_steps) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=total_updates,
        pct_start=0.1, anneal_strategy='cos', cycle_momentum=False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()

    run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = args.runs_dir / run_name
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    indices = set(np.linspace(0, len(val_data) - 1,
                              num=min(args.preview_count, len(val_data)), dtype=int).tolist())
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg.update(device=str(device), amp=use_amp,
               train_sequences=len(train_data), val_sequences=len(val_data),
               onecycle_total_updates=total_updates,
               loss=(f'BCEWithLogits + soft Dice + {args.cldice_weight} * hybrid clDice '
                     f'(soft predicted skeleton: {args.cldice_iters} iterations, hard GT skeleton)'),
               validation_threshold=0.5)
    (args.output_dir / 'run_config.json').write_text(json.dumps(cfg, indent=2))
    writer.add_text('config', json.dumps(cfg, indent=2).replace('\n', '  \n'))

    print(f'Device: {device} | CUDA AMP: {use_amp}')
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(device)
        print(f'GPU: {prop.name} | VRAM: {prop.total_memory / 2**30:.2f} GiB')
    print(f'Train / validation sequences: {len(train_data)} / {len(val_data)}')
    print(f'Feature mode: {args.feature_mode} | input channels: {train_data.in_channels}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Batch size: {args.batch_size} | accumulation: {args.accum_steps} | '
          f'optimizer updates planned: {total_updates}')
    print(f'TensorBoard: {log_dir}')

    columns = ['epoch', 'train_loss', 'train_bce', 'train_dice_loss', 'train_cldice',
               'train_topo_precision', 'train_topo_sensitivity',
               'train_dice', 'train_iou', 'train_precision', 'train_recall',
               'val_loss', 'val_bce', 'val_dice_loss', 'val_cldice',
               'val_topo_precision', 'val_topo_sensitivity', 'val_dice', 'val_iou',
               'val_precision', 'val_recall', 'lr', 'optimizer_updates', 'grad_norm', 'clip_fraction',]
    history_file = args.output_dir / 'history.csv'
    best_dice = -1.0
    with history_file.open('w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=columns)
        csv_writer.writeheader()
        try:
            for epoch in range(1, args.epochs + 1):
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                train = train_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn,
                                    device, use_amp, args.accum_steps, args.grad_clip,
                                    args.cldice_weight, args.cldice_iters,
                                    not args.no_cldice_checkpoint)
                val, previews = validate(model, val_loader, loss_fn, device, use_amp,
                                         indices, args.preview_size, args.cldice_weight, args.cldice_iters)
                lr = optimizer.param_groups[0]['lr']
                for name in ('loss', 'bce', 'dice_loss', 'cldice', 'topo_precision', 'topo_sensitivity'):
                    writer.add_scalar(f'train/{name}', train[name], epoch)
                    writer.add_scalar(f'validation/{name}', val[name], epoch)
                for name in ('dice', 'iou', 'precision', 'recall'):
                    writer.add_scalar(f'train/{name}', train[name], epoch)
                    writer.add_scalar(f'validation/{name}', val[name], epoch)
                writer.add_scalar('train/lr', lr, epoch)
                writer.add_scalar('train/grad_norm', train['grad_norm'], epoch)
                writer.add_scalar('train/clip_fraction', train['clip_fraction'], epoch)
                if device.type == 'cuda':
                    allocated = torch.cuda.max_memory_allocated() / 2**30
                    reserved = torch.cuda.max_memory_reserved() / 2**30
                    writer.add_scalar('memory/peak_allocated_GiB', allocated, epoch)
                    writer.add_scalar('memory/peak_reserved_GiB', reserved, epoch)
                for sequence, rgb in previews.items():
                    writer.add_image(f'validation_preview/{sequence}', rgb, epoch,
                                     dataformats='CHW')

                row = {
                    'epoch': epoch,
                    'train_loss': train['loss'], 'train_bce': train['bce'],
                    'train_dice_loss': train['dice_loss'], 'train_cldice': train['cldice'],
                    'train_topo_precision': train['topo_precision'],
                    'train_topo_sensitivity': train['topo_sensitivity'],
                    'train_dice': train['dice'], 'train_iou': train['iou'],
                    'train_precision': train['precision'], 'train_recall': train['recall'],
                    'val_loss': val['loss'], 'val_bce': val['bce'],
                    'val_dice_loss': val['dice_loss'], 'val_cldice': val['cldice'],
                    'val_topo_precision': val['topo_precision'],
                    'val_topo_sensitivity': val['topo_sensitivity'],
                    'val_dice': val['dice'], 'val_iou': val['iou'],
                    'val_precision': val['precision'], 'val_recall': val['recall'],
                    'lr': lr, 'optimizer_updates': train['updates'],
                    'grad_norm': train['grad_norm'],
                    'clip_fraction': train['clip_fraction'],
                }
                csv_writer.writerow(row)
                csvfile.flush()

                is_best = val['dice'] > best_dice
                if is_best:
                    best_dice = val['dice']
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch, 'best_val_dice': best_dice,
                        'validation_metrics': val,
                        'config': cfg,
                    }
                    torch.save(checkpoint, args.checkpoints_dir / 'best.pt')
                    (args.output_dir / 'best_metrics.json').write_text(json.dumps(
                        {'epoch': epoch, **val}, indent=2))
                    save_best_preview_images(previews, args.output_dir)

                suffix = '  *best*' if is_best else ''
                ram = (f' | peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB'
                       if device.type == 'cuda' else '')
                print(f'Epoch {epoch:3d}/{args.epochs} | train {train["loss"]:.4f} (Dice {train["dice"]:.4f}) | '
                      f'val {val["loss"]:.4f} (Dice {val["dice"]:.4f}) | '
                      f'IoU {val["iou"]:.4f} | lr {lr:.2e}{ram}{suffix}', flush=True)
                writer.flush()
        finally:
            writer.close()
    print(f'Best validation Dice: {best_dice:.4f} | checkpoint: {args.checkpoints_dir / "best.pt"}')
    print(f'History/visuals: {args.output_dir}')


if __name__ == '__main__':
    main()
