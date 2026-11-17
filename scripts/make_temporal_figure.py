#!/usr/bin/env python3
"""Create the README's E/T diagnostic figure from the local DIAS dataset.

    python scripts/make_temporal_figure.py --data-root /path/to/DIAS --sequence 39

Only visualization contrast is adjusted: images fed to the model are unchanged.
"""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import DIASDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=Path('data/DIAS'))
    ap.add_argument('--sequence', type=int, default=39)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--output', type=Path,
                    default=PROJECT_ROOT / 'assets' / 'temporal_features_s39.png')
    args = ap.parse_args()

    dataset = DIASDataset(args.data_root, split=args.split,
                          augment=False, feature_mode='raw_et')
    idx = next((i for i, sample in enumerate(dataset.samples)
                if sample[0] == args.sequence), None)
    if idx is None:
        raise ValueError(f'Sequence s{args.sequence} not found in {args.split}')

    x, mask = dataset[idx]
    x, mask = x.numpy(), mask[0].numpy() > 0.5
    last_frame, E, T = x[4], x[5], x[6]

    gt_overlay = np.repeat(last_frame[..., None], 3, axis=-1).copy()
    gt_overlay[mask] = .55 * gt_overlay[mask] + .45 * np.array([0., 1., 0.])

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    ax[0].imshow(last_frame, cmap='gray', vmin=0, vmax=1)
    for a, arr, title in ((ax[1], E, 'E: max spatial contrast'),
                          (ax[2], T, 'T: max temporal darkening')):
        high = max(float(np.percentile(arr, 99.5)), 1e-6)
        a.imshow(arr, cmap='gray', vmin=0, vmax=high)
        a.set_title(title)
    ax[3].imshow(gt_overlay)
    ax[0].set_title('Last raw frame')
    ax[3].set_title('GT overlay')
    for a in ax:
        a.axis('off')
    fig.suptitle(f'DIAS s{args.sequence}: temporal evidence (display scaling only)')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
