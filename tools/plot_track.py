#!/usr/bin/env python3
"""Track borders + GT trajectory (+ optional submission) -> png.

    python3 tools/plot_track.py cache gt.csv track.png [submission.csv]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(cache, gt_csv, out, sub_csv=None):
    b = np.load(f'{cache}/track_bounds.npz')
    gt = np.genfromtxt(gt_csv, delimiter=',', names=True)
    fig, ax = plt.subplots(figsize=(14, 14))
    for k in b.files:
        ax.plot(b[k][:, 0], b[k][:, 1], 'k-' if k.startswith('track') else 'b-', lw=0.6, label=k)
    sc = ax.scatter(gt['x'], gt['y'], c=(gt['timestamp_ns'] - gt['timestamp_ns'][0]) / 1e9, s=0.3, cmap='viridis')
    fig.colorbar(sc, ax=ax, label='t [s]', shrink=0.6)
    ax.plot(gt['x'][0], gt['y'][0], 'r*', ms=15, label='start')
    if sub_csv:
        s = np.genfromtxt(sub_csv, delimiter=',', names=True)
        ax.plot(s['x'], s['y'], 'r-', lw=0.8, label='estimate')
    ax.set_aspect('equal')
    ax.legend()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('->', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
