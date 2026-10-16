#!/usr/bin/env python3
"""Result overview: estimate vs GT on the track map, and the error over time.

    python3 tools/plot_result.py cache gt.csv submission.csv out/result.png [--lap 7] [--title "..."]

Left: track borders, GT and estimate for one lap (default: the last validation lap).
Right: lateral and longitudinal error over the whole run (GT car frame), validation laps shaded.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GT_C, EST_C = '#2a78d6', '#eb6834'          # categorical slots 1, 2
INK, MUTED, GRID, BORDER = '#2b2b29', '#6b6a63', '#e4e3de', '#b9b8b0'


def errors(gt, sb):
    t = sb.timestamp_ns.values
    gyaw = np.unwrap(gt['yaw'].values)
    gx, gy = np.interp(t, gt.timestamp_ns, gt.x), np.interp(t, gt.timestamp_ns, gt.y)
    gh = np.interp(t, gt.timestamp_ns, gyaw)
    dx, dy = sb.x.values - gx, sb.y.values - gy
    return np.cos(gh) * dx + np.sin(gh) * dy, -np.sin(gh) * dx + np.cos(gh) * dy


def style(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('sub'); ap.add_argument('out')
    ap.add_argument('--lap', type=int, default=None)
    ap.add_argument('--title', default='')
    a = ap.parse_args()

    gt, sb = pd.read_csv(a.gt), pd.read_csv(a.sub)
    laps = pd.read_csv(f'{a.cache}/laps.csv')
    val = laps[laps.split == 'val']
    lap = laps.set_index('lap').loc[a.lap if a.lap is not None else val.lap.max()]
    b = np.load(f'{a.cache}/track_bounds.npz')
    lon, lat = errors(gt, sb)
    t0 = gt.timestamp_ns.iloc[0]
    ts = (sb.timestamp_ns.values - t0) / 1e9

    fig = plt.figure(figsize=(15, 8.5), facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    for k in b.files:
        ax.plot(b[k][:, 0], b[k][:, 1], color=BORDER, lw=0.8)
    mg = (gt.timestamp_ns >= lap.t0_ns) & (gt.timestamp_ns <= lap.t1_ns)
    ms = (sb.timestamp_ns >= lap.t0_ns) & (sb.timestamp_ns <= lap.t1_ns)
    ax.plot(gt.x[mg], gt.y[mg], color=GT_C, lw=2, label='ground truth')
    ax.plot(sb.x[ms], sb.y[ms], color=EST_C, lw=2, label='estimate')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=10, labelcolor=INK)
    pos = np.hypot(lon, lat)
    sel = ms.values
    ax.set_title(f'Lap {lap.name} ({lap.split}, vmax {lap.vmax:.0f} m/s): position RMSE '
                 f'{np.sqrt(np.mean(pos[sel] ** 2)):.2f} m', color=INK, fontsize=11, loc='left')

    for row, (e, name) in enumerate([(lat, 'Lateral error [m]'), (lon, 'Longitudinal error [m]')]):
        ax2 = fig.add_subplot(gs[row, 1])
        style(ax2)
        for _, v in val.iterrows():
            ax2.axvspan((v.t0_ns - t0) / 1e9, (v.t1_ns - t0) / 1e9, color='#f1f0ec', lw=0)
            if row == 0:
                ax2.text((v.t0_ns - t0) / 1e9 + 3, 1, f'val lap {v.lap}', transform=ax2.get_xaxis_transform(),
                         va='top', color=MUTED, fontsize=9)
        ax2.axhline(0, color=MUTED, lw=0.8)
        ax2.plot(ts, e, color=EST_C, lw=1.2)
        ax2.set_title(f'{name}   RMSE {np.sqrt(np.mean(e ** 2)):.2f} m (full run)', color=INK, fontsize=11,
                      loc='left')
        lim = max(1.0, np.percentile(np.abs(e), 99) * 1.1)
        ax2.set_ylim(-lim, lim)
        if row == 1:
            ax2.set_xlabel('time [s]', color=MUTED, fontsize=9)
    if a.title:
        fig.suptitle(a.title, x=0.02, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.savefig(a.out, dpi=120, bbox_inches='tight', facecolor='white')
    print('->', a.out)


if __name__ == '__main__':
    main()
