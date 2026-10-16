#!/usr/bin/env python3
"""Accuracy of per-image relocalization fixes against GT, per split (development only).

    python3 scripts/eval_reloc.py gt.csv cache/reloc_fl.csv cache [--min-inl 15]
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gt'); ap.add_argument('fixes'); ap.add_argument('cache')
    ap.add_argument('--min-inl', type=int, default=15)
    a = ap.parse_args()
    g, f = pd.read_csv(a.gt), pd.read_csv(a.fixes)
    laps = pd.read_csv(f'{a.cache}/laps.csv')
    t = f.timestamp_ns.values
    gyaw = np.unwrap(g['yaw'].values)
    gx, gy = np.interp(t, g.timestamp_ns, g.x), np.interp(t, g.timestamp_ns, g.y)
    gh = np.interp(t, g.timestamp_ns, gyaw)
    ex, ey = f.x - gx, f.y - gy
    f['lon'], f['lat'] = np.cos(gh) * ex + np.sin(gh) * ey, -np.sin(gh) * ex + np.cos(gh) * ey
    f['dyaw'] = np.degrees(np.angle(np.exp(1j * (f['yaw'] - gh))))
    f['split'] = 'other'
    for r in laps.itertuples():
        f.loc[(t >= r.t0_ns) & (t <= r.t1_ns), 'split'] = r.split
    if 'ms_loc' in f:
        print(f'timing (median): features {f.ms_feat.median():.0f} ms/worker, localize {f.ms_loc.median():.0f} ms')
    for s, d in f.groupby('split'):
        ok = d[(d.ok == 1) & (d.n_inl >= a.min_inl)]
        print(f'\n[{s}] frames {len(d)}, fixes {len(ok)} ({len(ok) / len(d) * 100:.0f} %), median inliers {ok.n_inl.median():.0f}')
        for c in ['lat', 'lon', 'dyaw']:
            e = ok[c].abs()
            unit = 'deg' if c == 'dyaw' else 'm'
            print(f'  |{c:4s}| median {e.median():.2f} {unit}  p90 {e.quantile(0.9):.2f}  p99 {e.quantile(0.99):.2f}  '
                  f'<0.5: {np.mean(e < 0.5) * 100:3.0f} %  <1: {np.mean(e < 1) * 100:3.0f} %  >5: {np.mean(e > 5) * 100:.1f} %')


if __name__ == '__main__':
    main()
