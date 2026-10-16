#!/usr/bin/env python3
"""Particle filter: car-state odometry + track map prior [+ relocalization fixes].

    python3 scripts/run_pf.py cache gt.csv out/map_pf.csv                         # baseline 1, no cameras
    python3 scripts/run_pf.py cache gt.csv out/reloc_pf.csv --fixes cache/reloc_fl.csv

--fixes: per-image relocalization results (timestamp_ns, ok, x, y, yaw, n_inl), from
scripts/precompute_reloc.py. A fix with stamp t is applied at the output stamp t (causal).
Inputs: state speeds / yaw rate, track_bounds, camera fixes, first GT pose only.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.trackmap import TrackMap
from camloc.motion import increments
from camloc.pf import ParticleFilter, PFParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('out')
    ap.add_argument('--fixes', default=None)
    ap.add_argument('--min-inl', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    for k, v in PFParams.__dataclass_fields__.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v.default), default=v.default)
    a = ap.parse_args()
    params = PFParams(**{k: getattr(a, k) for k in PFParams.__dataclass_fields__})

    tm = TrackMap(f'{a.cache}/track_bounds.npz')
    g0 = pd.read_csv(a.gt, nrows=1)
    t0 = int(g0.timestamp_ns[0])
    f = pd.read_csv(f'{a.cache}/frames.csv')
    img_t = np.sort(f[f.cam == 'fl'].timestamp_ns.values)
    inc = increments(f'{a.cache}/state.csv', img_t, t0)
    fixes = {}
    if a.fixes:
        fx = pd.read_csv(a.fixes)
        fx = fx[(fx.ok == 1) & (fx.n_inl >= a.min_inl)]
        fixes = {int(r.timestamp_ns): (r.x, r.y, r.yaw) for r in fx.itertuples()}

    pf = ParticleFilter(tm, (g0.x[0], g0.y[0], g0['yaw'][0]), params, a.seed)
    rows, n_used, n_gated, tic = [], 0, 0, time.perf_counter()
    for t, dx, dy, dyaw, dt, dist in inc.itertuples(index=False):
        pf.predict(dx, dy, dyaw, dt)
        if dist > 0.05:                       # the map constraints carry no information at standstill
            pf.update_map()
        if t in fixes:
            if pf.update_fix(*fixes[t]):
                n_used += 1
            else:
                n_gated += 1
        rows.append((t,) + pf.estimate())
    el = time.perf_counter() - tic

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    pd.DataFrame(rows, columns=['timestamp_ns', 'x', 'y', 'yaw']).to_csv(a.out, index=False, float_format='%.6f')
    print(f'{len(rows)} poses -> {a.out}   filter {el / len(rows) * 1e3:.2f} ms/frame   '
          f'fixes used {n_used}, gated {n_gated}')


if __name__ == '__main__':
    main()
