#!/usr/bin/env python3
"""Split the run into laps by GT crossings of a gate line, choose validation laps.

    python3 scripts/split_laps.py gt.csv cache

Gate: y = GATE_Y on the right leg of the track (x in GATE_X), crossed going up. It is far from the pit
lane, so pit-lane driving at the start and end of the run falls into the out-lap / in-lap segments.
GT is used here only to define the split, not as a method input.

cache/laps.csv     lap, t0_ns, t1_ns, kind (outlap | lap | inlap), split (train | val), duration_s, length_m, vmax
"""
import sys
import numpy as np
import pandas as pd

GATE_Y, GATE_X = 800.0, (20.0, 120.0)
VAL_LAPS = [3, 7]  # 1-based full-lap numbers: one medium (35 m/s) and one fastest (40 m/s) lap


def main(gt_csv, cache):
    g = pd.read_csv(gt_csv)
    t, x, y = g.timestamp_ns.values, g.x.values, g.y.values
    s = pd.read_csv(f'{cache}/state.csv')
    v = np.interp(t, s.timestamp_ns.values, s.v_mps.values)

    up = np.where((y[:-1] < GATE_Y) & (y[1:] >= GATE_Y) & (x[:-1] > GATE_X[0]) & (x[:-1] < GATE_X[1]))[0]
    edges = [0] + list(up + 1) + [len(t) - 1]
    rows = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        kind = 'outlap' if k == 0 else 'inlap' if k == len(edges) - 2 else 'lap'
        rows.append(dict(lap=k, t0_ns=t[a], t1_ns=t[b], kind=kind,
                         split='val' if kind == 'lap' and k in VAL_LAPS else 'train',
                         duration_s=round((t[b] - t[a]) / 1e9, 1),
                         length_m=round(float(np.sum(np.hypot(np.diff(x[a:b + 1]), np.diff(y[a:b + 1])))), 0),
                         vmax=round(float(v[a:b + 1].max()), 1)))
    laps = pd.DataFrame(rows)
    laps.to_csv(f'{cache}/laps.csv', index=False)
    print(laps.to_string(index=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
