#!/usr/bin/env python3
"""Run tools/evaluate.py on the full run and on each validation lap.

    python3 scripts/eval_laps.py gt.csv submission.csv [cache]

For a lap, GT is cropped to [t0, t1] so coverage and errors are computed on that lap only.
"""
import os, subprocess, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, '..', 'tools', 'evaluate.py')


def run(gt_path, sub_path, title):
    print(f'===== {title}')
    print(subprocess.run([sys.executable, EVAL, gt_path, sub_path], capture_output=True, text=True).stdout)


def main(gt_csv, sub_csv, cache='cache'):
    run(gt_csv, sub_csv, 'full run')
    gt = pd.read_csv(gt_csv)
    laps = pd.read_csv(f'{cache}/laps.csv')
    with tempfile.TemporaryDirectory() as d:
        for _, l in laps[laps.split == 'val'].iterrows():
            p = os.path.join(d, f'gt_lap{l.lap}.csv')
            gt[(gt.timestamp_ns >= l.t0_ns) & (gt.timestamp_ns <= l.t1_ns)].to_csv(p, index=False)
            run(p, sub_csv, f'val lap {l.lap} ({l.duration_s} s, vmax {l.vmax} m/s)')


if __name__ == '__main__':
    main(*sys.argv[1:])
