#!/usr/bin/env python3
"""Build a synthetic hidden-test run from one lap (or time window) of the training run.

    python3 scripts/make_testset.py BAG gt.csv --lap 6 --spec "@fast_125+lamps_off" --out out/testsets/l6_fast_lamps
    python3 scripts/make_testset.py BAG gt.csv --lap 2 --preset dark
    python3 scripts/make_testset.py --list            # all transforms and presets
    python3 scripts/make_testset.py BAG gt.csv --list-laps

Output folder: test.mcap (run any pipeline on it), gt.csv (score it with tools/evaluate.py),
manifest.json (exactly what was applied), preview.jpg (original vs test images).
Use held-out laps (2, 6) for decisions; laps 3 / 7 only for the final report.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import pandas as pd                                                   # noqa: E402
from testgen.transforms import describe_all                           # noqa: E402
from testgen.presets import PRESETS                                   # noqa: E402
from testgen.build import build, laps_from_gt                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', nargs='?'); ap.add_argument('gt', nargs='?')
    ap.add_argument('--lap', type=int, help='lap number as in scripts/split_laps.py (0 = out-lap)')
    ap.add_argument('--start-s', type=float, help='window start, seconds after the first GT pose')
    ap.add_argument('--dur-s', type=float, default=120.0)
    ap.add_argument('--spec', default='', help='transforms, e.g. "speedup:k=1.25+lamps_off+@dark"')
    ap.add_argument('--preset', default=None, help='shortcut for --spec @NAME')
    ap.add_argument('--out', default=None, help='output folder (default out/testsets/lap<N>_<preset or custom>)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--jpeg-quality', type=int, default=92, help='re-encoding quality for changed images')
    ap.add_argument('--keep-state-pose', action='store_true', help='do not blank the GT fields in state_estimation')
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--list', action='store_true'); ap.add_argument('--list-laps', action='store_true')
    a = ap.parse_args()

    if a.list:
        print('TRANSFORMS\n' + describe_all() + '\n\nPRESETS')
        for n, p in PRESETS.items():
            print(f'  @{n:16s} [{p["group"]}] {p["spec"]!r}\n      {p["note"]}')
        return
    if not (a.bag and a.gt):
        ap.error('BAG and gt.csv are required')
    gt = pd.read_csv(a.gt)
    laps = laps_from_gt(gt)
    if a.list_laps:
        print(laps.to_string(index=False)); return
    if a.preset:
        a.spec = '@' + a.preset + ('+' + a.spec if a.spec else '')
    if a.lap is not None:
        r = laps[laps.lap == a.lap].iloc[0]
        t0, t1, where = int(r.t0_ns), int(r.t1_ns), f'lap{a.lap}'
    elif a.start_s is not None:
        t0 = int(gt.timestamp_ns.iloc[0] + a.start_s * 1e9); t1 = int(t0 + a.dur_s * 1e9)
        where = f't{int(a.start_s)}'
    else:
        ap.error('give --lap or --start-s')
    tag = a.preset or ('custom' if a.spec else 'clean')
    out = a.out or os.path.join('out', 'testsets', f'{where}_{tag}')
    m = build(a.bag, a.gt, out, t0, t1, a.spec, a.seed, a.workers, a.jpeg_quality, a.keep_state_pose,
              not a.no_preview, label=f'{where}_{tag}')
    print(json.dumps({k: m[k] for k in ('label', 'window', 'speedup', 'images', 'size_mb', 'build_s')}, indent=1))
    print(f'-> {out}/test.mcap   score with: python3 tools/evaluate.py {out}/gt.csv <submission.csv>')


if __name__ == '__main__':
    main()
