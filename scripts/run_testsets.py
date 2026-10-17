#!/usr/bin/env python3
"""Robustness suite: build synthetic test runs, run a pipeline on each, score, and tabulate.

    python3 scripts/run_testsets.py BAG gt.csv --laps 2,6 --suite quick \
        --cmd "python3 run.py {mcap} {csv} --cams fl,rl --pf-config out/final/selected_config.json"

    --suite quick | single | combined | all  (testgen/presets.py), or --presets clean,dark,fl_off,
    or --specs "name=spec;name2=spec2" for ad-hoc combinations.
    --cmd placeholders: {mcap} test file, {csv} submission path, {dir} test folder. Without --cmd only the
    test files are built.

Writes <root>/results[_<tag>].csv and .md (one row per lap x scenario, with the change vs clean).
Test files are reused when they exist (--rebuild to force); submissions are re-run unless --skip-run-existing.
"""
import argparse, os, re, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
import pandas as pd                                                   # noqa: E402
from testgen.presets import PRESETS, SUITES                           # noqa: E402
from testgen.build import build, laps_from_gt                         # noqa: E402

METRICS = [('coverage', r'coverage \(1 s bins\)\s+([\d.]+)'), ('rmse', r'position RMSE\s+\[m\]\s+([\d.]+)'),
           ('lateral', r'lateral RMSE\s+\[m\]\s+([\d.]+)'), ('longitudinal', r'longitudinal RMSE \[m\]\s+([\d.]+)'),
           ('yaw_deg', r'yaw RMSE\s+\[deg\]\s+([\d.]+)'), ('p95', r'position median/p95\s+[\d.]+ / ([\d.]+)'),
           ('within_1m', r'within 1\.00 m\s+([\d.]+)')]


def score(gt, csv):
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'evaluate.py'), gt, csv],
                         capture_output=True, text=True).stdout
    return {k: (float(m.group(1)) if (m := re.search(rx, out)) else float('nan')) for k, rx in METRICS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag'); ap.add_argument('gt')
    ap.add_argument('--laps', default='2,6')
    ap.add_argument('--suite', default=None); ap.add_argument('--presets', default=None)
    ap.add_argument('--specs', default=None, help='"name=spec;name2=spec2"')
    ap.add_argument('--cmd', default=None)
    ap.add_argument('--tag', default='', help='name of this pipeline variant: writes submission_<tag>.csv, '
                    'run_<tag>.log, results_<tag>.csv/md, so several variants share the same test files')
    ap.add_argument('--root', default='out/testsets')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--rebuild', action='store_true'); ap.add_argument('--skip-run-existing', action='store_true')
    ap.add_argument('--timeout-s', type=float, default=7200)
    ap.add_argument('--wait-built', type=float, default=0, help='seconds: wait for another job to build each test '
                    'file instead of building it here (run a second variant alongside the first); 0 = build')
    a = ap.parse_args()
    sfx = f'_{a.tag}' if a.tag else ''

    scen = {}
    names = SUITES[a.suite] if a.suite else (a.presets.split(',') if a.presets else [])
    for n in names:
        scen[n] = '@' + n
    for part in filter(None, (a.specs or '').split(';')):
        n, _, s = part.partition('=')
        scen[n.strip()] = s.strip()
    if 'clean' not in scen:
        scen = {'clean': '', **scen}
    laps = laps_from_gt(pd.read_csv(a.gt)).set_index('lap')
    rows = []
    for lap in [int(x) for x in a.laps.split(',')]:
        t0, t1 = int(laps.loc[lap, 't0_ns']), int(laps.loc[lap, 't1_ns'])
        for name, spec in scen.items():
            d = os.path.join(a.root, f'lap{lap}_{name}')
            waited = 0
            failed = lambda: os.path.exists(os.path.join(d, 'BUILD_FAILED'))
            while a.wait_built and not os.path.exists(os.path.join(d, 'manifest.json')) and not failed() \
                    and waited < a.wait_built:
                if waited == 0:
                    print(f'[wait]  lap {lap} {name}: test file not built yet', flush=True)
                time.sleep(30); waited += 30
            if a.wait_built and failed() and not os.path.exists(os.path.join(d, 'manifest.json')):
                print(f'[skip]  lap {lap} {name}: the other job could not build it', flush=True)
                rows.append(dict(lap=lap, scenario=name, spec=spec, exit='build failed'))
                continue
            if a.rebuild or not os.path.exists(os.path.join(d, 'manifest.json')):
                print(f'[build] lap {lap} {name}: {spec!r}', flush=True)
                try:
                    if os.path.exists(os.path.join(d, 'BUILD_FAILED')):
                        os.remove(os.path.join(d, 'BUILD_FAILED'))
                    build(a.bag, a.gt, d, t0, t1, spec, a.seed, a.workers, label=f'lap{lap}_{name}')
                except Exception as e:                       # one broken scenario must not stop the suite
                    print(f'[error] lap {lap} {name}: build failed: {e!r}', flush=True)
                    os.makedirs(d, exist_ok=True)
                    open(os.path.join(d, 'BUILD_FAILED'), 'w').write(repr(e) + '\n')
                    rows.append(dict(lap=lap, scenario=name, spec=spec, exit='build failed'))
                    continue
            row = dict(lap=lap, scenario=name, spec=spec)
            if a.cmd:
                csv = os.path.join(d, f'submission{sfx}.csv')
                if not (a.skip_run_existing and os.path.exists(csv)):
                    cmd = a.cmd.format(mcap=shlex.quote(os.path.join(d, 'test.mcap')), csv=shlex.quote(csv),
                                       dir=shlex.quote(d))
                    print(f'[run]   {cmd}', flush=True)
                    tic = time.time()
                    with open(os.path.join(d, f'run{sfx}.log'), 'w') as log:
                        r = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout_s)
                    row.update(exit=r.returncode, run_s=round(time.time() - tic, 1))
                if os.path.exists(csv):
                    row.update(score(os.path.join(d, 'gt.csv'), csv))
            rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(a.root, f'results{sfx}.csv'), index=False)

    df = pd.DataFrame(rows)
    if 'rmse' in df:
        base = df[df.scenario == 'clean'].set_index('lap')
        for m in ('rmse', 'lateral', 'longitudinal', 'within_1m'):
            df[f'd_{m}'] = df.apply(lambda r: r[m] - base.loc[r.lap, m] if r.lap in base.index else float('nan'), axis=1)
    df.to_csv(os.path.join(a.root, f'results{sfx}.csv'), index=False)
    cols = [c for c in ['lap', 'scenario', 'coverage', 'rmse', 'd_rmse', 'lateral', 'longitudinal', 'yaw_deg',
                        'p95', 'within_1m', 'd_within_1m', 'run_s'] if c in df]
    with open(os.path.join(a.root, f'results{sfx}.md'), 'w') as f:
        try:
            f.write(df[cols].to_markdown(index=False, floatfmt='.3f'))
        except ImportError:                      # tabulate not installed
            f.write(df[cols].to_string(index=False))
    print(df[cols].to_string(index=False))
    print(f'-> {a.root}/results{sfx}.csv, results{sfx}.md')


if __name__ == '__main__':
    main()
