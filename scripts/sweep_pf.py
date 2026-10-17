#!/usr/bin/env python3
"""Grid search of filter parameters and camera subsets by replaying logged fixes.

    python3 scripts/sweep_pf.py cache gt.csv out/lolo/log.csv --tune-laps 2,6 --cams fl fl,cl fl,rl fl,cl,rl

The log must come from an online run whose maps EXCLUDE the tuning laps (build_kf_map --exclude-laps),
so the fixes on those laps are as honest as on unseen data. Only the tuning laps are scored; validation
laps are never evaluated here, so no choice can depend on them.
"""
import argparse, itertools, json, os, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from run_pf import run_filter, load_fixes
from camloc.pf import PFParams

GRID = dict(fix_lat=[0.4, 0.2, 0.1], fix_lon=[1.5, 0.6, 0.3], fix_gate_pos=[10.0, 3.0], fix_floor=[0.05, 0.2])
_A = {}


def rmse(gt, poses, windows):
    t = poses.timestamp_ns.values
    m = np.zeros(len(t), bool)
    for t0, t1 in windows:
        m |= (t >= t0) & (t <= t1)
    gx, gy = np.interp(t[m], gt.timestamp_ns, gt.x), np.interp(t[m], gt.timestamp_ns, gt.y)
    e = np.hypot(poses.x.values[m] - gx, poses.y.values[m] - gy)
    return np.sqrt(np.mean(e ** 2)), np.mean(e < 1.0) * 100


def work(job):
    cams, combo = job
    p = PFParams(**dict(zip(GRID, combo)))
    poses, used, gated, _ = run_filter(_A['cache'], _A['gt'], p, _A['fixes'][cams], stamps=_A['stamps'])
    r = dict(cams=cams, **dict(zip(GRID, combo)), used=used, gated=gated)
    for lap, w in _A['windows'].items():
        r[f'lap{lap}_rmse'], r[f'lap{lap}_<1m%'] = rmse(_A['gtdf'], poses, [w])
    r['tune_rmse'], r['tune_<1m%'] = rmse(_A['gtdf'], poses, list(_A['windows'].values()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('log')
    ap.add_argument('--tune-laps', required=True)
    ap.add_argument('--cams', nargs='+', default=['fl'])
    a = ap.parse_args()
    laps = pd.read_csv(f'{a.cache}/laps.csv').set_index('lap')
    tune = [int(x) for x in a.tune_laps.split(',')]
    assert not (laps.loc[tune, 'split'] == 'val').any(), 'tuning laps must be training laps'
    log = pd.read_csv(a.log)
    fixes = {}
    for c in a.cams:
        sub = a.log + '.tmp_' + c.replace(',', '_')
        log[log.cam.isin(c.split(','))].to_csv(sub, index=False)
        fixes[c] = load_fixes(sub)
        os.remove(sub)
    _A.update(cache=a.cache, gt=a.gt, fixes=fixes, stamps=log.timestamp_ns.unique(), gtdf=pd.read_csv(a.gt),
              windows={l: (laps.loc[l, 't0_ns'], laps.loc[l, 't1_ns']) for l in tune})
    jobs = [(c, combo) for c in a.cams for combo in itertools.product(*GRID.values())]
    with Pool(4) as pool:
        res = pd.DataFrame(pool.map(work, jobs)).sort_values('tune_rmse')
    res.to_csv(os.path.splitext(a.log)[0] + '_sweep.csv', index=False)
    print(res.round(3).to_string(index=False))
    best = res.iloc[0]
    sel = dict(cams=best.cams, params={k: float(best[k]) for k in GRID}, tune_laps=tune,
               tune_rmse=float(best.tune_rmse), rule='lowest position RMSE on the tuning laps')
    with open(os.path.splitext(a.log)[0] + '_best.json', 'w') as o:
        json.dump(sel, o, indent=1)
    print('selected:', sel)


if __name__ == '__main__':
    main()
