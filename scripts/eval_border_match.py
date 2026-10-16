#!/usr/bin/env python3
"""Offline check of the border-matching measurement on cached 1 Hz images (development only).

    python3 scripts/eval_border_match.py cache gt.csv [--split train] [--every 3]

For each camera and image, scores a grid of lateral / yaw offsets around the GT pose (and around a
   pose shifted by a known offset) and reports how far the minimum is from the truth.
"""
import argparse, os, sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras, CAMS
from camloc.trackmap import TrackMap
from camloc.border_match import BorderMatcher
from camloc.body_masks import body_masks

DYS = np.arange(-3.0, 3.01, 0.1)
DYAWS = np.radians(np.arange(-3.0, 3.01, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache'); ap.add_argument('gt')
    ap.add_argument('--split', default='train')
    ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--shift', type=float, default=1.5, help='m, known lateral offset of the reference pose')
    ap.add_argument('--edge', default='', help='edge config overrides, e.g. "clahe=0,blur=7,lo=20,hi=60,min_len=30"')
    ap.add_argument('--cams', default='cl,cr,rl,rr,fl,fr,r')
    a = ap.parse_args()

    import camloc.border_match as bmod
    for kv in filter(None, a.edge.split(',')):
        k, v = kv.split('=')
        bmod.EDGE_CFG[k] = type(bmod.EDGE_CFG[k])(float(v)) if k != 'clahe' else bool(int(v))
    print('edge config', bmod.EDGE_CFG)
    cams = load_cameras(f'{a.cache}/calib.json')
    tm = TrackMap(f'{a.cache}/track_bounds.npz')
    bm = BorderMatcher(cams, tm, body_masks())
    gt = pd.read_csv(a.gt)
    gyaw = np.unwrap(gt['yaw'].values)
    laps = pd.read_csv(f'{a.cache}/laps.csv')
    laps = laps[(laps.split == a.split) & (laps.kind == 'lap')]

    res = []
    for c in a.cams.split(','):
        files = np.array(sorted(int(p[:-4]) for p in os.listdir(f'{a.cache}/img/{c}')))
        inlap = np.zeros(len(files), bool)
        for _, l in laps.iterrows():
            inlap |= (files >= l.t0_ns) & (files <= l.t1_ns)
        for t in files[inlap][:: a.every]:
            pose = np.array([np.interp(t, gt.timestamp_ns, gt.x), np.interp(t, gt.timestamp_ns, gt.y),
                             np.interp(t, gt.timestamp_ns, gyaw)])
            dt, _ = bm.dt_image(c, cv2.imread(f'{a.cache}/img/{c}/{t}.jpg'))
            # reference shifted left by `shift` in the car frame: the true offset is then -shift
            ref = pose + [-np.sin(pose[2]) * a.shift, np.cos(pose[2]) * a.shift, 0]
            S, n = bm.score_grid(ref, {c: dt}, DYS, DYAWS)
            if n == 0:
                continue
            i, j = np.unravel_index(np.argmin(S), S.shape)
            res.append(dict(cam=c, t=t, n=n, err_dy=DYS[j] + a.shift, err_dyaw=np.degrees(DYAWS[i]),
                            contrast=np.median(S) - S[i, j]))
    r = pd.DataFrame(res)
    r.to_csv(f'{a.cache}/border_match_eval.csv', index=False)
    g = r.groupby('cam')
    print(pd.DataFrame({
        'frames': g.size(),
        'pts_med': g.n.median(),
        'dy_med_abs': g.err_dy.apply(lambda e: np.median(np.abs(e))).round(2),
        'dy<0.3m %': g.err_dy.apply(lambda e: np.mean(np.abs(e) < 0.3) * 100).round(0),
        'dy<0.5m %': g.err_dy.apply(lambda e: np.mean(np.abs(e) < 0.5) * 100).round(0),
        'dy<1m %': g.err_dy.apply(lambda e: np.mean(np.abs(e) < 1.0) * 100).round(0),
        'dyaw_med_abs': g.err_dyaw.apply(lambda e: np.median(np.abs(e))).round(2),
    }).to_string())


if __name__ == '__main__':
    main()
