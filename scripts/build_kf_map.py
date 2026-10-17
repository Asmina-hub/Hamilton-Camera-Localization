#!/usr/bin/env python3
"""Build the ALIKED + LightGlue keyframe map from the training split (GT poses allowed for training).

    python scripts/build_kf_map.py BAG cache gt.csv maps/kf_fl.npz --cam fl

Every frame of the 'train' segments of cache/laps.csv is processed; every STRIDE-th frame becomes a
keyframe, matched with the frame K later and triangulated. Only triangulated keypoints are stored.
"""
import argparse, os, sys, time
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc.features import Features, subset, to_numpy
from camloc.landmarks import pose_T, triangulate
from build_landmark_map import images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('out')
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--k', type=int, default=3, help='frame gap for triangulation')
    ap.add_argument('--stride', type=int, default=2, help='keyframe every STRIDE frames')
    ap.add_argument('--exclude-laps', default='', help='training laps to leave out (held-out tuning laps), e.g. 2,6')
    a = ap.parse_args()

    cam = load_cameras(f'{a.cache}/calib.json')[a.cam]
    mask = body_masks()[a.cam]
    g = pd.read_csv(a.gt)
    gyaw = np.unwrap(g['yaw'].values)
    T = lambda t: pose_T(*(np.interp(t, g.timestamp_ns, v) for v in (g.x, g.y, g.z, gyaw)))
    laps = pd.read_csv(f'{a.cache}/laps.csv')
    excl = [int(x) for x in a.exclude_laps.split(',') if x]
    use = laps[(laps.split == 'train') & ~laps.lap.isin(excl)]
    windows = [(int(r.t0_ns), int(r.t1_ns)) for r in use.itertuples()]
    print(f'map from laps {list(use.lap)} (excluded {excl})', flush=True)
    feats = Features()

    kf_pose, kp, de, sc, X3 = [], [], [], [], []
    buf, n, tic = [], 0, time.time()
    for t, jpeg in images(a.bag, a.cam, windows):
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if buf and t - buf[-1][0] > 150_000_000:          # gap: restart the window
            buf = []
        buf.append((t, feats.extract(gray, mask), n))
        n += 1
        if len(buf) <= a.k:
            continue
        ti, fi, ni = buf[-1 - a.k]
        buf = buf[-a.k - 1:]
        if ni % a.stride:
            continue
        mi, mj = feats.match(fi, buf[-1][1])
        if len(mi) < 15:
            continue
        Ti = T(ti)
        X, ok = triangulate(cam, Ti, T(buf[-1][0]),
                            fi['keypoints'][0].cpu().numpy()[mi], buf[-1][1]['keypoints'][0].cpu().numpy()[mj])
        if ok.sum() < 15:
            continue
        f = to_numpy(subset(fi, mi[ok]))
        kf_pose.append([Ti[0, 3], Ti[1, 3], np.arctan2(Ti[1, 0], Ti[0, 0])])
        kp.append(f['kpts']); de.append(f['desc']); sc.append(f['score']); X3.append(X[ok].astype(np.float32))
        if len(kf_pose) % 500 == 0:
            print(f'{n} frames, {len(kf_pose)} keyframes, {sum(map(len, X3))} landmarks, '
                  f'{n / (time.time() - tic):.1f} fps', flush=True)

    off = np.concatenate([[0], np.cumsum([len(x) for x in X3])])
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out, kf_pose=np.array(kf_pose, np.float64), off=off, kpts=np.vstack(kp), desc=np.vstack(de),
             score=np.concatenate(sc), X=np.vstack(X3), image_size=np.array([cam.w, cam.h]))
    print(f'{n} frames -> {len(kf_pose)} keyframes, {off[-1]} landmarks -> {a.out} ({time.time() - tic:.0f} s)')


if __name__ == '__main__':
    main()
