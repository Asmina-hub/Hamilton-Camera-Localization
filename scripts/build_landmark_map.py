#!/usr/bin/env python3
"""Build the landmark map from the training split (GT poses are allowed for training).

    python3 scripts/build_landmark_map.py BAG cache gt.csv cache/landmarks_fl.npz --cam fl

Only frames inside 'train' segments of cache/laps.csv are used, so the validation laps stay unseen.
"""
import argparse, os, sys, time
from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc import landmarks as L

_W = {}


def _init(cam_name):
    _W['sift'] = L.make_sift()
    _W['mask'] = (~body_masks()[cam_name]).astype(np.uint8) * 255


def _work(item):
    t, jpeg = item
    g = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return (t,) + L.extract(_W['sift'], g, _W['mask'])


def images(bag, cam, windows):
    with open(bag, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for t0, t1 in windows:
            for _, _, _, msg in r.iter_decoded_messages(
                    topics=[f'/constructor0/sensor/camera_{cam}/compressed_image'], start_time=t0 - 10**8,
                    end_time=t1):
                ts = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if t0 <= ts <= t1:
                    yield ts, bytes(msg.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('out')
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--k', type=int, default=3, help='frame gap for triangulation')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--voxel', type=float, default=0.5, help='m, one landmark per voxel')
    a = ap.parse_args()

    cam = load_cameras(f'{a.cache}/calib.json')[a.cam]
    g = pd.read_csv(a.gt)
    gyaw = np.unwrap(g['yaw'].values)
    T = lambda t: L.pose_T(*(np.interp(t, g.timestamp_ns, v) for v in (g.x, g.y, g.z, gyaw)))
    laps = pd.read_csv(f'{a.cache}/laps.csv')
    windows = [(int(r.t0_ns), int(r.t1_ns)) for r in laps[laps.split == 'train'].itertuples()]
    bf = cv2.BFMatcher(cv2.NORM_L2)

    pts, desc, src, buf = [], [], [], []
    tic, n = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(a.cam,)) as pool:
        for t, p, d in pool.imap(_work, images(a.bag, a.cam, windows), chunksize=4):
            n += 1
            if buf and t - buf[-1][0] > 150_000_000:        # gap (dropped frame or next window): restart
                buf = []
            buf.append((t, p, d))
            if len(buf) <= a.k:
                continue
            ti, pi, di = buf[-1 - a.k]
            buf = buf[-a.k - 1:]
            if len(di) < 10 or len(d) < 10:
                continue
            m = [q[0] for q in bf.knnMatch(di.astype(np.float32), d.astype(np.float32), k=2)
                 if len(q) == 2 and q[0].distance < 0.75 * q[1].distance]
            if len(m) < 10:
                continue
            qi, qj = np.array([x.queryIdx for x in m]), np.array([x.trainIdx for x in m])
            Ti = T(ti)
            X, ok = L.triangulate(cam, Ti, T(t), pi[qi], p[qj])
            pts.append(X[ok].astype(np.float32))
            desc.append(di[qi[ok]])
            src.append(np.tile([Ti[0, 3], Ti[1, 3], np.arctan2(Ti[1, 0], Ti[0, 0])], (ok.sum(), 1)).astype(np.float32))
            if n % 500 == 0:
                print(f'{n} frames, {sum(map(len, pts))} landmarks, {n / (time.time() - tic):.1f} fps', flush=True)
    P, D, S = L.voxel_dedupe(np.vstack(pts), np.vstack(desc), np.vstack(src), a.voxel)
    np.savez(a.out, pts=P, desc=D, src=S)
    print(f'{n} frames -> {sum(map(len, pts))} landmarks, {len(P)} after dedupe -> {a.out}  ({time.time() - tic:.0f} s)')


if __name__ == '__main__':
    main()
