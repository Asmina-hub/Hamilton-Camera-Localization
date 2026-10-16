#!/usr/bin/env python3
"""Feasibility experiment for landmark relocalization (slide 16, method 3). Development only.

    python3 scripts/exp_landmarks.py cache gt.csv BAG --map-lap 4 --query-lap 3 --cam fl --t-from 20 --dur 20

Map: SIFT features in 10 Hz images of one training lap, triangulated between frames k apart using GT poses.
Query: images of another lap over the same stretch of track; match to the map, PnP, compare with GT.
GT is used to build the map (allowed: training data) and, in this experiment only, to pick the map
points near the query; the real system will use the filter's predicted pose instead.
"""
import argparse, os, sys
import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def read_images(bag, cam, t0, t1):
    out = []
    with open(bag, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in r.iter_decoded_messages(
                topics=[f'/constructor0/sensor/camera_{cam}/compressed_image'], start_time=t0 - 10**8, end_time=t1):
            ts = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if t0 <= ts <= t1:
                out.append((ts, cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)))
    return out


class GT:
    def __init__(self, path):
        g = pd.read_csv(path)
        self.t, self.x, self.y, self.z = g.timestamp_ns.values, g.x.values, g.y.values, g.z.values
        self.yaw = np.unwrap(g['yaw'].values)

    def pose(self, t):
        return np.array([np.interp(t, self.t, v) for v in (self.x, self.y, self.z, self.yaw)])

    def T_map_base(self, t):
        x, y, z, yaw = self.pose(t)
        T = np.eye(4)
        T[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        T[:3, 3] = x, y, z
        return T


def features(sift, img, mask):
    kp, des = sift.detectAndCompute(CLAHE.apply(img), mask)
    return np.array([k.pt for k in kp], np.float32).reshape(-1, 2), des


def undist(cam, pts):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), cam.K, cam.D).reshape(-1, 2)   # normalised coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('bag')
    ap.add_argument('--map-lap', type=int, default=4); ap.add_argument('--query-lap', type=int, default=3)
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--t-from', type=float, default=20.0); ap.add_argument('--dur', type=float, default=20.0)
    ap.add_argument('--k', type=int, default=3, help='frame gap for triangulation')
    a = ap.parse_args()

    from camloc.body_masks import body_masks
    cams = load_cameras(f'{a.cache}/calib.json'); cam = cams[a.cam]
    mask = (~body_masks()[a.cam]).astype(np.uint8) * 255
    gt = GT(a.gt)
    laps = pd.read_csv(f'{a.cache}/laps.csv').set_index('lap')
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # ---- map from the training lap
    t0 = int(laps.loc[a.map_lap, 't0_ns'] + a.t_from * 1e9); t1 = int(t0 + a.dur * 1e9)
    frames = read_images(a.bag, a.cam, t0, t1)
    feats = [features(sift, im, mask) for _, im in frames]
    P_cam = [np.linalg.inv(gt.T_map_base(t) @ cam.T_base_cam) for t, _ in frames]   # map -> camera
    pts3d, desc, src = [], [], []
    for i in range(len(frames) - a.k):
        j = i + a.k
        (pi, di), (pj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(di) < 10 or len(dj) < 10:
            continue
        m = [p for p in bf.knnMatch(di, dj, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(m) < 10:
            continue
        qi, qj = np.array([x[0].queryIdx for x in m]), np.array([x[0].trainIdx for x in m])
        ni, nj = undist(cam, pi[qi]), undist(cam, pj[qj])
        X = cv2.triangulatePoints(P_cam[i][:3], P_cam[j][:3], ni.T, nj.T)
        X = (X[:3] / X[3]).T
        ok = np.ones(len(X), bool)
        for P, n in ((P_cam[i], ni), (P_cam[j], nj)):
            Xc = X @ P[:3, :3].T + P[:3, 3]
            ok &= (Xc[:, 2] > 3) & (Xc[:, 2] < 200)
            err = np.hypot(Xc[:, 0] / Xc[:, 2] - n[:, 0], Xc[:, 1] / Xc[:, 2] - n[:, 1]) * cam.K[0, 0]
            ok &= err < 2.0
        ok &= X[:, 2] < 30          # drop the sky / floodlights far above the track
        pts3d.append(X[ok]); desc.append(di[qi[ok]]); src.append(np.full(ok.sum(), frames[i][0]))
    pts3d, desc, src = np.vstack(pts3d), np.vstack(desc), np.concatenate(src)
    print(f'map: {len(frames)} frames, {len(pts3d)} landmarks, height range {np.percentile(pts3d[:, 2], [5, 95]).round(1)} m')

    # ---- queries from the other lap over the same stretch (by GT position, experiment only)
    q0, q1 = int(laps.loc[a.query_lap, 't0_ns']), int(laps.loc[a.query_lap, 't1_ns'])
    gq = (gt.t >= q0) & (gt.t <= q1)
    pos_map = np.array([gt.pose(t)[:2] for t, _ in frames])
    d = np.min(np.hypot(gt.x[gq][:, None] - pos_map[None, :, 0], gt.y[gq][:, None] - pos_map[None, :, 1]), 1)
    tq = gt.t[gq][d < 2.0]
    queries = read_images(a.bag, a.cam, int(tq.min()), int(tq.max()))
    src_pos = {t: gt.pose(t)[:2] for t, _ in frames}
    rows = []
    for t, im in queries:
        x, y, _, yaw = gt.pose(t)
        near = np.array([np.hypot(*(src_pos[s] - [x, y])) < 40 for s in src])
        if near.sum() < 20:
            continue
        pq, dq = features(sift, im, mask)
        m = [p for p in bf.knnMatch(dq, desc[near], k=2) if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(m) < 8:
            rows.append(dict(t=t, n_match=len(m), n_inl=0)); continue
        img_pts = pq[[x_[0].queryIdx for x_ in m]]
        obj = pts3d[near][[x_[0].trainIdx for x_ in m]]
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img_pts, cam.K, cam.D, reprojectionError=3.0,
                                                 iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            rows.append(dict(t=t, n_match=len(m), n_inl=0 if inl is None else len(inl))); continue
        ok, rvec, tvec = cv2.solvePnP(obj[inl[:, 0]], img_pts[inl[:, 0]], cam.K, cam.D, rvec, tvec, True)
        T_cam_map = np.eye(4); T_cam_map[:3, :3] = cv2.Rodrigues(rvec)[0]; T_cam_map[:3, 3] = tvec.ravel()
        T_map_base = np.linalg.inv(T_cam_map) @ cam.T_cam_base
        ex, ey = T_map_base[0, 3] - x, T_map_base[1, 3] - y
        eyaw = np.degrees(np.angle(np.exp(1j * (np.arctan2(T_map_base[1, 0], T_map_base[0, 0]) - yaw))))
        rows.append(dict(t=t, n_match=len(m), n_inl=len(inl),
                         lon=np.cos(yaw) * ex + np.sin(yaw) * ey, lat=-np.sin(yaw) * ex + np.cos(yaw) * ey,
                         dyaw=eyaw))
    r = pd.DataFrame(rows)
    good = r.dropna(subset=['lat'])
    print(f'queries {len(r)}, solved {len(good)} ({len(good) / max(len(r), 1) * 100:.0f} %), '
          f'median inliers {good.n_inl.median() if len(good) else 0}')
    if len(good):
        for c in ['lat', 'lon', 'dyaw']:
            e = good[c].abs()
            print(f'  |{c}| median {e.median():.2f}  p90 {e.quantile(0.9):.2f}  '
                  f'< 0.5: {np.mean(e < 0.5) * 100:.0f} %  < 1: {np.mean(e < 1) * 100:.0f} %')


if __name__ == '__main__':
    main()
