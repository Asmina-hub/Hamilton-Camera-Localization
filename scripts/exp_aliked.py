#!/usr/bin/env python3
"""Feasibility: ALIKED + LightGlue keyframe relocalization vs the SIFT experiment (development only).

    python scripts/exp_aliked.py cache gt.csv BAG --map-lap 4 --query-lap 3 --t-from 20 --dur 20

Same setup as scripts/exp_landmarks.py. Map: ALIKED keypoints of every frame of one training lap; frame i
is matched to i+K with LightGlue and the matches are triangulated with GT poses. Query: match an image of
another lap to the N keyframes nearest the GT pose (experiment only), collect 2D-3D pairs, PnP.
"""
import argparse, os, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
from lightglue import ALIKED, LightGlue

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc.landmarks import pose_T, triangulate
from exp_landmarks import read_images, GT

DEV = torch.device('cuda')


class Extractor:
    def __init__(self, cam_name, n_kpts, clahe):
        self.net = ALIKED(max_num_keypoints=n_kpts, detection_threshold=0.0).eval().to(DEV)
        self.mask = body_masks()[cam_name]
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

    @torch.no_grad()
    def __call__(self, gray):
        g = self.clahe.apply(gray) if self.clahe is not None else gray
        img = torch.from_numpy(np.repeat(g[None], 3, 0)).float().div(255).to(DEV)
        f = self.net.extract(img, resize=1024)
        k = f['keypoints'][0].cpu().numpy()
        keep = ~self.mask[np.clip(k[:, 1].astype(int), 0, gray.shape[0] - 1),
                          np.clip(k[:, 0].astype(int), 0, gray.shape[1] - 1)]
        kt = torch.from_numpy(np.nonzero(keep)[0]).to(DEV)
        return {'keypoints': f['keypoints'][:, kt], 'descriptors': f['descriptors'][:, kt],
                'keypoint_scores': f['keypoint_scores'][:, kt], 'image_size': f['image_size']}


@torch.no_grad()
def match(lg, a, b):
    m = lg({'image0': a, 'image1': b})['matches'][0]
    return m[:, 0].cpu().numpy(), m[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('bag')
    ap.add_argument('--map-lap', type=int, default=4); ap.add_argument('--query-lap', type=int, default=3)
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--t-from', type=float, default=20.0); ap.add_argument('--dur', type=float, default=20.0)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--n-kpts', type=int, default=2048)
    ap.add_argument('--n-kf', type=int, default=2, help='keyframes matched per query')
    ap.add_argument('--clahe', type=int, default=1)
    a = ap.parse_args()

    cam = load_cameras(f'{a.cache}/calib.json')[a.cam]
    gt = GT(a.gt)
    laps = pd.read_csv(f'{a.cache}/laps.csv').set_index('lap')
    ex = Extractor(a.cam, a.n_kpts, a.clahe)
    lg = LightGlue(features='aliked').eval().to(DEV)

    # ---- map
    t0 = int(laps.loc[a.map_lap, 't0_ns'] + a.t_from * 1e9); t1 = int(t0 + a.dur * 1e9)
    frames = read_images(a.bag, a.cam, t0, t1)
    feats = [ex(im) for _, im in frames]
    Ts = [pose_T(*gt.pose(t)) for t, _ in frames]
    X3 = [np.full((f['keypoints'].shape[1], 3), np.nan) for f in feats]     # 3D of each keypoint, or NaN
    for i in range(len(frames) - a.k):
        j = i + a.k
        mi, mj = match(lg, feats[i], feats[j])
        if len(mi) < 10:
            continue
        pi = feats[i]['keypoints'][0].cpu().numpy()[mi]
        pj = feats[j]['keypoints'][0].cpu().numpy()[mj]
        X, ok = triangulate(cam, Ts[i], Ts[j], pi, pj)
        X3[i][mi[ok]] = X[ok]
    kf_pos = np.array([[T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])] for T in Ts])
    print(f'map: {len(frames)} keyframes, {sum(np.isfinite(x[:, 0]).sum() for x in X3)} landmarks, '
          f'median kpts {np.median([f["keypoints"].shape[1] for f in feats]):.0f}')

    # ---- queries
    q0, q1 = int(laps.loc[a.query_lap, 't0_ns']), int(laps.loc[a.query_lap, 't1_ns'])
    gq = (gt.t >= q0) & (gt.t <= q1)
    d = np.min(np.hypot(gt.x[gq][:, None] - kf_pos[None, :, 0], gt.y[gq][:, None] - kf_pos[None, :, 1]), 1)
    tq = gt.t[gq][d < 2.0]
    queries = read_images(a.bag, a.cam, int(tq.min()), int(tq.max()))
    rows = []
    for t, im in queries:
        x, y, _, yaw = gt.pose(t)
        torch.cuda.synchronize(); tic = time.perf_counter()
        fq = ex(im)
        torch.cuda.synchronize(); t_ex = time.perf_counter() - tic
        dist = np.hypot(kf_pos[:, 0] - x, kf_pos[:, 1] - y)
        dist[np.abs(np.angle(np.exp(1j * (kf_pos[:, 2] - yaw)))) > np.radians(45)] = np.inf
        img_pts, obj_pts = [], []
        for kf in np.argsort(dist)[:a.n_kf]:
            mq, mk = match(lg, fq, feats[kf])
            has = np.isfinite(X3[kf][mk, 0])
            img_pts.append(fq['keypoints'][0].cpu().numpy()[mq[has]]); obj_pts.append(X3[kf][mk[has]])
        torch.cuda.synchronize(); t_m = time.perf_counter() - tic - t_ex
        img_pts, obj_pts = np.vstack(img_pts).astype(np.float64), np.vstack(obj_pts).astype(np.float64)
        row = dict(t=t, n_match=len(img_pts), n_inl=0, ms_ex=t_ex * 1e3, ms_match=t_m * 1e3)
        if len(img_pts) >= 15:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj_pts, img_pts, cam.K, cam.D, reprojectionError=3.0,
                                                     iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
            if ok and inl is not None and len(inl) >= 15:
                ok, rvec, tvec = cv2.solvePnP(obj_pts[inl[:, 0]], img_pts[inl[:, 0]], cam.K, cam.D, rvec, tvec, True)
                Tcm = np.eye(4); Tcm[:3, :3] = cv2.Rodrigues(rvec)[0]; Tcm[:3, 3] = tvec.ravel()
                T = np.linalg.inv(Tcm) @ cam.T_cam_base
                ex_, ey_ = T[0, 3] - x, T[1, 3] - y
                row.update(n_inl=len(inl), lon=np.cos(yaw) * ex_ + np.sin(yaw) * ey_,
                           lat=-np.sin(yaw) * ex_ + np.cos(yaw) * ey_,
                           dyaw=np.degrees(np.angle(np.exp(1j * (np.arctan2(T[1, 0], T[0, 0]) - yaw)))))
        rows.append(row)
    r = pd.DataFrame(rows)
    good = r.dropna(subset=['lat']) if 'lat' in r else r.iloc[:0]
    print(f'queries {len(r)}, solved {len(good)} ({len(good) / len(r) * 100:.0f} %), median inliers '
          f'{good.n_inl.median():.0f} | ALIKED {r.ms_ex.median():.0f} ms, LightGlue x{a.n_kf} {r.ms_match.median():.0f} ms')
    for c in ['lat', 'lon', 'dyaw']:
        e = good[c].abs()
        print(f'  |{c}| median {e.median():.2f}  p90 {e.quantile(0.9):.2f}  '
              f'< 0.5: {np.mean(e < 0.5) * 100:.0f} %  < 1: {np.mean(e < 1) * 100:.0f} %')


if __name__ == '__main__':
    main()
