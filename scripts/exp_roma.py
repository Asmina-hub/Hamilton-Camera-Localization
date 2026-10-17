#!/usr/bin/env python3
"""Matcher shoot-out: ALIKED + LightGlue (current pipeline) vs RoMa vs Tiny RoMa for absolute relocalization,
on clean and perturbed query images (development only, uses GT to pick reference frames and to score).

    python scripts/exp_roma.py BAG cache gt.csv --query-lap 6 --matchers aliked,roma,tiny_roma \
        --perturb "clean;@dark;lamps_off;@day_proxy;@blur_only" --out out/exp_roma

For every N-th query image of the held-out lap (default lap 6, camera fl):
  r1 = nearest training-lap frame by GT pose (laps 2, 3, 6, 7 excluded), r2 = the frame k later (0.3 s);
  r1 <-> r2 matches are triangulated with the GT poses (exactly how the keyframe maps are built);
  query <-> r1 matches give 2D-3D pairs -> PnP + RANSAC -> base_link pose -> error vs GT.
The same reference frames and 3D construction are used for every matcher; only the query is perturbed
(testgen transforms), like a hidden run in different conditions matched against a night map.

Output: <out>/per_query.csv, <out>/summary.csv, <out>/summary.md, <out>/vis_*.jpg
"""
import argparse, os, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from camloc.camera import load_cameras                       # noqa: E402
from camloc.body_masks import body_masks                     # noqa: E402
from camloc.landmarks import pose_T, triangulate             # noqa: E402
from testgen.transforms import parse_spec                    # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXCLUDE = {2, 3, 6, 7}                                       # held-out tuning + validation laps


def sync():
    if DEV.type == 'cuda':
        torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------- matchers

class AlikedLG:
    """The current pipeline's matcher (camloc.features: ALIKED 2048 kpts + LightGlue, CLAHE)."""
    name = 'aliked'

    def __init__(self, a):
        from camloc.features import Features
        self.f = Features()
        self.cache = {}

    def _feat(self, key, gray, mask):
        if key is None or key not in self.cache:
            f = self.f.extract(gray, mask)
            if key is None:
                return f
            self.cache[key] = f
        return self.cache[key]

    def pair(self, key_a, img_a, mask_a, key_b, img_b, mask_b):
        """-> pixel arrays pa (N,2), pb (N,2) of matches a <-> b."""
        fa = self._feat(key_a, cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), mask_a)
        fb = self._feat(key_b, cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), mask_b)
        ia, ib = self.f.match(fa, fb)
        ka, kb = fa['keypoints'][0].cpu().numpy(), fb['keypoints'][0].cpu().numpy()
        return ka[ia], kb[ib]

    def lookup(self, key_r, img_r, mask_r, pts_r, key_q, img_q, mask_q):
        """Query pixels for reference pixels pts_r (only exact keypoint matches) -> (idx into pts_r, q pixels)."""
        pr, pq = self.pair(key_r, img_r, mask_r, key_q, img_q, mask_q)
        if len(pr) == 0 or len(pts_r) == 0:
            return np.zeros(0, int), np.zeros((0, 2))
        d = np.linalg.norm(pts_r[:, None] - pr[None], axis=2)      # pts_r are a subset of r's keypoints
        j = d.argmin(1)
        ok = d[np.arange(len(pts_r)), j] < 0.5
        return np.where(ok)[0], pq[j[ok]]


class Roma:
    """RoMa / Tiny RoMa dense matching (romatch). Reference 3D at sampled r1 pixels, query looked up in the warp."""

    def __init__(self, a, tiny=False):
        import romatch
        self.tiny = tiny
        self.name = 'tiny_roma' if tiny else 'roma'
        if tiny:
            self.m = romatch.tiny_roma_v1_outdoor(device=DEV)
        else:
            kw = dict(coarse_res=a.roma_coarse, upsample_res=a.roma_upsample)
            self.m = romatch.roma_outdoor(device=DEV, **kw)
        self.num, self.cert_th = a.num, a.cert_th

    @staticmethod
    def _pil(img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _warp_AB(self, img_a, img_b):
        warp, cert = self.m.match(self._pil(img_a), self._pil(img_b), device=DEV) if not self.tiny else \
            self.m.match(self._pil(img_a), self._pil(img_b))
        if warp.dim() == 4:
            warp, cert = warp[0], cert[0]
        return warp, cert

    def pair(self, key_a, img_a, mask_a, key_b, img_b, mask_b):
        warp, cert = self._warp_AB(img_a, img_b)
        m, c = self.m.sample(warp, cert, num=self.num)
        m = m[c > self.cert_th]
        ha, wa = img_a.shape[:2]; hb, wb = img_b.shape[:2]
        pa, pb = self.m.to_pixel_coordinates(m, ha, wa, hb, wb)
        pa, pb = pa.float().cpu().numpy(), pb.float().cpu().numpy()
        keep = np.ones(len(pa), bool)
        for p, mk, (h, w) in ((pa, mask_a, (ha, wa)), (pb, mask_b, (hb, wb))):
            if mk is not None:
                u = np.clip(p[:, 0].astype(int), 0, w - 1); v = np.clip(p[:, 1].astype(int), 0, h - 1)
                keep &= ~mk[v, u]
        return pa[keep], pb[keep]

    def lookup(self, key_r, img_r, mask_r, pts_r, key_q, img_q, mask_q):
        """Dense: sample the r -> q warp at every reference pixel that has 3D."""
        if len(pts_r) == 0:
            return np.zeros(0, int), np.zeros((0, 2))
        warp, cert = self._warp_AB(img_r, img_q)
        W = warp.shape[1] // 2 if (not self.tiny and getattr(self.m, 'symmetric', False)) else warp.shape[1]
        ab = warp[:, :W, 2:].permute(2, 0, 1)[None].float()          # (1, 2, H, W): r pixel -> q coords
        cr = cert[:, :W][None, None].float()
        h, w = img_r.shape[:2]
        xn = torch.from_numpy(np.c_[2 * pts_r[:, 0] / w - 1, 2 * pts_r[:, 1] / h - 1]).float().to(warp.device)
        q = F.grid_sample(ab, xn[None, None], align_corners=False, mode='bilinear')[0, :, 0].T
        c = F.grid_sample(cr, xn[None, None], align_corners=False, mode='bilinear')[0, 0, 0]
        hq, wq = img_q.shape[:2]
        qp = torch.stack((wq / 2 * (q[:, 0] + 1), hq / 2 * (q[:, 1] + 1)), -1).cpu().numpy()
        ok = (c > self.cert_th).cpu().numpy()
        if mask_q is not None:
            u = np.clip(qp[:, 0].astype(int), 0, wq - 1); v = np.clip(qp[:, 1].astype(int), 0, hq - 1)
            ok &= ~mask_q[v, u]
        return np.where(ok)[0], qp[ok]


# ------------------------------------------------------------------------------------------- geometry

def pnp(cam, obj, img, reproj=3.0, min_inl=15):
    if len(obj) < min_inl:
        return None, 0
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj.astype(np.float64), img.astype(np.float64), cam.K, cam.D,
                                             reprojectionError=reproj, iterationsCount=500,
                                             flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inl:
        return None, 0 if inl is None else len(inl)
    inl = inl[:, 0]
    ok, rvec, tvec = cv2.solvePnP(obj[inl].astype(np.float64), img[inl].astype(np.float64), cam.K, cam.D,
                                  rvec, tvec, True)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
    Tb = np.linalg.inv(T) @ cam.T_cam_base
    return (Tb[0, 3], Tb[1, 3], np.arctan2(Tb[1, 0], Tb[0, 0])), len(inl)


# ------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag'); ap.add_argument('cache'); ap.add_argument('gt')
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--query-lap', type=int, default=6)
    ap.add_argument('--every', type=int, default=10, help='use every N-th query image (10 = 1 per second)')
    ap.add_argument('--max-queries', type=int, default=0)
    ap.add_argument('--k', type=int, default=3, help='r2 = r1 + k frames (triangulation baseline)')
    ap.add_argument('--max-dist', type=float, default=3.0); ap.add_argument('--max-dyaw', type=float, default=15.0)
    ap.add_argument('--matchers', default='aliked,roma')
    ap.add_argument('--perturb', default='clean;@dark;lamps_off;@day_proxy;@blur_only',
                    help='";"-separated testgen specs applied to the query image ("clean" = none)')
    ap.add_argument('--roma-coarse', type=int, default=560, help='RoMa coarse resolution (multiple of 14)')
    ap.add_argument('--roma-upsample', type=int, default=864)
    ap.add_argument('--num', type=int, default=5000, help='RoMa matches sampled for triangulation')
    ap.add_argument('--cert-th', type=float, default=0.3)
    ap.add_argument('--vis', type=int, default=3, help='save match pictures for the first N queries')
    ap.add_argument('--out', default='out/exp_roma')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cam = load_cameras(f'{a.cache}/calib.json')[a.cam]
    mask = body_masks()[a.cam]
    g = pd.read_csv(a.gt); gyaw = np.unwrap(g['yaw'].values)
    gp = lambda t: np.array([np.interp(t, g.timestamp_ns, g.x), np.interp(t, g.timestamp_ns, g.y),
                             np.interp(t, g.timestamp_ns, g.z), np.interp(t, g.timestamp_ns, gyaw)])
    gv = np.hypot(np.gradient(g.x.values), np.gradient(g.y.values)) / np.maximum(np.gradient(g.timestamp_ns.values) / 1e9, 1e-3)
    speed = lambda t: float(np.interp(t, g.timestamp_ns, gv))
    laps = pd.read_csv(f'{a.cache}/laps.csv')
    fr = pd.read_csv(f'{a.cache}/frames.csv')
    ts = np.sort(fr[fr.cam == a.cam].timestamp_ns.values)
    lap_of = np.full(len(ts), -1)
    for r in laps.itertuples():
        lap_of[(ts >= r.t0_ns) & (ts <= r.t1_ns)] = r.lap
    P = np.array([gp(t) for t in ts])

    # ---- queries and their reference pairs (GT used only to pick references and to score)
    qi = np.where(lap_of == a.query_lap)[0][::a.every]
    ref_ok = ~np.isin(lap_of, list(EXCLUDE | {a.query_lap})) & (lap_of >= 0)
    pairs = []
    for i in qi:
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        dy = np.abs(np.degrees(np.angle(np.exp(1j * (P[:, 3] - P[i, 3])))))
        cand = np.where(ref_ok & (d < a.max_dist) & (dy < a.max_dyaw))[0]
        cand = cand[cand + a.k < len(ts)]
        cand = cand[(ts[cand + a.k] - ts[cand]) < (a.k + 0.5) * 1e8]          # no dropped frame in between
        if len(cand) == 0:
            continue
        j = cand[np.argmin(d[cand])]
        pairs.append((ts[i], ts[j], ts[j + a.k], d[j]))
        if a.max_queries and len(pairs) >= a.max_queries:
            break
    print(f'{len(pairs)} queries from lap {a.query_lap} (of {len(qi)}), median reference distance '
          f'{np.median([p[3] for p in pairs]):.2f} m', flush=True)

    need = {t for p in pairs for t in p[:3]}
    imgs = {}
    topic = f'/constructor0/sensor/camera_{a.cam}/compressed_image'
    with open(a.bag, 'rb') as f:
        for _, _, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[topic]):
            t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if t in need:
                imgs[t] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    print(f'{len(imgs)} images loaded', flush=True)

    specs = [s.strip() for s in a.perturb.split(';') if s.strip()]
    tfs = {s: [t for t in parse_spec('' if s == 'clean' else s) if t.kind == 'image'] for s in specs}
    for s in specs:
        for t in tfs[s]:
            t.setup(dict(dur_s=200.0, t0_ns=0))

    rows = []
    for mname in a.matchers.split(','):
        tic = time.time()
        M = AlikedLG(a) if mname == 'aliked' else Roma(a, tiny=(mname == 'tiny_roma'))
        print(f'== {mname} loaded ({time.time() - tic:.0f} s)', flush=True)
        for n, (tq, t1, t2, dref) in enumerate(pairs):
            if tq not in imgs or t1 not in imgs or t2 not in imgs:
                continue
            # 3D at reference pixels: r1 <-> r2 triangulated with GT (same for every perturbation)
            sync(); t_a = time.perf_counter()
            p1, p2 = M.pair(('r1', t1), imgs[t1], mask, ('r2', t2), imgs[t2], mask)
            sync(); ms_ref = (time.perf_counter() - t_a) * 1e3
            X, ok = triangulate(cam, pose_T(*gp(t1)), pose_T(*gp(t2)), p1.astype(np.float32), p2.astype(np.float32)) \
                if len(p1) >= 8 else (np.zeros((0, 3)), np.zeros(0, bool))
            pts_r, X = p1[ok], X[ok]
            x, y, z, yaw = gp(tq)
            for s in specs:
                q = imgs[tq]
                for t in tfs[s]:
                    q = t.image(q, a.cam, (tq - pairs[0][0]) / 1e9, speed(tq))
                sync(); t_b = time.perf_counter()
                idx, qp = M.lookup(('r1', t1), imgs[t1], mask, pts_r, None, q, mask)
                sync(); ms_q = (time.perf_counter() - t_b) * 1e3
                pose, n_inl = pnp(cam, X[idx], qp)
                row = dict(matcher=mname, perturb=s, t=tq, ref_dist=dref, n_3d=len(X), n_corr=len(idx),
                           n_inl=n_inl, ok=pose is not None, ms_query=ms_q, ms_ref=ms_ref)
                if pose is not None:
                    ex, ey = pose[0] - x, pose[1] - y
                    row.update(lat=-np.sin(yaw) * ex + np.cos(yaw) * ey, lon=np.cos(yaw) * ex + np.sin(yaw) * ey,
                               dyaw=np.degrees(np.angle(np.exp(1j * (pose[2] - yaw)))))
                rows.append(row)
                if n < a.vis and s in (specs[0], specs[-1]) and len(idx):
                    _vis(f'{a.out}/vis_{mname}_{s.strip("@").replace(":", "_")[:20]}_{n}.jpg', imgs[t1], q, pts_r[idx], qp)
            if n % 20 == 0:
                print(f'  {mname} {n}/{len(pairs)}', flush=True)
        del M
        if DEV.type == 'cuda':
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(f'{a.out}/per_query.csv', index=False)

    r = pd.DataFrame(rows)
    for c in ('lat', 'lon', 'dyaw'):
        if c not in r:
            r[c] = np.nan
    r.to_csv(f'{a.out}/per_query.csv', index=False)
    gr = r.groupby(['matcher', 'perturb'], sort=False)
    pos = lambda d: np.hypot(d.lat, d.lon)
    s = pd.DataFrame({
        'queries': gr.size(),
        'solved_%': (gr.ok.mean() * 100).round(1),
        'inliers_med': gr.n_inl.median(),
        'lat_med_m': gr.apply(lambda d: d.lat.abs().median()).round(3),
        'lon_med_m': gr.apply(lambda d: d.lon.abs().median()).round(3),
        'pos_p90_m': gr.apply(lambda d: pos(d).quantile(0.9)).round(2),
        'within_1m_%': gr.apply(lambda d: (pos(d) < 1).sum() / len(d) * 100).round(1),
        'yaw_med_deg': gr.apply(lambda d: d.dyaw.abs().median()).round(3),
        'ms_per_match': gr.ms_query.median().round(0),
    }).reset_index()
    s.to_csv(f'{a.out}/summary.csv', index=False)
    try:
        md = s.to_markdown(index=False)
    except ImportError:
        md = s.to_string(index=False)
    with open(f'{a.out}/summary.md', 'w') as f:
        f.write(f'Query lap {a.query_lap}, camera {a.cam}, {len(pairs)} queries, device {DEV}. Errors are over solved '
                f'fixes; within_1m_% counts unsolved queries as failures.\n\n{md}\n')
    print(md)


def _vis(path, ref, q, pr, pq, n=150):
    h = max(ref.shape[0], q.shape[0])
    canvas = np.zeros((h, ref.shape[1] + q.shape[1], 3), np.uint8)
    canvas[:ref.shape[0], :ref.shape[1]] = ref
    canvas[:q.shape[0], ref.shape[1]:] = q if q.ndim == 3 else cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    sel = np.random.default_rng(0).choice(len(pr), min(n, len(pr)), replace=False)
    for i in sel:
        a = tuple(int(v) for v in pr[i]); b = (int(pq[i, 0]) + ref.shape[1], int(pq[i, 1]))
        cv2.line(canvas, a, b, (0, 255, 0), 1)
    cv2.imwrite(path, cv2.resize(canvas, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    main()
