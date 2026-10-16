"""Camera measurement: chamfer match of projected map borders against image edges.

For a pose hypothesis, the map borders near the car are put on the flat ground (z = 0 in base_link),
projected into each camera, and scored by the distance transform of the image's edge map. A grid of
lateral / yaw offsets around a reference pose gives a score surface; its minimum is the measurement.
"""
import cv2
import numpy as np

from camloc.camera import map_to_base

TRUNC = 15.0     # px, chamfer truncation (robust to missing / extra edges)


def project_fast(cam, P_base):
    """Vectorised plumb_bob projection of base-link points (..., 3) -> u, v, valid (z > 0.5)."""
    R, t = cam.T_cam_base[:3, :3], cam.T_cam_base[:3, 3]
    Pc = P_base @ R.T + t
    z = Pc[..., 2]
    valid = z > 0.5
    zs = np.where(valid, z, 1.0)
    x, y = Pc[..., 0] / zs, Pc[..., 1] / zs
    k1, k2, p1, p2, k3 = (list(cam.D) + [0] * 5)[:5]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = cam.K[0, 0] * xd + cam.K[0, 2]
    v = cam.K[1, 1] * yd + cam.K[1, 2]
    valid &= (u >= 0) & (u < cam.w - 1) & (v >= 0) & (v < cam.h - 1)
    return u, v, valid


def ground_mask(cam, max_range=40.0, min_range=2.0):
    """Pixels whose ray hits the flat ground between min_range and max_range metres."""
    vv, uu = np.mgrid[0:cam.h, 0:cam.w]
    pts = cv2.undistortPoints(np.c_[uu.ravel(), vv.ravel()].astype(np.float32).reshape(-1, 1, 2),
                              cam.K, cam.D).reshape(-1, 2)
    rays = np.c_[pts, np.ones(len(pts))] @ cam.T_base_cam[:3, :3].T     # ray directions in base_link
    o = cam.T_base_cam[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = -o[2] / rays[:, 2]
    hit = (lam > 0) & np.isfinite(lam)
    rng = np.hypot(o[0] + lam * rays[:, 0], o[1] + lam * rays[:, 1])
    m = hit & (rng > min_range) & (rng < max_range)
    return m.reshape(cam.h, cam.w)


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
N_ORI = 6        # edge orientation bins over [0, pi)


EDGE_CFG = dict(clahe=False, blur=7, lo=20, hi=60, min_len=30)


def edges(img, clahe=None, blur=None, lo=None, hi=None, min_len=None):
    """Canny edges, keeping only long connected curves (lines, kerbs), and the edge orientation (rad, mod pi)."""
    c = dict(EDGE_CFG)
    c.update({k: v for k, v in dict(clahe=clahe, blur=blur, lo=lo, hi=hi, min_len=min_len).items() if v is not None})
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if c['clahe']:
        g = _CLAHE.apply(g)
    g = cv2.GaussianBlur(g, (c['blur'], c['blur']), 0)
    e = cv2.Canny(g, c['lo'], c['hi'])
    if c['min_len']:
        n, lab, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
        keep = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) >= c['min_len']
        keep[0] = False
        e = (keep[lab] * 255).astype(np.uint8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ori = (np.arctan2(gy, gx) + np.pi / 2) % np.pi          # edge direction = gradient direction + 90 deg
    return e, ori


class BorderMatcher:
    def __init__(self, cams, trackmap, static_masks=None, max_range=25.0, step=0.5, mask_range=60.0):
        self.cams, self.tm, self.max_range = cams, trackmap, max_range
        self.valid_px = {}
        for n, c in cams.items():
            m = ground_mask(c, max_range=mask_range)
            if static_masks is not None and n in static_masks:
                m &= ~static_masks[n]
            self.valid_px[n] = m
        # border points resampled every `step` metres
        self.border = trackmap.border_pts
        self.border_dir = np.c_[np.cos(trackmap.border_ang), np.sin(trackmap.border_ang)]

    def dt_image(self, name, img):
        """Oriented distance transforms: (N_ORI, h, w), bin k holds edges with direction near k*pi/N_ORI."""
        e, ori = edges(img)
        e[~self.valid_px[name]] = 0
        b = np.round(ori / np.pi * N_ORI).astype(int) % N_ORI
        dts = np.empty((N_ORI,) + e.shape, np.float32)
        for k in range(N_ORI):
            ek = (e > 0) & ((b == k) | (b == (k + 1) % N_ORI) | (b == (k - 1) % N_ORI))   # +-1 bin tolerance
            dts[k] = np.minimum(cv2.distanceTransform((~ek).astype(np.uint8), cv2.DIST_L2, 3), TRUNC)
        return dts, e

    def score_grid(self, pose, dts, dys, dyaws, min_pts=20):
        """Chamfer score for every (dy, dyaw) offset of `pose` (lateral in the car frame, yaw).

        dts: {camera: truncated distance-transform image}. Returns (score[len(dyaws), len(dys)], n_points).
        Points that leave the image or the ground mask under a hypothesis count as TRUNC.
        """
        x, y, yaw = pose
        near = np.hypot(self.border[:, 0] - x, self.border[:, 1] - y) < self.max_range
        P = self.border[near]
        Pb = map_to_base(P, pose)                                       # (M, 3) at the reference pose
        Pb2 = map_to_base(P + 0.5 * self.border_dir[near], pose)        # 0.5 m along the border
        total = np.zeros((len(dyaws), len(dys)))
        count = 0
        for name, dt in dts.items():
            cam = self.cams[name]
            u0, v0, ok0 = project_fast(cam, Pb)
            u2, v2, ok2 = project_fast(cam, Pb2)
            ok0 &= ok2 & self.valid_px[name][np.clip(v0.astype(int), 0, cam.h - 1), np.clip(u0.astype(int), 0, cam.w - 1)]
            if ok0.sum() < min_pts:
                continue
            Q = Pb[ok0]                                                  # points visible at the reference
            # orientation bin of the projected border at each point (fixed at the reference pose)
            kb = np.round((np.arctan2(v2 - v0, u2 - u0)[ok0] % np.pi) / np.pi * N_ORI).astype(int) % N_ORI
            # hypothesis: the car is at +dy (left) and rotated by dyaw -> points move by the inverse
            c, s = np.cos(-dyaws)[:, None, None], np.sin(-dyaws)[:, None, None]
            qx, qy = Q[None, None, :, 0], Q[None, None, :, 1] - dys[None, :, None]
            hx, hy = c * qx - s * qy, s * qx + c * qy
            H = np.stack([hx, hy, np.zeros_like(hx)], -1)
            u, v, ok = project_fast(cam, H)
            ui, vi = np.clip(u.astype(int), 0, cam.w - 1), np.clip(v.astype(int), 0, cam.h - 1)
            # score only points that stay visible under every hypothesis, so all hypotheses are compared on
            # the same point set (otherwise the reference pose wins because its points are all visible)
            vis = (ok & self.valid_px[name][vi, ui]).all(axis=(0, 1))
            if vis.sum() < min_pts:
                continue
            d = dt[kb[vis], vi[..., vis], ui[..., vis]]
            total += d.sum(-1)
            count += int(vis.sum())
        return (total / count if count else total), count
