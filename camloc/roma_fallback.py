"""RoMa fallback relocalization against the existing keyframe map (used when the ALIKED + LightGlue fix fails).

The keyframe maps (`kf_<cam>.npz`, scripts/build_kf_map.py) store, per keyframe, the pixel positions of the
triangulated keypoints and their 3D points. RoMa matches the query image densely against the keyframe image,
the stored keypoint pixels are looked up in that dense warp, and the resulting 2D-3D pairs go through the same
PnP as KeyframeMap.localize. The keyframe images come from `kf_<cam>_img.npz` (scripts/add_kf_images.py).

Measured in scripts/exp_roma.py (lap 6, fl): RoMa solves 87 % of darkened queries where ALIKED solves 74 %, at
about 470 ms per match on an A100, so it is called only after several frames without an accepted fix.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_roma(model='outdoor', coarse=560, upsample=864, fused_corr=False):
    """One RoMa model shared by all cameras. model: 'outdoor' (RoMa) or 'tiny' (Tiny RoMa, for quick tests)."""
    import romatch
    if model == 'tiny':
        return romatch.tiny_roma_v1_outdoor(device=DEV)
    return romatch.roma_outdoor(device=DEV, coarse_res=coarse, upsample_res=upsample, use_custom_corr=fused_corr)


class KeyframeImages:
    """Keyframe images stored as JPEG bytes: npz with img_bytes (uint8, flat), img_off (K + 1), kf_t (K,)."""

    def __init__(self, path):
        d = np.load(path)
        self.buf, self.off, self.kf_t = d['img_bytes'], d['img_off'], d['kf_t']
        self._cache = {}

    def get(self, k):
        if k not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            jpeg = self.buf[self.off[k]:self.off[k + 1]]
            self._cache[k] = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        return self._cache[k]


class RomaFallback:
    def __init__(self, model, kfmap, kf_images, mask=None, cert_th=0.3):
        self.m, self.kf, self.imgs, self.mask, self.cert_th = model, kfmap, kf_images, mask, cert_th
        self.tiny = not hasattr(model, 'symmetric') or 'Tiny' in type(model).__name__

    @staticmethod
    def _pil(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @torch.inference_mode()
    def _lookup(self, img_r, img_q, pts_norm):
        """Normalized reference coords (N, 2) in [-1, 1] -> normalized query coords (N, 2), certainty (N,)."""
        if self.tiny:
            warp, cert = self.m.match(self._pil(img_r), self._pil(img_q))
        else:
            warp, cert = self.m.match(self._pil(img_r), self._pil(img_q), device=DEV)
        if warp.dim() == 4:
            warp, cert = warp[0], cert[0]
        W = warp.shape[1] // 2 if (not self.tiny and getattr(self.m, 'symmetric', False)) else warp.shape[1]
        ab = warp[:, :W, 2:].permute(2, 0, 1)[None].float()
        cr = cert[:, :W][None, None].float()
        xn = torch.from_numpy(pts_norm).float().to(warp.device)[None, None]
        q = F.grid_sample(ab, xn, align_corners=False, mode='bilinear')[0, :, 0].T
        c = F.grid_sample(cr, xn, align_corners=False, mode='bilinear')[0, 0, 0]
        return q.cpu().numpy(), c.cpu().numpy()

    def localize(self, jpeg, pred, n_kf=1, reproj=3.0, min_inliers=15):
        """Pose of base_link from one query image (JPEG bytes) near the predicted pose (x, y, yaw).

        Same return format as KeyframeMap.localize, plus src='roma'."""
        out = dict(ok=False, x=np.nan, y=np.nan, yaw=np.nan, n_match=0, n_inl=0, kf=-1, src='roma')
        kfs = self.kf.nearest(pred, n_kf)
        if len(kfs) == 0:
            return out
        q = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        w, h = self.kf.image_size
        img, obj = [], []
        for k in kfs:
            s = slice(self.kf.off[k], self.kf.off[k + 1])
            pts, X = self.kf.kpts[s].astype(np.float64), self.kf.X[s]
            if len(pts) < min_inliers:
                continue
            pn = np.c_[2 * pts[:, 0] / w - 1, 2 * pts[:, 1] / h - 1]
            qn, c = self._lookup(self.imgs.get(k), q, pn)
            qp = np.c_[w / 2 * (qn[:, 0] + 1), h / 2 * (qn[:, 1] + 1)]
            ok = (c > self.cert_th) & (qp[:, 0] >= 0) & (qp[:, 0] < w) & (qp[:, 1] >= 0) & (qp[:, 1] < h)
            if self.mask is not None:
                u = np.clip(qp[:, 0].astype(int), 0, w - 1); v = np.clip(qp[:, 1].astype(int), 0, h - 1)
                ok &= ~self.mask[v, u]
            img.append(qp[ok]); obj.append(X[ok])
        if not img:
            return out
        img, obj = np.vstack(img).astype(np.float64), np.vstack(obj).astype(np.float64)
        out.update(n_match=len(img), kf=int(kfs[0]))
        if len(img) < min_inliers:
            return out
        cam = self.kf.cam
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, cam.K, cam.D, reprojectionError=reproj,
                                                 iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < min_inliers:
            out['n_inl'] = 0 if inl is None else len(inl)
            return out
        inl = inl[:, 0]
        ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], cam.K, cam.D, rvec, tvec, True)
        T_cam_map = np.eye(4)
        T_cam_map[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_map[:3, 3] = tvec.ravel()
        T = np.linalg.inv(T_cam_map) @ cam.T_cam_base
        out.update(ok=True, x=T[0, 3], y=T[1, 3], yaw=np.arctan2(T[1, 0], T[0, 0]), n_inl=len(inl))
        return out
