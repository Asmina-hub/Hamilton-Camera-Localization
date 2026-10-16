"""Landmark map for relocalization (slide 16, method 3a).

Map: SIFT features of training images, triangulated between frames K apart with the GT poses.
Query: match an image's features to landmarks seen from near the predicted pose, then PnP.
"""
import cv2
import numpy as np

SCALE = 0.75                 # images are downscaled before SIFT (runtime); keypoints are in full-res pixels
N_FEAT = 2000
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def make_sift():
    return cv2.SIFT_create(nfeatures=N_FEAT, contrastThreshold=0.02)


def extract(sift, gray, mask=None):
    """-> keypoints (N, 2) float32 in full-res pixels, descriptors (N, 128) uint8."""
    g = _CLAHE.apply(cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    m = None if mask is None else cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    kp, des = sift.detectAndCompute(g, m)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2) / SCALE
    return pts, np.clip(des, 0, 255).astype(np.uint8)


def pose_T(x, y, z, yaw):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = x, y, z
    return T


def triangulate(cam, T_map_base_i, T_map_base_j, pi, pj, max_err_px=2.0, z_max=30.0):
    """Triangulate matched pixels of two frames. Returns points (M, 3) in map and the kept mask."""
    Pi = np.linalg.inv(T_map_base_i @ cam.T_base_cam)
    Pj = np.linalg.inv(T_map_base_j @ cam.T_base_cam)
    ni = cv2.undistortPoints(pi.reshape(-1, 1, 2), cam.K, cam.D).reshape(-1, 2)
    nj = cv2.undistortPoints(pj.reshape(-1, 1, 2), cam.K, cam.D).reshape(-1, 2)
    X = cv2.triangulatePoints(Pi[:3], Pj[:3], ni.T, nj.T)
    X = (X[:3] / X[3]).T
    ok = np.isfinite(X).all(1)
    for P, n in ((Pi, ni), (Pj, nj)):
        Xc = X @ P[:3, :3].T + P[:3, 3]
        z = np.where(Xc[:, 2] > 1e-6, Xc[:, 2], 1e-6)
        ok &= (Xc[:, 2] > 3) & (Xc[:, 2] < 200)
        ok &= np.hypot(Xc[:, 0] / z - n[:, 0], Xc[:, 1] / z - n[:, 1]) * cam.K[0, 0] < max_err_px
    ok &= X[:, 2] - T_map_base_i[2, 3] < z_max          # drop floodlights high above the track
    return X, ok


class LandmarkMap:
    """pts (N, 3) map frame, desc (N, 128) uint8, src (N, 3) x, y, yaw of the camera pose that saw it."""

    def __init__(self, path, cam):
        d = np.load(path)
        self.pts, self.desc, self.src = d['pts'], d['desc'], d['src']
        self.cam = cam
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def localize(self, pts2d, des, pred, radius=30.0, max_dyaw=np.radians(45), ratio=0.8,
                 reproj=3.0, min_inliers=12):
        """Pose of base_link from one image, using landmarks seen from near the predicted pose (x, y, yaw).

        Returns dict(ok, x, y, yaw, n_match, n_inl)."""
        out = dict(ok=False, x=np.nan, y=np.nan, yaw=np.nan, n_match=0, n_inl=0)
        dx, dy = self.src[:, 0] - pred[0], self.src[:, 1] - pred[1]
        dyaw = np.abs(np.angle(np.exp(1j * (self.src[:, 2] - pred[2]))))
        near = np.where((dx * dx + dy * dy < radius ** 2) & (dyaw < max_dyaw))[0]
        if len(near) < 20 or len(des) < 20:
            return out
        m = self.bf.knnMatch(des.astype(np.float32), self.desc[near].astype(np.float32), k=2)
        m = [p[0] for p in m if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        out['n_match'] = len(m)
        if len(m) < min_inliers:
            return out
        img = pts2d[[x.queryIdx for x in m]].astype(np.float64)
        obj = self.pts[near][[x.trainIdx for x in m]].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.cam.K, self.cam.D, reprojectionError=reproj,
                                                 iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < min_inliers:
            out['n_inl'] = 0 if inl is None else len(inl)
            return out
        inl = inl[:, 0]
        ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], self.cam.K, self.cam.D, rvec, tvec, True)
        T_cam_map = np.eye(4)
        T_cam_map[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_map[:3, 3] = tvec.ravel()
        T = np.linalg.inv(T_cam_map) @ self.cam.T_cam_base
        out.update(ok=True, x=T[0, 3], y=T[1, 3], yaw=np.arctan2(T[1, 0], T[0, 0]), n_inl=len(inl))
        return out


def voxel_dedupe(pts, desc, src, voxel=0.5):
    """Keep one landmark per voxel (the same physical point is triangulated by many frame pairs and laps)."""
    key = np.floor(pts / voxel).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    keep.sort()
    return pts[keep], desc[keep], src[keep]
