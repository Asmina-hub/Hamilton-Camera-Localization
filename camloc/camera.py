"""Camera models from cache/calib.json and projection of map points into images."""
import json
import cv2
import numpy as np

CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']


class Camera:
    def __init__(self, name, c):
        self.name = name
        self.K = np.array(c['K'], float).reshape(3, 3)
        self.D = np.array(c['D'], float)
        self.w, self.h = c['width'], c['height']
        self.T_base_cam = np.array(c['T_base_cam'], float)
        self.T_cam_base = np.linalg.inv(self.T_base_cam)

    def project_base(self, P_base, min_z=0.5):
        """Base-link points (N, 3) -> pixels (M, 2) and the indices of the points that are in view."""
        Pc = P_base @ self.T_cam_base[:3, :3].T + self.T_cam_base[:3, 3]
        ok = Pc[:, 2] > min_z
        if not ok.any():
            return np.zeros((0, 2)), np.zeros(0, int)
        uv, _ = cv2.projectPoints(Pc[ok], np.zeros(3), np.zeros(3), self.K, self.D)
        uv = uv.reshape(-1, 2)
        idx = np.where(ok)[0]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < self.w) & (uv[:, 1] >= 0) & (uv[:, 1] < self.h)
        return uv[inside], idx[inside]


def load_cameras(calib_json):
    c = json.load(open(calib_json))
    return {n: Camera(n, c[n]) for n in CAMS}


def map_to_base(P_xy, pose, ground_z=0.0):
    """Map-frame 2D points -> base_link 3D points on a flat ground plane at height ground_z."""
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    d = P_xy - [x, y]
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], np.full(len(d), ground_z)]
