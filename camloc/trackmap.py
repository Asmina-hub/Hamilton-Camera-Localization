"""Track map prior from /constructor0/track_bounds: drivable area, distance to it, border directions."""
import cv2
import numpy as np
from scipy.spatial import cKDTree


def _tangent_angles(p):
    d = np.gradient(p, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


class TrackMap:
    """Raster of the drivable area (main track ring + pit lane) in the map frame.

    dist_outside(x, y): metres from (x, y) to the drivable area, 0 inside.
    border_dir(x, y):   direction [rad] of the nearest border, defined modulo pi (axial).
    """

    def __init__(self, bounds_npz, res=0.25, margin=30.0):
        b = np.load(bounds_npz)
        self.lines = {k: b[k] for k in b.files}
        allp = np.vstack(list(self.lines.values()))
        self.res = res
        self.x0, self.y0 = allp.min(0) - margin
        x1, y1 = allp.max(0) + margin
        self.w, self.h = int((x1 - self.x0) / res) + 1, int((y1 - self.y0) / res) + 1

        px = lambda p: np.round((p - [self.x0, self.y0]) / res).astype(np.int32).reshape(-1, 1, 2)
        left, right = np.zeros((self.h, self.w), np.uint8), np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(left, [px(self.lines['track_left'])], 1)
        cv2.fillPoly(right, [px(self.lines['track_right'])], 1)
        mask = left ^ right                                   # ring between the two closed borders
        pit = np.zeros_like(mask)
        cv2.fillPoly(pit, [px(np.vstack([self.lines['pitlane_left'], self.lines['pitlane_right'][::-1]]))], 1)
        self.mask = mask | pit
        self.dist = cv2.distanceTransform((1 - self.mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * res

        self.border_pts = allp
        self.border_ang = np.concatenate([_tangent_angles(p) for p in self.lines.values()])
        self.kd = cKDTree(allp)

    def _ij(self, x, y):
        i = np.clip(((y - self.y0) / self.res).astype(int), 0, self.h - 1)
        j = np.clip(((x - self.x0) / self.res).astype(int), 0, self.w - 1)
        return i, j

    def dist_outside(self, x, y):
        i, j = self._ij(np.asarray(x), np.asarray(y))
        return self.dist[i, j]

    def border_dir(self, x, y):
        _, k = self.kd.query(np.c_[x, y])
        return self.border_ang[k]
