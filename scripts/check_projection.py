#!/usr/bin/env python3
"""Calibration check: draw the map borders into the images using the GT pose (development only).

    python3 scripts/check_projection.py cache gt.csv out/proj_<t>.jpg <timestamp_ns> [ground_z]

Uses the nearest cached 1 Hz image per camera; the GT pose is interpolated at that image's stamp.
"""
import os, sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras, map_to_base

COLORS = {'track_left': (0, 255, 0), 'track_right': (0, 0, 255), 'pitlane_left': (255, 255, 0),
          'pitlane_right': (255, 0, 255)}
LAYOUT = [['cl', 'fl', 'fr', 'cr'], ['rl', 'r', 'rr', None]]


def gt_pose(gt, t):
    yaw = np.unwrap(gt['yaw'].values)
    return (np.interp(t, gt.timestamp_ns, gt.x), np.interp(t, gt.timestamp_ns, gt.y),
            np.interp(t, gt.timestamp_ns, yaw))


def main(cache, gt_csv, out, t, ground_z=0.0):
    t, ground_z = int(t), float(ground_z)
    cams = load_cameras(f'{cache}/calib.json')
    b = np.load(f'{cache}/track_bounds.npz')
    gt = pd.read_csv(gt_csv)
    rows = []
    for r in LAYOUT:
        tiles = []
        for c in r:
            if c is None:
                tiles.append(np.zeros((364, 753, 3), np.uint8)); continue
            files = np.array([int(p[:-4]) for p in os.listdir(f'{cache}/img/{c}')])
            tc = files[np.argmin(abs(files - t))]
            im = cv2.imread(f'{cache}/img/{c}/{tc}.jpg')
            pose = gt_pose(gt, tc)
            for k in b.files:
                P = b[k]
                near = np.hypot(P[:, 0] - pose[0], P[:, 1] - pose[1]) < 120
                uv, _ = cams[c].project_base(map_to_base(P[near], pose, ground_z))
                for u, v in uv.astype(int):
                    cv2.circle(im, (u, v), 3, COLORS[k], -1)
            im = cv2.resize(im, (753, 364))
            cv2.putText(im, c, (10, 40), 0, 1.2, (0, 255, 255), 3)
            tiles.append(im)
        rows.append(np.hstack(tiles))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('->', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
