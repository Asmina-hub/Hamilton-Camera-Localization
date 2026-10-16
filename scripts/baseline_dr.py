#!/usr/bin/env python3
"""Baseline 0: dead reckoning from car-state speed and yaw rate, no cameras.

    python3 scripts/baseline_dr.py cache gt.csv out/baseline_dr.csv

Inputs (allowed, slide 9): vx, vy, wz from state_estimation; the first GT pose of the run.
A pose is written at every image-set stamp (camera_fl header.stamp; the cameras are hardware-synced),
using only state messages with stamp <= that image stamp (causal).
"""
import os, sys
import numpy as np
import pandas as pd


def first_pose(gt_csv):
    g = pd.read_csv(gt_csv, nrows=1)
    return g.timestamp_ns[0], g.x[0], g.y[0], g['yaw'][0]


def main(cache, gt_csv, out):
    s = pd.read_csv(f'{cache}/state.csv')
    f = pd.read_csv(f'{cache}/frames.csv')
    img_t = np.sort(f[f.cam == 'fl'].timestamp_ns.values)

    t0, x, y, yaw = first_pose(gt_csv)
    st, vx, vy, wz = s.timestamp_ns.values, s.vx_mps.values, s.vy_mps.values, s.wz_radps.values
    k = np.searchsorted(st, t0, side='right')      # first state message after the initial pose
    t_prev = t0
    rows = []
    for ti in img_t:
        if ti < t0:
            continue
        # integrate state messages up to the image stamp (zero-order hold on the latest state)
        while k < len(st) and st[k] <= ti:
            dt = (st[k] - t_prev) * 1e-9
            j = k - 1 if k > 0 else 0
            yaw_mid = yaw + 0.5 * wz[j] * dt
            x += (vx[j] * np.cos(yaw_mid) - vy[j] * np.sin(yaw_mid)) * dt
            y += (vx[j] * np.sin(yaw_mid) + vy[j] * np.cos(yaw_mid)) * dt
            yaw += wz[j] * dt
            t_prev = st[k]
            k += 1
        # extrapolate from the last state message to the image stamp, without changing the integrator state
        dt = (ti - t_prev) * 1e-9
        j = max(k - 1, 0)
        yo = yaw + wz[j] * dt
        xo = x + (vx[j] * np.cos(yaw) - vy[j] * np.sin(yaw)) * dt
        yy = y + (vx[j] * np.sin(yaw) + vy[j] * np.cos(yaw)) * dt
        rows.append((ti, xo, yy, np.arctan2(np.sin(yo), np.cos(yo))))

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    pd.DataFrame(rows, columns=['timestamp_ns', 'x', 'y', 'yaw']).to_csv(out, index=False, float_format='%.6f')
    print(f'{len(rows)} poses -> {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
