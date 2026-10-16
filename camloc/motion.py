"""Car-state odometry: body-frame motion increments between consecutive image stamps."""
import numpy as np
import pandas as pd


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def increments(state_csv, img_t, t0):
    """Integrate vx, vy, wz from state messages with stamp <= each image stamp (causal).

    Returns a DataFrame per image stamp t_k (k >= 1, t_k > t0) with the motion since the previous output
    stamp, expressed in the body frame at the previous stamp: dx, dy [m], dyaw [rad], dt [s], dist [m].
    The first row is the motion from t0 (first pose) to the first image stamp.
    """
    s = pd.read_csv(state_csv, usecols=['timestamp_ns', 'vx_mps', 'vy_mps', 'wz_radps'])
    st, vx, vy, wz = s.timestamp_ns.values, s.vx_mps.values, s.vy_mps.values, s.wz_radps.values
    img_t = img_t[img_t > t0]
    out = np.zeros((len(img_t), 5))
    k = np.searchsorted(st, t0, side='right') - 1       # latest state at or before t0
    t_prev = t0
    for n, ti in enumerate(img_t):
        x = y = yaw = dist = 0.0
        t_from = t_prev
        while True:
            t_next = min(st[k + 1], ti) if k + 1 < len(st) else ti
            dt = (t_next - t_prev) * 1e-9
            j = max(k, 0)
            ym = yaw + 0.5 * wz[j] * dt
            x += (vx[j] * np.cos(ym) - vy[j] * np.sin(ym)) * dt
            y += (vx[j] * np.sin(ym) + vy[j] * np.cos(ym)) * dt
            yaw += wz[j] * dt
            dist += abs(vx[j]) * dt
            t_prev = t_next
            if k + 1 < len(st) and st[k + 1] <= ti:
                k += 1
            if t_prev >= ti:
                break
        out[n] = x, y, yaw, (ti - t_from) * 1e-9, dist
    df = pd.DataFrame(out, columns=['dx', 'dy', 'dyaw', 'dt', 'dist'])
    df.insert(0, 'timestamp_ns', img_t)
    return df
