#!/usr/bin/env python3
"""Add the keyframe images to an existing keyframe map, for the RoMa fallback (camloc/roma_fallback.py).

    python scripts/add_kf_images.py BAG cache gt.csv maps_no26/kf_fl.npz --cam fl
    -> maps_no26/kf_fl_img.npz

The map stores each keyframe's pose (x, y, yaw of base_link from GT at the keyframe stamp) but not the stamp.
The stamp is recovered by matching that pose to the GT pose at every image stamp of the camera (residual
~0 m for the right frame, consecutive frames are metres apart at speed). No map rebuild is needed, and the
3D keypoints stay exactly those of the map. Images are stored as JPEG at --scale (RoMa works in normalized
coordinates, so the resolution only affects the match quality, not the geometry).
"""
import argparse, os, sys, time
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag'); ap.add_argument('cache'); ap.add_argument('gt'); ap.add_argument('kfmap')
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--scale', type=float, default=0.75, help='image scale stored (1.0 = full 1506 x 728)')
    ap.add_argument('--quality', type=int, default=92)
    ap.add_argument('--out', default=None, help='default: <kfmap stem>_img.npz')
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.kfmap)[0] + '_img.npz'
    tic = time.time()

    kf_pose = np.load(a.kfmap)['kf_pose']                                  # (K, 3) x, y, yaw
    fr = pd.read_csv(f'{a.cache}/frames.csv')
    ts = np.sort(fr[fr.cam == a.cam].timestamp_ns.values)
    g = pd.read_csv(a.gt)
    gyaw = np.unwrap(g['yaw'].values)
    P = np.c_[np.interp(ts, g.timestamp_ns, g.x), np.interp(ts, g.timestamp_ns, g.y),
              np.interp(ts, g.timestamp_ns, gyaw)]
    tree = cKDTree(P[:, :2])
    d, cand = tree.query(kf_pose[:, :2], k=8)
    kf_t = np.zeros(len(kf_pose), np.int64)
    resid = np.zeros(len(kf_pose))
    for i in range(len(kf_pose)):
        dyaw = np.abs(np.angle(np.exp(1j * (P[cand[i], 2] - kf_pose[i, 2]))))
        score = d[i] + 5.0 * dyaw                                        # position + heading must agree
        j = cand[i][np.argmin(score)]
        kf_t[i] = ts[j]; resid[i] = np.hypot(*(P[j, :2] - kf_pose[i, :2]))
    print(f'{len(kf_pose)} keyframes matched to {a.cam} stamps: residual median {np.median(resid):.4f} m, '
          f'max {resid.max():.3f} m', flush=True)
    if np.median(resid) > 0.05:
        print('WARNING: large residuals; is this the right camera / gt.csv?', flush=True)

    need = {}
    for i, t in enumerate(kf_t):
        need.setdefault(int(t), []).append(i)          # several keyframes can share a frame
    blobs = [None] * len(kf_t)
    topic = f'/constructor0/sensor/camera_{a.cam}/compressed_image'
    with open(a.bag, 'rb') as f:
        for _, _, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[topic]):
            t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if t in need:
                img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
                if a.scale != 1.0:
                    img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                for i in need[t]:
                    blobs[i] = enc.tobytes()
    missing = sum(b is None for b in blobs)
    if missing:
        raise SystemExit(f'{missing} keyframe images not found in the bag')
    off = np.concatenate([[0], np.cumsum([len(b) for b in blobs])])
    np.savez(out, img_bytes=np.frombuffer(b''.join(blobs), np.uint8), img_off=off, kf_t=kf_t,
             scale=a.scale, resid=resid.astype(np.float32))
    print(f'-> {out}  ({off[-1] / 2**20:.0f} MB, {time.time() - tic:.0f} s)')


if __name__ == '__main__':
    main()
