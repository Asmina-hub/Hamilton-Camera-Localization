#!/usr/bin/env python3
"""Relocalize every image of one camera against the landmark map (development speed-up).

    python3 scripts/precompute_reloc.py BAG cache cache/landmarks_fl.npz out/map_pf.csv cache/reloc_fl.csv --cam fl

The predicted pose that selects nearby landmarks comes from an earlier causal run (pred_csv, e.g. the
no-camera filter), interpolated at the image stamp. The online pipeline uses the filter's own prediction
instead; the selection radius (30 m) makes the result insensitive to that difference.
"""
import argparse, os, sys, time
from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from camloc.camera import load_cameras
from camloc.body_masks import body_masks
from camloc import landmarks as L

_W = {}


def _init(cam_name):
    cv2.setNumThreads(1)          # one process per core; avoid oversubscribing the CPU
    _W['sift'] = L.make_sift()
    _W['mask'] = (~body_masks()[cam_name]).astype(np.uint8) * 255


def _features(item):
    """Worker (CPU): decode + SIFT. Matching and PnP run in the main process, which owns the GPU."""
    t, jpeg, pred = item
    tic = time.perf_counter()
    g = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    p, d = L.extract(_W['sift'], g, _W['mask'])
    return t, p, d, pred, (time.perf_counter() - tic) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('cache'); ap.add_argument('map'); ap.add_argument('pred'); ap.add_argument('out')
    ap.add_argument('--cam', default='fl')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    pr = pd.read_csv(a.pred)
    pyaw = np.unwrap(pr['yaw'].values)
    pred = lambda t: (np.interp(t, pr.timestamp_ns, pr.x), np.interp(t, pr.timestamp_ns, pr.y),
                      np.interp(t, pr.timestamp_ns, pyaw))

    def items():
        with open(a.bag, 'rb') as f:
            r = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in r.iter_decoded_messages(topics=[f'/constructor0/sensor/camera_{a.cam}/compressed_image']):
                t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                yield t, bytes(msg.data), pred(t)

    lmap = L.LandmarkMap(a.map, load_cameras(f'{a.cache}/calib.json')[a.cam])
    rows, tic = [], time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.cam,)) as pool:
        for i, (t, p, d, pr_, ms_feat) in enumerate(pool.imap(_features, items(), chunksize=4)):
            t1 = time.perf_counter()
            r = lmap.localize(p, d, pr_)
            r.update(timestamp_ns=t, ms_feat=ms_feat, ms_loc=(time.perf_counter() - t1) * 1e3)
            rows.append(r)
            if i % 1000 == 0:
                ok = np.mean([x['ok'] for x in rows]) * 100
                print(f'{i} frames, {ok:.0f} % solved, {i / (time.time() - tic + 1e-9):.1f} fps', flush=True)
    df = pd.DataFrame(rows)[['timestamp_ns', 'ok', 'x', 'y', 'yaw', 'n_match', 'n_inl', 'ms_feat', 'ms_loc']]
    df['ok'] = df.ok.astype(int)
    df.to_csv(a.out, index=False, float_format='%.6f')
    print(f'{len(df)} frames -> {a.out}  solved {df.ok.mean() * 100:.0f} %  features {df.ms_feat.median():.0f} ms/worker, '
          f'localize {df.ms_loc.median():.0f} ms (median)')


if __name__ == '__main__':
    main()
