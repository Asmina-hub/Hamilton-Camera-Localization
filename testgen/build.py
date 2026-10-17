"""Build one synthetic test run: <out>/test.mcap, <out>/gt.csv, <out>/manifest.json, <out>/preview.jpg.

The test file looks like the hidden run as far as we know it:
  - one window of the training run (normally one held-out lap), all 7 cameras, car state, calibration
    (camera_info, /tf_static) and track borders; the latched topics are repeated at the start
  - /tf holds only the first pose of the window (the only ground truth allowed at inference)
  - leak guard: pose and track-state fields inside /constructor0/state_estimation are set to NaN, so a
    pipeline that accidentally reads ground truth fails loudly (--keep-state-pose to disable)
  - the transforms of the spec applied (images, timing, calibration, car state, first pose)
The ground truth of the window goes to gt.csv (time-mapped like the file) for tools/evaluate.py.
"""
import json, math, os, struct, time
import multiprocessing as mp
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from mcap.reader import make_reader
from mcap.writer import Writer, CompressionType
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2._dynamic import serialize_dynamic

from testgen.transforms import parse_spec, SpeedUp
from testgen.geometry import set_pose_2d, yaw_of, NS

IMG_PREFIX = f'/{NS}/sensor/camera_'
STATE = f'/{NS}/state_estimation'
BOUNDS = f'/{NS}/track_bounds'
LATCHED = ('/tf_static', BOUNDS)
GATE_Y, GATE_X = 800.0, (20.0, 120.0)        # same lap gate as scripts/split_laps.py


# ------------------------------------------------------------------------------------------------ laps

def laps_from_gt(gt):
    """Lap windows exactly like scripts/split_laps.py: lap k from gate crossing k to k+1 (0 = out-lap)."""
    t, x, y = gt.timestamp_ns.values, gt.x.values, gt.y.values
    up = np.where((y[:-1] < GATE_Y) & (y[1:] >= GATE_Y) & (x[:-1] > GATE_X[0]) & (x[:-1] < GATE_X[1]))[0]
    edges = [0] + list(up + 1) + [len(t) - 1]
    rows = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        kind = 'outlap' if k == 0 else 'inlap' if k == len(edges) - 2 else 'lap'
        rows.append(dict(lap=k, t0_ns=int(t[a]), t1_ns=int(t[b]), kind=kind, duration_s=round((t[b] - t[a]) / 1e9, 1)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ raw CDR helpers

def _endian(data):
    return '<' if data[1] == 1 else '>'


def read_stamp(data):
    e = _endian(data)
    sec, nsec = struct.unpack_from(e + 'iI', data, 4)
    return sec * 1_000_000_000 + nsec


def patch_stamp(data, t_ns):
    """Header stamp is the first field of these messages: bytes 4..12 after the encapsulation header."""
    b = bytearray(data)
    struct.pack_into(_endian(data) + 'iI', b, 4, int(t_ns // 1_000_000_000), int(t_ns % 1_000_000_000))
    return bytes(b)


def replace_image_data(data, jpeg):
    """sensor_msgs/CompressedImage: header(stamp, frame_id) | format | uint8[] data -> new data bytes."""
    e = _endian(data)
    o = 4 + 8                                              # encapsulation + stamp
    for _ in range(2):                                     # frame_id, format (CDR strings, 4-byte aligned)
        o += (-(o - 4)) % 4
        n = struct.unpack_from(e + 'I', data, o)[0]
        o += 4 + n
    o += (-(o - 4)) % 4
    return bytes(data[:o]) + struct.pack(e + 'I', len(jpeg)) + jpeg


# ------------------------------------------------------------------------------------------------ worker

_W = {}


def _init_worker(spec, seed, setup_ctx, jpeg_q):
    cv2.setNumThreads(1)
    _W['tf'] = [t for t in parse_spec(spec, seed) if t.kind == 'image']
    for t in _W['tf']:
        t.setup(setup_ctx)
    _W['q'] = jpeg_q


def _work(item):
    if item[0] != 'img' or not _W['tf']:
        return item
    _, ch_id, log_t, data, cam, t_rel, speed, extra = item
    o = 4 + 8
    jpeg = None
    # locate the jpeg bytes inside the CDR message
    e = _endian(data)
    for _ in range(2):
        o += (-(o - 4)) % 4
        o += 4 + struct.unpack_from(e + 'I', data, o)[0]
    o += (-(o - 4)) % 4
    n = struct.unpack_from(e + 'I', data, o)[0]
    jpeg = data[o + 4:o + 4 + n]
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
    for t in _W['tf']:
        img = t.image(img, cam, t_rel, speed)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, _W['q']])
    return ('img', ch_id, log_t, replace_image_data(data, enc.tobytes()), cam, t_rel, speed, extra)


# ------------------------------------------------------------------------------------------------ builder

class _Codec:
    def __init__(self):
        self.dec, self.enc, self.df = {}, {}, DecoderFactory()

    def decode(self, schema, data):
        if schema.id not in self.dec:
            self.dec[schema.id] = self.df.decoder_for('cdr', schema)
        return self.dec[schema.id](data)

    def encode(self, schema, msg):
        if schema.id not in self.enc:
            self.enc[schema.id] = serialize_dynamic(schema.name, schema.data.decode())[schema.name]
        return self.enc[schema.id](msg)


def _nan_pose(msg):
    nan = float('nan')
    for f in ('x_m', 'y_m', 'z_m', 'roll_rad', 'pitch_rad', 'yaw_rad'):
        setattr(msg, f, nan)
    msg.pos_accuracy = [nan, nan, nan]
    sm = msg.sn_map_state
    sm.current_lane = 0
    for lane in (sm.track_sn_state, sm.pitlane_sn_state):
        lane.is_inside_borders = False
        lane.dist_to_border = nan
        s = lane.sn_state
        s.idx = 0
        s.ds = s.d_idx = s.n = s.epsi = nan
        s.status = 0


def build(bag, gt_csv, out_dir, t0, t1, spec='', seed=0, workers=4, jpeg_quality=92, keep_state_pose=False,
          preview=True, label=''):
    tic = time.time()
    os.makedirs(out_dir, exist_ok=True)
    tfs = parse_spec(spec, seed)
    dur_s = (t1 - t0) / 1e9
    ctx = dict(dur_s=dur_s, t0_ns=t0)
    for t in tfs:
        t.setup(ctx)
    k = next((t.p['k'] for t in tfs if isinstance(t, SpeedUp)), 1.0)
    state_tfs = [t for t in tfs if t.kind in ('state', 'time')]
    calib_tfs = [t for t in tfs if t.kind == 'calib']
    init_tfs = [t for t in tfs if t.kind == 'init']
    has_img_tf = any(t.kind == 'image' for t in tfs)
    tmap = (lambda t: t0 + int(round((t - t0) / k))) if k != 1.0 else (lambda t: t)

    # ---- ground truth of the window
    gt = pd.read_csv(gt_csv)
    g = gt[(gt.timestamp_ns >= t0) & (gt.timestamp_ns <= t1)].copy()
    g['timestamp_ns'] = [tmap(int(t)) for t in g.timestamp_ns]
    g.to_csv(os.path.join(out_dir, 'gt.csv'), index=False)

    codec = _Codec()
    counts = {}
    f = open(bag, 'rb')
    reader = make_reader(f)
    summ = reader.get_summary()
    out_path = os.path.join(out_dir, 'test.mcap')
    fo = open(out_path, 'wb')
    w = Writer(fo, compression=CompressionType.ZSTD)
    w.start(profile='ros2', library='camloc-testgen')
    sid = {s.id: w.register_schema(s.name, s.encoding, s.data) for s in summ.schemas.values()}
    cid = {c.id: w.register_channel(c.topic, c.message_encoding, sid[c.schema_id], dict(c.metadata or {}))
           for c in summ.channels.values()}
    schemas, channels = summ.schemas, summ.channels
    by_topic = {c.topic: c for c in channels.values()}

    def write(ch, log_t, data):
        w.add_message(cid[ch.id], log_time=int(log_t), data=data, publish_time=int(log_t))

    def tf_static_msg(schema, data):
        if not calib_tfs:
            return data
        m = codec.decode(schema, data)
        for t in calib_tfs:
            t.tf_static(m)
        return codec.encode(schema, m)

    # ---- latched topics from just before the window, re-emitted at its start
    last = {}
    for schema, ch, m in reader.iter_messages(start_time=t0 - 12_000_000_000, end_time=t0):
        if ch.topic in LATCHED or ch.topic.endswith('/camera_info'):
            last[ch.topic] = (schema, ch, m.data)
    for topic, (schema, ch, data) in sorted(last.items()):
        if topic == '/tf_static':
            data = tf_static_msg(schema, data)
        elif topic.endswith('/camera_info'):
            data = patch_stamp(data, t0)
        write(ch, t0, data)

    # ---- speeds for speed-dependent transforms (car state, allowed data)
    sp_t, sp_v = [], []
    for schema, ch, m in reader.iter_messages(topics=[STATE], start_time=t0 - 1_000_000_000, end_time=t1):
        msg = codec.decode(schema, m.data)
        sp_t.append(read_stamp(m.data)); sp_v.append(float(msg.v_mps))
    sp_t, sp_v = np.array(sp_t), np.array(sp_v)
    speed_at = lambda t: float(np.interp(t, sp_t, sp_v)) if len(sp_t) else 0.0

    # ---- main pass
    state = dict(first_img=None, first_pose=False)

    def items():
        for schema, ch, m in reader.iter_messages(start_time=t0, end_time=t1):
            topic = ch.topic
            if topic.startswith(IMG_PREFIX) and topic.endswith('compressed_image'):
                cam = topic[len(IMG_PREFIX):].split('/')[0]
                ts = read_stamp(m.data)
                if ts < t0 or ts > t1:
                    continue
                if state['first_img'] is None:
                    state['first_img'] = ts
                idx = int(round((ts - state['first_img']) / 1e8))
                t_rel = (ts - t0) / 1e9
                c = counts.setdefault(cam, dict(kept=0, dropped=0))
                if not all(t.keep_image(cam, t_rel, idx) for t in tfs):
                    c['dropped'] += 1
                    continue
                c['kept'] += 1
                yield ('img', ch.id, m.log_time, m.data, cam, t_rel, speed_at(ts), ts)
            else:
                yield ('raw', ch.id, m.log_time, m.data, schema.id, None, None, None)

    pool = None
    if has_img_tf and workers > 1:
        # fresh ('spawn') workers: a forked worker can hang forever on Linux (OpenCV / BLAS threads of the parent);
        # only images go to the workers, everything else passes through in order; a dead or stuck worker raises
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=_init_worker,
                                   initargs=(spec, seed, ctx, jpeg_quality))
        it = _ordered(pool, items(), ahead=16 * workers, label=label)
    elif has_img_tf:
        _init_worker(spec, seed, ctx, jpeg_quality)
        it = map(_work, items())
    else:
        it = map(_work_passthrough, items())
    n_out = 0
    for item in it:
        kind, ch_id, log_t, data = item[:4]
        ch = channels[ch_id]
        if kind == 'img':
            cam, t_rel, ts = item[4], item[5], item[7]
            off = sum(t.stamp_offset_ns(cam, t_rel) for t in tfs)
            new_ts = tmap(ts) + off
            if new_ts != ts:
                data = patch_stamp(data, new_ts)
            write(ch, tmap(log_t) + off, data)
            n_out += 1
            continue
        schema = schemas[item[4]]
        topic = ch.topic
        if topic == '/tf':
            if state['first_pose']:
                continue
            msg = codec.decode(schema, data)
            tr = [x for x in msg.transforms if x.header.frame_id == 'map' and x.child_frame_id == f'{NS}/base_link']
            if not tr:
                continue
            x0 = tr[0]
            px, py, pyaw = x0.transform.translation.x, x0.transform.translation.y, yaw_of(x0.transform.rotation)
            for t in init_tfs:
                px, py, pyaw = t.init_pose(px, py, pyaw)
            set_pose_2d(x0.transform, px, py, pyaw)
            t_first = x0.header.stamp.sec * 1_000_000_000 + x0.header.stamp.nanosec
            tm = tmap(t_first)
            x0.header.stamp.sec, x0.header.stamp.nanosec = tm // 1_000_000_000, tm % 1_000_000_000
            msg.transforms = [x0]
            write(ch, tmap(log_t), codec.encode(schema, msg))
            state['first_pose'] = True
            state['first_pose_xyyaw'] = (px, py, pyaw, tm)
        elif topic == STATE:
            msg = codec.decode(schema, data)
            ts = read_stamp(data)
            for t in state_tfs:
                t.state(msg, (ts - t0) / 1e9)
            if not keep_state_pose:
                _nan_pose(msg)
            tm = tmap(ts)
            msg.header.stamp.sec, msg.header.stamp.nanosec = tm // 1_000_000_000, tm % 1_000_000_000
            write(ch, tmap(log_t), codec.encode(schema, msg))
        elif topic == '/tf_static':
            write(ch, tmap(log_t), tf_static_msg(schema, data))
        elif topic.endswith('/camera_info'):
            write(ch, tmap(log_t), patch_stamp(data, tmap(read_stamp(data))) if k != 1.0 else data)
        else:                                               # track_bounds and anything else: copy
            write(ch, tmap(log_t), data)
        n_out += 1
    if pool:
        pool.shutdown()
    w.finish(); fo.close(); f.close()

    manifest = dict(
        label=label, created=time.strftime('%Y-%m-%d %H:%M:%S'), source=os.path.abspath(bag),
        window=dict(t0_ns=t0, t1_ns=t1, dur_s=round(dur_s, 2), dur_out_s=round(dur_s / k, 2)),
        spec=spec, seed=seed, speedup=k, jpeg_quality=jpeg_quality if has_img_tf else 'original bytes',
        leak_guard=not keep_state_pose, transforms=[t.describe() for t in tfs], images=counts,
        first_pose=dict(zip(['x', 'y', 'yaw', 't_ns'], state.get('first_pose_xyyaw', ()))),
        gt_rows=len(g), messages_out=n_out, size_mb=round(os.path.getsize(out_path) / 2**20, 1),
        build_s=round(time.time() - tic, 1))
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as fm:
        json.dump(manifest, fm, indent=1, default=str)
    if preview and has_img_tf:
        make_preview(bag, t0, t1, spec, seed, ctx, os.path.join(out_dir, 'preview.jpg'), speed_at)
    return manifest


def _work_passthrough(item):
    return item


def _ordered(pool, items, ahead=64, timeout_s=600, label=''):
    """Images through the worker pool, other messages straight through, output in input order."""
    q, n, tic = deque(), 0, time.time()

    def out(x):
        nonlocal n
        if isinstance(x, Future):
            x = x.result(timeout=timeout_s)          # TimeoutError / BrokenProcessPool instead of a silent hang
            n += 1
            if n % 2000 == 0:
                print(f'  [{label}] {n} images, {n / (time.time() - tic):.0f} img/s', flush=True)
        return x

    for item in items:
        q.append(pool.submit(_work, item) if item[0] == 'img' else item)
        while len(q) > ahead:
            yield out(q.popleft())
    while q:
        yield out(q.popleft())


def make_preview(bag, t0, t1, spec, seed, ctx, path, speed_at, frac=0.35):
    """Original (left) vs transformed (right) for all 7 cameras at one moment of the window."""
    tq = t0 + int((t1 - t0) * frac)
    tfs = [t for t in parse_spec(spec, seed) if t.kind == 'image']
    for t in tfs:
        t.setup(ctx)
    got = {}
    with open(bag, 'rb') as f:
        for schema, ch, m in make_reader(f).iter_messages(start_time=tq, end_time=tq + 2_000_000_000):
            if ch.topic.startswith(IMG_PREFIX) and ch.topic.endswith('compressed_image'):
                cam = ch.topic[len(IMG_PREFIX):].split('/')[0]
                if cam not in got:
                    got[cam] = (read_stamp(m.data), m.data)
            if len(got) == 7:
                break
    rows = []
    for cam in ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']:
        if cam not in got:
            continue
        ts, data = got[cam]
        e = _endian(data); o = 12
        for _ in range(2):
            o += (-(o - 4)) % 4
            o += 4 + struct.unpack_from(e + 'I', data, o)[0]
        o += (-(o - 4)) % 4
        n = struct.unpack_from(e + 'I', data, o)[0]
        img = cv2.imdecode(np.frombuffer(data[o + 4:o + 4 + n], np.uint8), cv2.IMREAD_COLOR)
        out = img.copy()
        for t in tfs:
            out = t.image(out, cam, (ts - t0) / 1e9, speed_at(ts))
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        a, b = cv2.resize(img, (602, 291)), cv2.resize(out, (602, 291))
        for im, lab in ((a, f'{cam} original'), (b, f'{cam} test')):
            cv2.putText(im, lab, (8, 26), 0, 0.8, (0, 255, 255), 2)
        rows.append(np.hstack([a, b]))
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
