"""Perturbations for synthetic test runs. Each one is separate, has its own parameters, and can be combined.

Spec syntax (testgen.transforms.parse_spec):
    name                      default parameters
    name:key=val,key=val      parameters; lists use '|' (cams=fl|rl)
    a+b+c                     several transforms, applied in this order
    @preset                   expands to a named preset from testgen/presets.py

All transforms are deterministic functions of (seed, camera, time), so a test file can be rebuilt exactly and
image work can run in parallel worker processes.

Kinds of hooks (a transform implements only the ones it needs):
    keep_image(cam, t_rel, set_idx) -> bool     drop images (frame drops, camera failure, blackout)
    image(img, cam, t_rel, speed) -> img        change pixels (lighting, blur, occlusion, ...)
    stamp_offset_ns(cam, t_rel) -> int          shift a camera's header stamps (timing errors)
    state(msg, t_rel)                           change car-state fields in place
    tf_static(msg)                              change the calibration the pipeline reads
    init_pose(x, y, yaw) -> (x, y, yaw)         change the first pose given to the pipeline
    speedup factor (class SpeedUp)              time-compress the whole run (handled by the builder)
"""
import math
import zlib

import cv2
import numpy as np

CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']
SIDE_CAMS = {'cl', 'cr', 'rl', 'rr'}
REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def _parse_value(v):
    if '|' in v:
        return [_parse_value(x) for x in v.split('|') if x != '']
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    return v


class Transform:
    name = ''
    kind = ''
    help = ''
    params = {}

    def __init__(self, seed=0, index=0, **kw):
        unknown = set(kw) - set(self.params)
        if unknown:
            raise ValueError(f'{self.name}: unknown parameter(s) {sorted(unknown)}; allowed {sorted(self.params)}')
        self.p = dict(self.params)
        self.p.update(kw)
        self.given = set(kw)
        if 'cams' in self.p and isinstance(self.p['cams'], str):
            self.p['cams'] = CAMS if self.p['cams'] == 'all' else [self.p['cams']]
        self.seed = (zlib.crc32(f'{self.name}/{index}'.encode()) ^ (seed * 2654435761)) & 0xFFFFFFFF

    def rng(self, *key):
        """Deterministic RNG for a (camera, time, ...) key."""
        k = [self.seed] + [zlib.crc32(str(x).encode()) for x in key]
        return np.random.default_rng(k)

    def applies(self, cam):
        return 'cams' not in self.p or cam in self.p['cams']

    def describe(self):
        return {'name': self.name, 'kind': self.kind, 'params': self.p}

    def setup(self, ctx):
        """Called once by the builder with dict(dur_s=..., t0_ns=...) of the test window."""

    # hooks with no effect by default
    def keep_image(self, cam, t_rel, set_idx):
        return True

    def image(self, img, cam, t_rel, speed):
        return img

    def stamp_offset_ns(self, cam, t_rel):
        return 0

    def state(self, msg, t_rel):
        pass

    def tf_static(self, msg):
        pass

    def init_pose(self, x, y, yaw):
        return x, y, yaw


# ======================================================================================= speed (higher speeds)

@register
class SpeedUp(Transform):
    name, kind = 'speedup', 'time'
    help = ('Physically consistent faster run: keep image sets so that the car moves k times farther between '
            'frames, compress every timestamp by k (camera rate stays ~10 Hz), scale car-state speeds / yaw '
            'rates by k and accelerations by k^2. Non-integer k gives some stamp jitter (e.g. 1.25: 80/80/80/160 ms).')
    params = {'k': 1.25}
    VEL = ['vx_mps', 'vy_mps', 'vz_mps', 'wx_radps', 'wy_radps', 'wz_radps', 'omega_w_fl', 'omega_w_fr',
           'omega_w_rl', 'omega_w_rr', 'v_mps', 'v_raw_mps', 'yaw_vel_rad', 'dbeta_radps', 'rpm']
    ACC = ['ax_mps2', 'ay_mps2', 'az_mps2', 'ax_vel_mps2', 'ay_vel_mps2', 'ddyaw_radps2']

    def keep_image(self, cam, t_rel, set_idx):
        k = self.p['k']
        return set_idx == int(math.floor(math.floor(set_idx / k + 0.5) * k + 0.5))

    def state(self, msg, t_rel):
        k = self.p['k']
        for f in self.VEL:
            setattr(msg, f, getattr(msg, f) * k)
        for f in self.ACC:
            setattr(msg, f, getattr(msg, f) * k * k)


# ======================================================================================= lighting

class _Pixel(Transform):
    kind = 'image'
    params = {'cams': CAMS}

    def image(self, img, cam, t_rel, speed):
        if not self.applies(cam):
            return img
        return np.clip(self.apply(img.astype(np.float32), cam, t_rel, speed), 0, 255).astype(np.uint8)


@register
class Exposure(_Pixel):
    name = 'exposure'
    help = 'Multiply brightness by gain and clip (gain > 1: overexposed / daylight-like, < 1: darker).'
    params = {'gain': 1.8, 'cams': CAMS}

    def apply(self, f, cam, t, v):
        return f * self.p['gain']


@register
class Gamma(_Pixel):
    name = 'gamma'
    help = 'out = 255 * (in / 255) ^ g  (g < 1 lifts shadows, g > 1 darkens mid-tones).'
    params = {'g': 0.6, 'cams': CAMS}

    def apply(self, f, cam, t, v):
        return 255.0 * (f / 255.0) ** self.p['g']


@register
class Contrast(_Pixel):
    name = 'contrast'
    help = 'Scale contrast around the image mean (c < 1 flatter, c > 1 harsher).'
    params = {'c': 0.6, 'cams': CAMS}

    def apply(self, f, cam, t, v):
        m = f.mean()
        return m + self.p['c'] * (f - m)


@register
class Haze(_Pixel):
    name = 'haze'
    help = 'Blend toward a flat veil (dusk, haze, dirty lens): out = (1 - a) * in + a * level.'
    params = {'a': 0.3, 'level': 170, 'cams': CAMS}

    def apply(self, f, cam, t, v):
        return (1 - self.p['a']) * f + self.p['a'] * self.p['level']


@register
class Flicker(_Pixel):
    name = 'flicker'
    help = 'Auto-exposure fluctuation: per camera and frame, brightness gain ~ N(1, sigma).'
    params = {'sigma': 0.15, 'cams': CAMS}

    def apply(self, f, cam, t, v):
        return f * max(0.2, self.rng(cam, round(t, 3)).normal(1.0, self.p['sigma']))


@register
class Noise(_Pixel):
    name = 'noise'
    help = 'Sensor noise: Gaussian read noise (sigma, px) plus shot noise proportional to sqrt(intensity).'
    params = {'read': 4.0, 'shot': 0.6, 'cams': CAMS}

    def apply(self, f, cam, t, v):
        r = self.rng(cam, round(t, 3))
        z = r.standard_normal(f.shape, dtype=np.float32)
        return f + z * (self.p['read'] + self.p['shot'] * np.sqrt(np.maximum(f, 0)))


@register
class Jpeg(Transform):
    name, kind = 'jpeg', 'image'
    help = 'Stronger JPEG compression (artifacts on low-texture asphalt).'
    params = {'quality': 50, 'cams': CAMS}

    def image(self, img, cam, t_rel, speed):
        if not self.applies(cam):
            return img
        ok, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(self.p['quality'])])
        return cv2.imdecode(b, cv2.IMREAD_UNCHANGED)


@register
class LampsOff(Transform):
    name, kind = 'lamps_off', 'image'
    help = ('Remove the floodlights with their glow (and their reflections): saturated cores (>= thr) are grown '
            'into the connected halo (>= halo) and inpainted from the surroundings. Blobs larger than max_blob px '
            '(lit signs, buildings) are kept. Tests whether the localization depends on the lamps, which will '
            'not dominate in daylight.')
    params = {'thr': 230, 'halo': 90, 'grow': 7, 'max_blob': 6000, 'glow': 3.5, 'max_core': 4000, 'radius': 7,
              'cams': CAMS}

    def image(self, img, cam, t_rel, speed):
        if not self.applies(cam):
            return img
        v = img.max(axis=2) if img.ndim == 3 else img
        core = (v >= self.p['thr']).astype(np.uint8)
        if not core.any():
            return img
        # (1) small lamps: the whole connected halo around the core
        halo = (v >= self.p['halo']).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(halo, connectivity=8)
        hit = np.zeros(n, bool)
        hit[np.unique(lab[core > 0])] = True
        hit[0] = False
        hit &= st[:, cv2.CC_STAT_AREA] <= self.p['max_blob']
        m = hit[lab].astype(np.uint8)
        # (2) every lamp core: a disk sized to the core (removes the glow of big lamps whose halo merges with
        #     a bright hazy sky); very large cores are lit surfaces, not lamps, and are kept
        nc, lc, sc, cc = cv2.connectedComponentsWithStats(core, connectivity=8)
        for i in range(1, nc):
            area = sc[i, cv2.CC_STAT_AREA]
            if area > self.p['max_core']:
                continue
            r = int(np.clip(self.p['glow'] * math.sqrt(area / math.pi) + self.p['grow'], 6, 70))
            cv2.circle(m, (int(cc[i, 0]), int(cc[i, 1])), r, 1, -1)
        if not m.any():
            return img
        k = int(self.p['grow']) * 2 + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        s = 0.5                                              # inpaint at half resolution (fast)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        fill = cv2.inpaint(small, ms, self.p['radius'], cv2.INPAINT_TELEA)
        fill = cv2.resize(fill, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
        soft = soft[..., None] if img.ndim == 3 else soft
        return (img * (1 - soft) + fill * soft).astype(np.uint8)


@register
class Glare(_Pixel):
    name = 'glare'
    help = ('Low sun / strong light in view: additive radial glow, two flare streaks and a veil that lowers '
            'contrast. pos = (x, y) as fractions of the image, drifting slowly with time.')
    params = {'strength': 0.6, 'radius': 0.12, 'streak': 0.35, 'pos': [0.72, 0.18], 'drift': 0.01, 'veil': 0.12,
              'cams': ['fl', 'fr']}

    def apply(self, f, cam, t, v):
        h, w = f.shape[:2]
        r0 = self.rng(cam)
        cx = (self.p['pos'][0] + self.p['drift'] * math.sin(t / 7 + r0.uniform(0, 6))) * w
        cy = self.p['pos'][1] * h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        sig = self.p['radius'] * w
        glow = np.exp(-d2 / (2 * sig ** 2))
        ang = r0.uniform(0, np.pi)
        streak = np.zeros_like(glow)
        for a in (ang, ang + np.pi / 2.3):
            dist = np.abs((xx - cx) * math.sin(a) - (yy - cy) * math.cos(a))
            streak += np.exp(-dist ** 2 / (2 * (0.003 * w) ** 2)) * np.exp(-d2 / (2 * (2.0 * sig) ** 2))
        add = 255 * self.p['strength'] * np.clip(glow + self.p['streak'] * streak, 0, 1.5)
        f = (1 - self.p['veil']) * f + self.p['veil'] * 200
        return f + (add[..., None] if f.ndim == 3 else add)


@register
class Shadows(_Pixel):
    name = 'shadows'
    help = ('Daylight shadows on the track: soft dark slanted bands in the lower half that scroll toward the car '
            'like real shadows of gantries / grandstands.')
    params = {'n': 3, 'dark': 0.45, 'speed_px': 60, 'cams': CAMS}

    def apply(self, f, cam, t, v):
        h, w = f.shape[:2]
        r = self.rng(cam)
        mask = np.zeros((h, w), np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for i in range(int(self.p['n'])):
            slope = r.uniform(-0.4, 0.4)
            width = r.uniform(25, 90)
            period = h * 0.5 + width * 2
            y0 = h * 0.5 + ((r.uniform(0, period) + self.p['speed_px'] * t * (1 + 0.5 * i)) % period)
            band = np.abs(yy - (y0 + slope * (xx - w / 2))) < width / 2
            mask = np.maximum(mask, band.astype(np.float32))
        mask = cv2.GaussianBlur(mask, (0, 0), 6)
        mask[: h // 2] = 0
        g = 1 - self.p['dark'] * mask
        return f * (g[..., None] if f.ndim == 3 else g)


# ======================================================================================= blur / occlusion

@register
class MotionBlur(Transform):
    name, kind = 'motion_blur', 'image'
    help = ('Speed-dependent motion blur: side cameras get a horizontal line kernel of px_per_mps * speed px, '
            'front / rear cameras a zoom blur of zoom_per_mps * speed (fraction of the image). Set fixed_px to '
            'ignore speed.')
    params = {'px_per_mps': 0.25, 'zoom_per_mps': 0.0006, 'fixed_px': 0, 'cams': CAMS}

    def image(self, img, cam, t_rel, speed):
        if not self.applies(cam):
            return img
        v = abs(speed or 0.0)
        if cam in SIDE_CAMS:
            L = int(round(self.p['fixed_px'] or self.p['px_per_mps'] * v))
            if L < 2:
                return img
            k = np.zeros((1, L), np.float32); k[:] = 1.0 / L
            return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
        z = (self.p['fixed_px'] / img.shape[1]) if self.p['fixed_px'] else self.p['zoom_per_mps'] * v
        if z < 1e-3:
            return img
        h, w = img.shape[:2]
        acc = np.zeros(img.shape, np.float32)
        n = 6
        for i in range(n):
            s = 1 + z * i / (n - 1)
            M = np.array([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]], np.float32)
            acc += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return (acc / n).astype(np.uint8)


@register
class Occlusion(Transform):
    name, kind = 'occlusion', 'image'
    help = 'A car ahead / object in view: a dark ellipse covering frac of the image, drifting slowly.'
    params = {'frac': 0.12, 'cams': ['fl'], 'level': 25}

    def image(self, img, cam, t_rel, speed):
        if not self.applies(cam):
            return img
        h, w = img.shape[:2]
        r = self.rng(cam)
        area = self.p['frac'] * h * w
        ax = int(math.sqrt(area / math.pi * 2.2)); ay = int(ax / 2.2)
        cx = int(w * (0.5 + 0.2 * math.sin(t_rel / 9 + r.uniform(0, 6))))
        cy = int(h * 0.55)
        out = img.copy()
        cv2.ellipse(out, (cx, cy), (ax, ay), 0, 0, 360, (self.p['level'],) * 3, -1)
        return out


@register
class Dirt(Transform):
    name, kind = 'dirt', 'image'
    help = 'Dirt / droplets on the lens: n static blurred semi-transparent blobs per camera.'
    params = {'n': 6, 'alpha': 0.6, 'cams': ['fl', 'rl']}

    def image(self, img, cam, t_rel, speed):
        if not self.applies(cam):
            return img
        h, w = img.shape[:2]
        r = self.rng(cam)
        m = np.zeros((h, w), np.float32)
        for _ in range(int(self.p['n'])):
            cv2.circle(m, (int(r.uniform(0, w)), int(r.uniform(0, h))), int(r.uniform(15, 60)), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 12) * self.p['alpha']
        blurred = cv2.GaussianBlur(img, (0, 0), 9)
        m = m[..., None] if img.ndim == 3 else m
        return (img * (1 - m) + blurred * 0.6 * m).astype(np.uint8)


# ======================================================================================= sensor failures / timing

@register
class DropFrames(Transform):
    name, kind = 'drop_frames', 'stream'
    help = ('Dropped images: random fraction p (whole image sets, or per camera with per_cam=true) plus `bursts` '
            'outages of burst_s seconds at random times (after the first 5 s).')
    params = {'p': 0.1, 'per_cam': False, 'bursts': 0, 'burst_s': 1.5, 'dur_s': 120.0}

    def setup(self, ctx):
        if 'dur_s' not in self.given:
            self.p['dur_s'] = ctx['dur_s']

    def _bursts(self):
        r = self.rng('bursts')
        return [(s, s + self.p['burst_s']) for s in r.uniform(5, max(6, self.p['dur_s'] - self.p['burst_s']),
                                                                   int(self.p['bursts']))]

    def keep_image(self, cam, t_rel, set_idx):
        if any(a <= t_rel < b for a, b in self._bursts()):
            return False
        key = (cam, set_idx) if self.p['per_cam'] else ('set', set_idx)
        return self.rng(*key).random() >= self.p['p']


@register
class CameraOff(Transform):
    name, kind = 'camera_off', 'stream'
    help = 'Remove cameras completely (camera failure).'
    params = {'cams': ['fl']}

    def keep_image(self, cam, t_rel, set_idx):
        return cam not in self.p['cams']


@register
class Blackout(Transform):
    name, kind = 'blackout', 'stream'
    help = ('No images at all from at_s for dur_s seconds: the filter runs blind and must recover afterwards '
            '(a data-level "kidnapping" test).')
    params = {'at_s': 40.0, 'dur_s': 5.0}

    def keep_image(self, cam, t_rel, set_idx):
        return not (self.p['at_s'] <= t_rel < self.p['at_s'] + self.p['dur_s'])


@register
class CamTimeOffset(Transform):
    name, kind = 'cam_time_offset', 'stream'
    help = ('Camera clock error: header stamps of the chosen cameras shifted by ms (+ Gaussian jitter_ms per '
            'image). The images still show the scene at the true time, so the stamps are wrong by that amount.')
    params = {'ms': 10.0, 'jitter_ms': 0.0, 'cams': CAMS}

    def stamp_offset_ns(self, cam, t_rel):
        if not self.applies(cam):
            return 0
        j = self.rng(cam, round(t_rel, 3)).normal(0, self.p['jitter_ms']) if self.p['jitter_ms'] else 0.0
        return int(round((self.p['ms'] + j) * 1e6))


# ======================================================================================= calibration / state / start

@register
class Extrinsic(Transform):
    name, kind = 'extrinsic', 'calib'
    help = ('Calibration mismatch: the camera extrinsics in /tf_static that the pipeline reads are rotated by '
            'droll / dpitch / dyaw degrees (base_link axes) and shifted by dx / dy / dz m; the images are unchanged.')
    params = {'cam': 'fl', 'droll': 0.0, 'dpitch': 0.0, 'dyaw': 1.0, 'dx': 0.0, 'dy': 0.0, 'dz': 0.0}

    def tf_static(self, msg):
        from testgen.geometry import perturb_camera
        perturb_camera(msg, self.p)


@register
class StateNoise(Transform):
    name, kind = 'state_noise', 'state'
    help = ('Car-state errors: speed scale bias (fraction), speed noise (m/s), yaw-rate bias and noise (deg/s), '
            'white noise per message.')
    params = {'speed_bias': 0.01, 'speed_sigma': 0.05, 'wz_bias_dps': 0.1, 'wz_sigma_dps': 0.2}

    def state(self, msg, t_rel):
        r = self.rng(round(t_rel, 4))
        g = 1 + self.p['speed_bias']
        msg.vx_mps = msg.vx_mps * g + r.normal(0, self.p['speed_sigma'])
        msg.vy_mps = msg.vy_mps * g + r.normal(0, self.p['speed_sigma'] * 0.2)
        msg.v_mps = msg.v_mps * g
        msg.wz_radps = msg.wz_radps + math.radians(self.p['wz_bias_dps'] + r.normal(0, self.p['wz_sigma_dps']))


@register
class StateRaw(Transform):
    name, kind = 'state_raw', 'state'
    help = 'Only raw wheel speed is trustworthy: vx := v_raw_mps, vy := 0 (in case the test file lacks fused speed).'
    params = {}

    def state(self, msg, t_rel):
        msg.vx_mps = msg.v_raw_mps
        msg.vy_mps = 0.0


@register
class InitOffset(Transform):
    name, kind = 'init_offset', 'init'
    help = 'The given first pose is wrong by lon / lat metres (car frame) and yaw_deg degrees.'
    params = {'lon': 3.0, 'lat': 0.0, 'yaw_deg': 0.0}

    def init_pose(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * self.p['lon'] - s * self.p['lat'], y + s * self.p['lon'] + c * self.p['lat'],
                yaw + math.radians(self.p['yaw_deg']))


# ======================================================================================= spec parsing

def parse_spec(spec, seed=0, presets=None, _depth=0):
    """'a:k=v,k=v+b+@preset' -> list of Transform instances (in order)."""
    if presets is None:
        from testgen.presets import PRESETS as presets
    out = []
    for part in [s.strip() for s in (spec or '').split('+') if s.strip()]:
        if part.startswith('@'):
            name = part[1:]
            if name not in presets:
                raise ValueError(f'unknown preset @{name}; known: {sorted(presets)}')
            if _depth > 5:
                raise ValueError('preset nesting too deep')
            out += parse_spec(presets[name]['spec'], seed, presets, _depth + 1)
            continue
        name, _, args = part.partition(':')
        if name not in REGISTRY:
            raise ValueError(f'unknown transform "{name}"; known: {sorted(REGISTRY)}')
        kw = {}
        for a in filter(None, args.split(',')):
            k, _, v = a.partition('=')
            kw[k.strip()] = _parse_value(v.strip())
        out.append(REGISTRY[name](seed=seed, index=len(out), **kw))
    if _depth == 0:
        n_speed = sum(isinstance(t, SpeedUp) for t in out)
        if n_speed > 1:
            raise ValueError('only one speedup per spec')
    return out


def describe_all():
    lines = []
    for n, c in sorted(REGISTRY.items()):
        lines.append(f'{n:16s} [{c.kind}] {c.params}\n    {c.help}')
    return '\n'.join(lines)
