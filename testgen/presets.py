"""Named scenarios. Use them alone (--spec @dark) or combined (--spec "@fast_125+@dark+drop_frames:p=0.05").

Grouped by the hidden-run risks the organizers named: higher speed, different lighting, plus the 5 known
problems (night light, motion blur, long straights, calibration, camera timing) and sensor failures.
Severity is mild / medium / severe where it makes sense.
"""

PRESETS = {
    # ---- reference
    'clean':         dict(spec='', group='reference', note='Cropped lap, ground truth stripped, no perturbation.'),

    # ---- higher speed
    'fast_125':      dict(spec='speedup:k=1.25+motion_blur:px_per_mps=0.25', group='speed',
                          note='25 % faster (40 -> 50 m/s) with matching motion blur.'),
    'fast_150':      dict(spec='speedup:k=1.5+motion_blur:px_per_mps=0.25', group='speed',
                          note='50 % faster (40 -> 60 m/s) with matching motion blur.'),
    'blur_only':     dict(spec='motion_blur:px_per_mps=0.4', group='speed',
                          note='Stronger motion blur at the original speed (longer exposure).'),

    # ---- lighting
    'dark':          dict(spec='exposure:gain=0.5+noise:read=4,shot=0.8', group='lighting',
                          note='Half the light, more sensor noise.'),
    'bright':        dict(spec='exposure:gain=1.8', group='lighting', note='Overexposed, highlights clipped.'),
    'dusk':          dict(spec='haze:a=0.3+contrast:c=0.7+gamma:g=0.85', group='lighting',
                          note='Flat, hazy, low-contrast light.'),
    'flicker':       dict(spec='flicker:sigma=0.15', group='lighting', note='Auto-exposure jumps between frames.'),
    'lamps_off':     dict(spec='lamps_off', group='lighting',
                          note='Floodlights and reflections removed: does the map depend on the lamps?'),
    'glare':         dict(spec='glare:cams=fl|fr|cl|cr', group='lighting', note='Low sun in the front cameras.'),
    'shadows':       dict(spec='shadows', group='lighting', note='Moving shadow bands on the track.'),
    'day_proxy':     dict(spec='lamps_off+exposure:gain=1.7+contrast:c=0.75+shadows:dark=0.35+glare:strength=0.35',
                          group='lighting',
                          note='Crude daylight stand-in: no lamps, bright, flatter, shadows, some glare. '
                               'NOT real daylight (no daylight textures).'),

    # ---- image quality / occlusion
    'jpeg_low':      dict(spec='jpeg:quality=40', group='image', note='Strong compression artifacts.'),
    'occlusion_fl':  dict(spec='occlusion:cams=fl,frac=0.12', group='image', note='Car ahead in the front camera.'),
    'dirt':          dict(spec='dirt:cams=fl|rl', group='image', note='Dirty lens on fl and rl.'),

    # ---- timing / calibration
    'cam_offset_10': dict(spec='cam_time_offset:ms=10', group='timing', note='Camera stamps 10 ms late.'),
    'cam_jitter':    dict(spec='cam_time_offset:ms=0,jitter_ms=5', group='timing', note='5 ms stamp jitter.'),
    'calib_1deg':    dict(spec='extrinsic:cam=fl,dyaw=1+extrinsic:cam=rl,dyaw=-1', group='calibration',
                          note='fl / rl extrinsics wrong by 1 deg yaw (remount between sessions).'),
    'calib_pitch':   dict(spec='extrinsic:cam=fl,dpitch=0.5+extrinsic:cam=rl,dpitch=0.5', group='calibration',
                          note='0.5 deg pitch error.'),

    # ---- failures
    'drops_10':      dict(spec='drop_frames:p=0.1', group='failure', note='10 % of image sets missing.'),
    'drop_bursts':   dict(spec='drop_frames:p=0,bursts=3,burst_s=1.5', group='failure', note='3 x 1.5 s outages.'),
    'fl_off':        dict(spec='camera_off:cams=fl', group='failure', note='Front camera dead.'),
    'rl_off':        dict(spec='camera_off:cams=rl', group='failure', note='Rear-left camera dead.'),
    'blackout_5s':   dict(spec='blackout:at_s=40,dur_s=5', group='failure', note='All cameras dark for 5 s, then recover.'),

    # ---- car state / start
    'state_noise':   dict(spec='state_noise', group='state', note='1 % speed bias, speed and yaw-rate noise.'),
    'state_raw':     dict(spec='state_raw', group='state', note='Raw wheel speed instead of fused speed.'),
    'init_off_3m':   dict(spec='init_offset:lon=3', group='start', note='First pose 3 m off along the track.'),
}

# combined scenarios (kept separate from the single-factor ones so every factor stays controllable)
PRESETS['test_like_night'] = dict(
    spec='@fast_125+flicker:sigma=0.08+noise:read=3,shot=0.5+drop_frames:p=0.02',
    group='combined', note='Faster night run with mild exposure flicker, noise and 2 % drops.')
PRESETS['test_like_day'] = dict(
    spec='@fast_125+@day_proxy+drop_frames:p=0.02',
    group='combined', note='Faster run in the crude daylight stand-in.')
PRESETS['worst_case'] = dict(
    spec='@fast_150+@day_proxy+@calib_1deg+drop_frames:p=0.05,bursts=1+state_noise',
    group='combined', note='Everything at once: tells you where it breaks, not what to expect.')

# default suites for scripts/run_testsets.py
SUITES = {
    'quick': ['clean', 'fast_125', 'dark', 'lamps_off', 'drops_10', 'fl_off'],
    'single': [k for k, v in PRESETS.items() if v['group'] not in ('combined',)],
    'combined': ['clean', 'test_like_night', 'test_like_day', 'worst_case'],
}
SUITES['all'] = SUITES['single'] + [k for k in SUITES['combined'] if k != 'clean']
