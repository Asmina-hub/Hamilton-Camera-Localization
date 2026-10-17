# Robustness testing: synthetic "hidden test runs"

The hidden test run "may include higher speeds and different lighting". We can't record that, so we
build test files that **look like the hidden run** from held-out laps of the training run, perturb them in
controlled ways, and run the unchanged pipeline on them.

Use **laps 2 and 6** (held out from map building and tuning) for decisions. Laps 3 / 7 only for the final report.

## What a test file is

`scripts/make_testset.py` writes a folder:

| file | content |
|---|---|
| `test.mcap` | one lap: all 7 cameras, car state, `camera_info`, `/tf_static`, track borders (latched topics repeated at the start), and **only the first pose** on `/tf` |
| `gt.csv` | the ground truth of that lap, on the same (possibly compressed) time axis, for `tools/evaluate.py` |
| `manifest.json` | exactly what was applied: window, spec, every transform with its parameters, seed, kept / dropped images |
| `preview.jpg` | original vs test image for all 7 cameras |

**Leak guard:** the pose and track-state fields inside `/constructor0/state_estimation` are set to NaN. A
pipeline that accidentally reads ground truth breaks visibly. (`--keep-state-pose` turns this off.)

Files are deterministic: the same spec and `--seed` rebuild the same file.

## Commands

```bash
B=path/to/c0_2026-10-13_22-53-42_camloc_0.mcap
python3 tools/extract_gt.py $B gt.csv

python3 scripts/make_testset.py --list                       # every transform and preset
python3 scripts/make_testset.py $B gt.csv --list-laps        # lap windows (same as split_laps.py)

# one test file
python3 scripts/make_testset.py $B gt.csv --lap 6 --preset fast_125 --out ~/Downloads/camloc_testsets/lap6_fast
python3 run.py ~/Downloads/camloc_testsets/lap6_fast/test.mcap out/lap6_fast.csv
python3 tools/evaluate.py ~/Downloads/camloc_testsets/lap6_fast/gt.csv out/lap6_fast.csv

# a whole suite through any pipeline, with a results table (change vs clean per lap)
python3 scripts/run_testsets.py $B gt.csv --laps 2,6 --suite quick --root ~/camloc_testsets \
    --cmd "python3 run.py {mcap} {csv} --cams fl,rl --maps maps_no26 --pf-config configs/fl_rl.json"
```

**Use the held-out maps (`maps_no26`, built with `--exclude-laps 2,6`)**: the final `maps/` contain laps 2 and 6,
so testing on those laps with them would score images the map has already seen.

On a GPU machine, `scripts/robustness_gpu.sh` does everything: cache, GT, lap split, held-out maps for fl and rl,
then the suite (`BAG=... SUITE=quick LAPS=2,6 bash scripts/robustness_gpu.sh`).

Known issue in the pipeline this will show: image sets are built around `fl` stamps, so `fl_off` gives no poses
(coverage 0 %, not ranked) and rl-only frames are skipped when fl drops a frame.

Keep the test folders **outside** the iCloud-synced Documents folder (each file is 0.5–0.7 GB).

## Composing scenarios

Every perturbation is a separate transform; combine them with `+`, set parameters with `:key=val,key=val`,
lists with `|`, presets with `@name`:

```
--spec "speedup:k=1.25"                               one factor
--spec "@fast_125+lamps_off"                          speed + lighting
--spec "@dark+camera_off:cams=fl+drop_frames:p=0.05"  three factors, each controllable
--spec "extrinsic:cam=cl,dyaw=2+extrinsic:cam=rl,dpitch=0.5"
```

Ad-hoc combinations in the suite runner: `--specs "fast_dark=@fast_125+@dark;no_lamps_fl_off=lamps_off+camera_off:cams=fl"`.

## Transforms

| group | transform | what it does | main parameters |
|---|---|---|---|
| speed | `speedup` | Physically consistent faster run: keeps image sets so the car moves k× farther between frames, compresses all stamps by k (camera rate stays ~10 Hz), scales speeds / yaw rates by k and accelerations by k². Verified: speed from GT positions matches the scaled car state (33.2 vs 33.1 m/s at k = 1.25). | `k` |
| speed | `motion_blur` | Speed-dependent blur: horizontal on side cameras, zoom blur on front / rear | `px_per_mps`, `zoom_per_mps`, `fixed_px` |
| lighting | `exposure` | Brighter / darker with clipping | `gain` |
| lighting | `gamma` | Tone curve | `g` |
| lighting | `contrast` | Flatter / harsher | `c` |
| lighting | `haze` | Veil toward grey (dusk, haze) | `a`, `level` |
| lighting | `flicker` | Auto-exposure jumps per frame | `sigma` |
| lighting | `lamps_off` | Floodlights + glow + reflections removed and inpainted: does the map depend on the lamps? | `thr`, `halo`, `glow`, `max_core` |
| lighting | `glare` | Low sun: glow, streaks, veil | `strength`, `radius`, `streak`, `pos`, `cams` |
| lighting | `shadows` | Moving shadow bands on the track | `n`, `dark`, `speed_px` |
| image | `noise` | Read + shot noise | `read`, `shot` |
| image | `jpeg` | Strong compression | `quality` |
| image | `occlusion` | Car ahead | `frac`, `cams` |
| image | `dirt` | Dirty lens | `n`, `alpha`, `cams` |
| timing | `cam_time_offset` | Camera stamps wrong by ms (+ jitter) | `ms`, `jitter_ms`, `cams` |
| calibration | `extrinsic` | The calibration the pipeline reads is off (images unchanged) | `cam`, `droll`, `dpitch`, `dyaw`, `dx`, `dy`, `dz` |
| failure | `drop_frames` | Random missing image sets (or per camera) + outage bursts | `p`, `per_cam`, `bursts`, `burst_s` |
| failure | `camera_off` | Cameras dead | `cams` |
| failure | `blackout` | All cameras dark for a while, then recover | `at_s`, `dur_s` |
| car state | `state_noise` | Speed bias / noise, yaw-rate bias / noise | `speed_bias`, `speed_sigma`, `wz_bias_dps`, `wz_sigma_dps` |
| car state | `state_raw` | Raw wheel speed instead of fused speed | |
| start | `init_offset` | Wrong first pose | `lon`, `lat`, `yaw_deg` |

Every test file also starts from a **different place** than the full run (start of the lap, not the pit lane).

## Presets and suites

`testgen/presets.py`: single-factor presets per group (`fast_125`, `fast_150`, `dark`, `bright`, `dusk`, `flicker`,
`lamps_off`, `glare`, `shadows`, `day_proxy`, `jpeg_low`, `occlusion_fl`, `dirt`, `cam_offset_10`, `cam_jitter`,
`calib_1deg`, `calib_pitch`, `drops_10`, `drop_bursts`, `fl_off`, `rl_off`, `blackout_5s`, `state_noise`, `state_raw`,
`init_off_3m`) and combined ones (`test_like_night`, `test_like_day`, `worst_case`).

Suites for `run_testsets.py`: `quick` (6), `single` (all single-factor), `combined`, `all`.

## Comparing pipeline variants on the same test files

`run_testsets.py --tag NAME` writes `submission_NAME.csv`, `run_NAME.log` and `results_NAME.csv/md`, so several
variants can be scored on the same files. `--wait-built SECONDS` waits for another job to build each test file
instead of building it, so a second variant can run as a parallel GPU job behind the first.

```bash
# job 1: baseline (builds maps_no26 and the test files)
BAG=... ROOT=$WS/camloc_testsets bash scripts/robustness_gpu.sh
# job 2, at the same time: RoMa fallback (waits for job 1's maps and test files)
BAG=... ROOT=$WS/camloc_testsets bash scripts/robustness_roma_gpu.sh
# side by side, with the differences
python scripts/compare_results.py $WS/camloc_testsets --tags ,roma --names baseline,roma
```

The RoMa fallback (`run.py --roma-fallback N`, `camloc/roma_fallback.py`) relocalizes against the same keyframe
map with RoMa after N image sets without an accepted ALIKED fix. It needs the keyframe images,
`maps_no26/kf_<cam>_img.npz`, from `scripts/add_kf_images.py` (one bag pass per camera, no map rebuild).

## Limits

- **Synthetic lighting is not real daylight.** Brightening a night image does not create daylight textures,
  lit grandstands or real shadows. `lamps_off` and `day_proxy` show how much the pipeline depends on the
  night-specific features; they cannot prove it works in daylight.
- `speedup` with non-integer k gives uneven frame gaps (k = 1.25: 80 / 80 / 80 / 160 ms). The real camera runs
  at a steady 10 Hz.
- `extrinsic` changes the calibration the pipeline reads, not the images: it simulates "the calibration is
  off", the same effect as a remounted camera.
