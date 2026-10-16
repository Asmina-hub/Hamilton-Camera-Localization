# Learned odometry from cameras: plan

This is Plan B2 from [PLAN.md](PLAN.md). The hidden test set may not contain speeds, so we train a model
that infers the car's motion from images. Training uses the training run's speeds and poses as labels;
inference uses images only, so it stays within the rules.

## Design decisions

| decision | choice | why |
|---|---|---|
| input | **two consecutive frames** (t−1, t), not a single image | Speed cannot be seen in one image; it comes from how far the scene moves between frames. Using t−1 and t keeps the method causal. |
| label | **relative motion between frames**: Δx, Δy, Δyaw in the car frame at t−1, computed from GT poses | It maps directly to image motion, and at inference it is added straight onto the previous pose. Fallback: `vx`/`wz` from `state.csv` interpolated to the image stamps. |
| representation | **optical flow or bird's-eye view (BEV)** instead of raw images | Makes the model learn motion rather than memorise places. |
| cameras | start with the side cameras `rl`/`rr` (or `cl`/`cr`) | They see the most ground, which makes the BEV most useful. |
| combination | fuse with geometric BEV odometry and track-border matching in the filter | The network fills in where texture is low or images are blurred; the borders correct drift. |

Timing facts: the 7 cameras are **hardware-synced** (identical stamps), and the frame interval is
exactly 100 ms. Camera `r` drops 5 frames; the other cameras drop none.

## Main risk: the model memorises the place, not the motion

The car drives at about the same speed at the same spot every lap. A network can learn "this lamp post
→ 40 m/s" instead of measuring motion. That looks accurate on the training run but fails on the hidden
test run, which may be faster.

Mitigations:
- **Split by lap**, never randomly by frame, so neighbouring frames don't end up in both train and
  validation.
- Feed **optical flow or BEV** (Farneback, or pretrained RAFT), not raw images.
- **Frame-skip augmentation**: pairs of (t−2, t) act like double speed and cover faster driving. Also
  augment brightness, blur and crops.
- Check the error per speed bin, and especially at the highest speeds.

## Schedule

| # | step | output | done when | time |
|---|---|---|---|---|
| 1 | **Export dataset**: every `rl`/`rr` (or `cl`/`cr`) frame from the mcap, downscaled, with Δx, Δy, Δyaw labels from GT and a lap index | `cache/odom/` frames + `labels.csv` | ~12k pairs per camera, labels sanity-checked against `state.csv` speeds | 1–2 h |
| 2 | **Lap split**: segment the run into laps by GT progress along the track; pick 1–2 held-out laps | `splits.json` | train/val laps fixed; pit-lane parts labelled | 0.5 h |
| 3 | **Geometric baseline**: warp to BEV using the extrinsics and camera height, then phase correlation / flow → Δx, Δy, Δyaw | `vo_geom.py` + error report | per-frame speed error and yaw-rate error on the val laps | 3–4 h |
| 4 | **Learned model**: small CNN, input = flow or BEV pair, output = Δx, Δy, Δyaw; train on the train laps | `vo_net.py`, weights | val error lower than step 3, also in the top speed bin | 4–6 h |
| 5 | **Generalisation checks**: frame-skip test (t−2, t), brightness and blur perturbation | short report | error stays reasonable on simulated higher speed | 1–2 h |
| 6 | **Integration**: plug the odometry into the shared filter in place of `state.csv` speeds, together with border matching | submission CSV for the val laps | `tools/evaluate.py` score on the held-out laps | 2–3 h |
| 7 | **Runtime**: measure the average time per frame on the RTX 2070 | timing table | ≤ 100 ms per frame (real-time) | 1 h |

Steps 3 and 4 can run in parallel. Step 3 is also the fallback if the network does not generalise.

## Deliverables

- Per-frame odometry error (speed, yaw rate) on the held-out laps: geometric vs learned.
- Full-pipeline `evaluate.py` output on the held-out laps.
- Runtime per frame, and the hardware used.
- Failure cases: low texture, motion blur, the pit lane, the highest speeds.
- Declaration of any pretrained models used (for example RAFT).
