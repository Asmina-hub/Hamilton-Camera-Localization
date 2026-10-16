# Camera-based localization on Yas Marina: two plans

Task: estimate x, y and yaw of `constructor0/base_link` in `map` at 10 Hz from the 7 cameras, causally,
starting from the first GT pose. The score is position RMSE, and lateral RMSE matters most.

Open question for the organizers: **does the hidden test mcap contain `/constructor0/state_estimation`,
and which fields may be used?** The task PDF says car state data (speed, accelerations) may be used, but
it also says the test set has "no IMU data", and the README treats the whole topic as ground truth.

## Shared architecture

Both plans use the same framework. The only difference is **where the motion comes from**.

```
              ┌──────────────────────────┐
 motion    ─▶ │  predict (106 / 10 Hz)   │
 (A or B)     │  EKF / particle filter   │──▶ x, y, yaw @ 10 Hz (each image stamp)
              │  state: s, n, yaw        │
 cameras   ─▶ │  update: border matching │◀── track_bounds
              └──────────────────────────┘
```

The state is kept in the **track (Frenet) frame**: s is the distance travelled along the centreline, and
n is the lateral offset. That way lateral and longitudinal errors can be handled separately.

### Dead-reckoning baseline (training run)

This integrates only `vx`, `vy` and `wz` from `state_estimation`, starting from the first GT pose,
with no cameras:

| elapsed time   | 10 s  | 30 s  | 60 s  | 120 s | 300 s | 1190 s (full run) |
|----------------|-------|-------|-------|-------|-------|-------------------|
| position error | 0.8 m | 1.5 m | 1.6 m | 2.9 m | 22 m  | 115 m             |

---

## Plan A: the hidden test set has state_estimation

**Motion source:** dead reckoning directly from `vx`, `vy`, `wz` at 106 Hz.

- The table above shows that the cameras only need to **correct drift slowly**. They don't have to give
  an absolute position every frame.
- **Lateral n and heading:** detect lane lines and kerbs with the side cameras (`cl/cr/rl/rr`),
  back-project them onto the ground and match them to `track_bounds`. This directly gives the distance
  to the left and right borders and the relative heading.
- **Longitudinal s:** nothing constrains it on the straights, so it is corrected in two ways:
  - Align the heading change integrated from `wz` with the map's corner curvature. Every corner gives one
    s correction.
  - Optional: use the GT to build a database from the training run that maps position to image features
    (lamp posts, kerb starts, grandstands), and use visual place recognition to fix s.
- **The start is in the pit lane:** keep hypotheses for both border sets, or decide from n whether the
  car is on the main track or in the pit lane.
- **Expected:** position RMSE 0.2–0.4 m, with lower lateral error. Simple to build, and easily real-time
  on an RTX 2070.
- **Risk:** the plan relies almost entirely on the speed data. `vx` and `wz` are fused estimates that
  already use GNSS and lidar. If the test set gives raw wheel speeds (`omega_w_*`, `v_raw_mps`) instead,
  wheel slip will cause errors, so validate with those fields on the training run as well.

## Plan B: the hidden test set has no state_estimation

**Motion source:** estimate speed and yaw rate from the cameras (visual odometry).

- **B1, geometric:** warp the images into a bird's-eye view (BEV) and run optical flow or phase
  correlation between consecutive frames to get translation and rotation directly. The camera height
  above the ground is known, so the scale is known too.
  - Hard parts: night-time asphalt has little texture, and at 40 m/s the car moves 4 m in 0.1 s, so
    motion blur is severe.
  - The side cameras `cl/cr/rl/rr` see the ground better than the front ones. Lane lines and kerbs are
    the main features.
- **B2, learned (recommended, in parallel with B1):** train a small network that takes two consecutive
  frames and outputs `vx` and `wz`, **using the training run's state as labels**. At inference it uses
  images only, so it stays within the rules. There are about 12k frame pairs × 7 cameras, enough for a
  lightweight CNN.
- Map matching is exactly the same as in Plan A, but it has to correct more often and more strongly,
  because visual odometry drifts faster.
- **Extra constraint:** the car must stay inside the track and roughly follow the track direction. With
  that as a prior, the main remaining error is longitudinal (s).
- **Expected:** position RMSE 0.5–1.5 m, mostly longitudinal error on the straights.
- **Risk:** the test set may have harsh daylight or higher speeds, so B2 may not generalise. Train with
  brightness and blur augmentation.

---

## Schedule (Oct 17–18)

1. **Build the shared parts first.** Both plans need them:
   - Frenet frame conversion
   - border detection, back-projection and matching
   - the filter
   - CSV output and a run of `tools/evaluate.py`

   During development, use the GT speed as the motion source so the map matching can be validated on
   its own.
2. **Plan A** is then mostly a matter of switching the motion source to `state.csv`. It can be done in
   half a day, which gives a fallback submission early.
3. Spend the remaining time on **Plan B**'s visual odometry. Even if state data turns out to be
   available, it can go into the filter as an independent measurement, and it is a technical highlight
   for the jury (30% of the score is qualitative).

Validation: hold out 1–2 laps of the training run and use them only for `tools/evaluate.py`. Don't
overfit to this night session, because the test run may be faster and have different lighting.
