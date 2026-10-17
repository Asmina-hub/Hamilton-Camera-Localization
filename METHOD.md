# Method and testing

## The method: slide methods 3a + 4

### Part 1: build the landmark map (offline, once, training laps only)

1. Read every front-camera (`fl`) image of the training laps (laps 1, 2, 4, 5, 6 plus the out-lap and
   in-lap) at 10 Hz, about 9,700 images.
2. Run **SIFT** on each image to find about 2,000 feature points: distinctive patches such as light
   spots, corners of grandstands and edges of billboards. Each point has a 128-dimensional
   "fingerprint" (descriptor).
3. Pair images 3 frames apart (0.3 s) and find the same points in both images.
4. **Triangulate.** The car's true pose (GT) is known on the training laps, so we know where each camera
   was and which way it pointed. Seen from two positions, a point lies where the two lines of sight
   meet; that gives its 3D position.
5. Drop unreliable points: reprojection error above 2 px, too near or too far, or more than 30 m above
   the track (e.g. floodlights).
6. Deduplicate with 0.5 m voxels. The result is **about 700k landmarks**, each with a 3D position, a
   fingerprint and the camera position it was seen from.

Using GT to build the map is allowed. Slide 16 says the training images all have ground-truth poses
and can be used to build a map.

### Part 2: localize each new image

1. Extract SIFT points from the new image.
2. Using the filter's **predicted pose**, select only the landmarks within 30 m of it with a similar
   heading (about 10k–30k).
3. On the GPU, find the landmark whose fingerprint is closest to each image point. If the best and
   second-best candidates are about equally close (ratio test), the match is ambiguous and dropped.
4. **PnP:** from a set of "pixel in the image ↔ point in 3D" correspondences, solve for the camera's
   position and orientation. RANSAC removes wrong matches; at least 15 inliers are required.
5. Convert the camera pose to the car's pose (`base_link`: x, y, yaw) with the camera extrinsics.

### Part 3: particle-filter fusion (method 4)

- **Predict:** car motion from the car-state speeds `vx`, `vy` and yaw rate `wz` (allowed by slide 9).
- **Map constraint:** the car must be on the track or in the pit lane, heading along the track.
- **Camera fixes:** the pose from part 2 is a measurement. Lateral is trusted more (σ 0.4 m) than
  longitudinal (σ 1.5 m). Fixes more than 10 m or 5° away from the current estimate are dropped. In
  this run 10,783 fixes were used and 48 were dropped.
- The run starts from the given first pose; after that, GT is never used.

---

## How it was tested

### Data split

- Training laps (laps 1, 2, 4, 5, 6 and the out-/in-laps): used to build the map and tune parameters.
- **Validation laps (laps 3 and 7): never used to build the map.** They are unseen test questions.
- Lap 7 is the fastest lap (40 m/s), chosen to test high-speed behaviour.

### Scoring

- The organizers' `tools/evaluate.py`, the same script used for the final ranking.
- GT is cropped to each validation lap's time window and each lap is scored separately
  (`scripts/eval_laps.py`).

Results:

| | position RMSE | lateral | longitudinal | yaw | within 1 m |
|---|---|---|---|---|---|
| Baseline without cameras, val lap 3 | 4.02 m | 3.09 m | 2.57 m | 0.85° | 5.7 % |
| **Landmarks + filter, val lap 3** | **0.41 m** | **0.18 m** | 0.37 m | **0.15°** | **98.6 %** |
| Baseline without cameras, val lap 7 (40 m/s) | 4.24 m | 2.61 m | 3.34 m | 0.88° | 4.0 % |
| **Landmarks + filter, val lap 7** | **0.43 m** | **0.21 m** | 0.38 m | **0.17°** | **96.4 %** |

### Limits of this test

1. **It is not yet a true online run.** In part 2, the "predicted pose" that selects landmarks currently
   comes from **the no-camera baseline's output** (error about 4 m), not from the camera-based filter's
   own prediction. This was done only so the localization results could be precomputed to save
   development time. With a 30 m selection radius, the online prediction should give similar or better
   results, since its error is only tenths of a metre. But **this still has to be confirmed by running
   the full online pipeline**, which the competition requires anyway: one command from raw data to
   result.
2. **The full-run score (0.31 m) is optimistic.** The training-lap images were used to build the map
   (questions already seen), so the validation laps are the numbers to trust.
3. **Validation and training laps are from the same night.** The hidden test run "may include higher
   speeds and different lighting". A night-time map may fail to match in daylight; this is the biggest
   unknown risk right now.
4. **Not fast enough yet:** about 150 ms per frame now; the requirement is ≤ 100 ms.
