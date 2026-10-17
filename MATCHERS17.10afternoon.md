# Matcher comparison: ALIKED + LightGlue vs LoMa-B vs Tiny RoMa vs RoMa (afternoon of 2026-10-17)

## 1. Summary

To reduce the **lighting risk** of the hidden test run, we compared four image matchers for absolute
relocalization against a night map: the current pipeline's **ALIKED + LightGlue**, the sparse matcher **LoMa-B**,
and the dense matchers **Tiny RoMa** and **RoMa**. All four use the same 113 query images (held-out lap 6, camera
`fl`), the same reference frames and the same 3D construction. Each query was tested clean and under four
synthetic perturbations (dark, lamps removed, crude daylight stand-in, motion blur).

- **When a fix succeeds, all four are equally precise:** about 5 cm lateral and 16–18 cm longitudinal median error.
- **They differ in how often they succeed.** RoMa is clearly the most robust. In the dark it still solves
  **87 %** of the queries where ALIKED solves **74 %**, and under the daylight stand-in **93 %** vs 81 %.
- **RoMa costs about 470 ms per match** on an A100, about 10 times ALIKED (49 ms). It cannot be the main
  matcher at 10 Hz, but it can be a **fallback** for the frames where ALIKED fails.
- **LoMa-B** is only slightly more robust than ALIKED (+6 points in the dark) at about 5 times the cost.
  **Tiny RoMa** is the fastest (30 ms) but as fragile as ALIKED in the dark.
- **The map does not depend on the floodlights.** Removing the lamps from the query images changes almost
  nothing for any matcher.

**Proposal:** keep ALIKED + LightGlue as the main matcher and add RoMa as a fallback when the ALIKED fix fails.
Combined (section 7), this solves 86.7 % of the dark queries instead of 73.5 %, with RoMa needed on 15–27 % of the
frames.

---

## 2. Question and test setup

| Item | Choice | Reason |
|---|---|---|
| Question | Which matcher keeps relocalizing when the images look different from the night map? | Lighting is the biggest risk listed in RESULTS17.10morning.md |
| Query images | Camera `fl`, **lap 6**, one image per second: **113 queries** | Lap 6 is a held-out tuning lap (fast, 40 m/s) |
| Reference images | Nearest training frame by GT pose (median distance 0.2 m), from laps 0, 1, 4, 5, 8 | Laps 2, 3, 6, 7 excluded: no held-out or validation image is used as a reference |
| 3D points | Reference frame r1 and the frame 0.3 s later (r2) matched by the **same** matcher, triangulated with GT poses | Same construction as the keyframe maps (`build_kf_map.py`) |
| Pose | Query ↔ r1 matches → 2D–3D pairs → PnP + RANSAC (≥ 15 inliers) → `base_link` x, y, yaw | Same as `KeyframeMap.localize` |
| Perturbations | Applied **to the query only** (testgen transforms): `clean`, `@dark`, `lamps_off`, `@day_proxy`, `@blur_only` | A different-condition run matched against a night map |
| Hardware | 1 × NVIDIA A100-SXM4-40GB (ZIH Alpha Centauri) | |

Script: `scripts/exp_roma.py`. Output: `out/exp_roma/summary.md`, `per_query.csv`, match pictures `vis_*.jpg`.

**Matchers**

| Name | Type | Model | Settings |
|---|---|---|---|
| `aliked` | sparse | ALIKED (2048 kpts, resize 1024, CLAHE) + LightGlue, as in `camloc/features.py` | unchanged from the pipeline |
| `loma_b` | sparse | LoMa-B (`lomatch`): DaD keypoints + DeDoDe-B descriptors + LightGlue-style matcher | 2048 kpts |
| `tiny_roma` | dense | Tiny RoMa v1 outdoor (`romatch`, XFeat backbone) | 5000 sampled matches, certainty > 0.3 |
| `roma` | dense | RoMa outdoor (`romatch`, DINOv2 backbone) | coarse 560, upsample 864, 5000 sampled matches, certainty > 0.3 |

**Perturbations** (`testgen/presets.py`, see TESTING.md)

| Name | What it does |
|---|---|
| `clean` | Original image |
| `@dark` | Half the light (gain 0.5) plus sensor noise |
| `lamps_off` | Floodlights, their glow and reflections removed and inpainted |
| `@day_proxy` | Lamps removed, brighter (gain 1.7), flatter contrast, moving shadow bands, some glare. **Not real daylight** |
| `@blur_only` | Speed-dependent motion blur (zoom blur on `fl`, 0.4 px per m/s) |

---

## 3. Results (113 queries, lap 6, camera `fl`)

Errors are over the solved fixes. "Within 1 m" counts unsolved queries as failures.
Time is per query match on the A100 (query feature extraction + matching for sparse, one dense match for RoMa).

| Matcher | Perturbation | Solved | Inliers (median) | Lateral median | Longitudinal median | Position p90 | Within 1 m | Yaw median | ms / match |
|---|---|---|---|---|---|---|---|---|---|
| ALIKED + LG | clean | 85.0 % | 186 | 0.056 m | 0.168 m | 0.96 m | 77.0 % | 0.063° | 49 |
| ALIKED + LG | dark | 73.5 % | 60 | 0.066 m | 0.181 m | 0.89 m | 67.3 % | 0.061° | 43 |
| ALIKED + LG | lamps off | 82.3 % | 185 | 0.047 m | 0.148 m | 0.74 m | 77.0 % | 0.054° | 49 |
| ALIKED + LG | day proxy | 81.4 % | 143 | 0.053 m | 0.164 m | 1.08 m | 72.6 % | 0.057° | 47 |
| ALIKED + LG | blur | 83.2 % | 168 | 0.092 m | 0.596 m | 1.49 m | 61.1 % | 0.064° | 49 |
| LoMa-B | clean | 86.7 % | 182 | 0.051 m | 0.170 m | 0.74 m | 81.4 % | 0.066° | 230 |
| LoMa-B | dark | 79.6 % | 75 | 0.056 m | 0.169 m | 0.74 m | 73.5 % | 0.072° | 234 |
| LoMa-B | lamps off | 85.0 % | 147 | 0.049 m | 0.164 m | 0.79 m | 79.6 % | 0.059° | 225 |
| LoMa-B | day proxy | 83.2 % | 101 | 0.054 m | 0.160 m | 0.64 m | 78.8 % | 0.059° | 232 |
| LoMa-B | blur | 87.6 % | 124 | 0.074 m | 0.678 m | 1.51 m | 59.3 % | 0.058° | 226 |
| Tiny RoMa | clean | 90.3 % | 503 | 0.053 m | 0.159 m | 0.92 m | 82.3 % | 0.061° | 30 |
| Tiny RoMa | dark | 74.3 % | 68 | 0.069 m | 0.182 m | 1.17 m | 65.5 % | 0.066° | 29 |
| Tiny RoMa | lamps off | 89.4 % | 428 | 0.049 m | 0.157 m | 1.44 m | 77.9 % | 0.056° | 31 |
| Tiny RoMa | day proxy | 83.2 % | 176 | 0.050 m | 0.189 m | 1.25 m | 71.7 % | 0.063° | 30 |
| Tiny RoMa | blur | 85.8 % | 296 | 0.089 m | 0.782 m | 1.75 m | 57.5 % | 0.071° | 30 |
| **RoMa** | clean | **92.9 %** | 775 | 0.045 m | 0.178 m | 1.37 m | 79.6 % | 0.061° | 469 |
| **RoMa** | dark | **86.7 %** | 253 | 0.039 m | 0.180 m | 1.36 m | **77.0 %** | 0.064° | 471 |
| **RoMa** | lamps off | **92.9 %** | 724 | 0.044 m | 0.179 m | 1.35 m | 79.6 % | 0.063° | 471 |
| **RoMa** | day proxy | **92.9 %** | 633 | 0.043 m | 0.196 m | 1.22 m | 78.8 % | 0.062° | 474 |
| **RoMa** | blur | **92.9 %** | 562 | 0.075 m | 0.639 m | 1.73 m | 61.9 % | 0.060° | 463 |

**Compact view: solved % / within 1 m %**

| Matcher | clean | dark | lamps off | day proxy | blur | ms / match |
|---|---|---|---|---|---|---|
| ALIKED + LG (current) | 85 / 77 | 74 / 67 | 82 / 77 | 81 / 73 | 83 / 61 | 49 |
| LoMa-B | 87 / 81 | 80 / 74 | 85 / 80 | 83 / 79 | 88 / 59 | 230 |
| Tiny RoMa | 90 / 82 | 74 / 66 | 89 / 78 | 83 / 72 | 86 / 58 | **30** |
| RoMa | **93** / 80 | **87 / 77** | **93** / 80 | **93** / 79 | **93** / 62 | 470 |

**Drop in solved fixes from clean (percentage points)**

| Matcher | dark | lamps off | day proxy | blur |
|---|---|---|---|---|
| ALIKED + LG | −11.5 | −2.7 | −3.6 | −1.8 |
| LoMa-B | −7.1 | −1.7 | −3.5 | +0.9 |
| Tiny RoMa | −16.0 | −0.9 | −7.1 | −4.5 |
| RoMa | **−6.2** | **0.0** | **0.0** | **0.0** |

---

## 4. Findings

1. **Precision is the same; robustness is not.** For solved fixes the lateral median is 4–7 cm and the
   longitudinal median 16–20 cm for every matcher. The differences are in the solve rate under changed images.
2. **RoMa is the most robust.** It loses 6 points in the dark and nothing under the daylight stand-in, lamp
   removal or blur. ALIKED loses 11.5 points in the dark. RoMa also keeps by far the most inliers (253 in the
   dark vs 60 for ALIKED), so its fixes stay well determined.
3. **RoMa's p90 is higher on clean images** (1.37 m vs 0.96 m). It solves harder queries that the others give up
   on, and some of those fixes are less accurate. In the particle filter these are handled by the fix gate.
4. **LoMa-B is a small step up from ALIKED** (+6 points solved in the dark, better p90 under the daylight
   stand-in) but takes about 230 ms. Part of that is our wrapper, which writes each query to a PNG file and uses
   LoMa's own loader. Most of it is the DeDoDe-B descriptor, which runs a large DINOv2 backbone. As tested, it is not
   worth replacing ALIKED.
5. **Tiny RoMa is the fastest** (30 ms) and good on clean images, but loses 16 points in the dark: as fragile as
   ALIKED where it matters.
6. **The map does not depend on the floodlights.** `lamps_off` changes the solve rate by at most 3 points for any
   matcher. The features that localize the car are on the track, walls and structures, not the lamps. This is
   good news for daylight.
7. **Motion blur hurts every matcher in the same way**, mostly longitudinally (median 0.17 → 0.6–0.8 m). See the
   caveat in section 5: the synthetic zoom blur shifts features forward, which acts like a timing offset.

---

## 5. Limits of this test

1. **Single frames, no filter.** These are raw fixes. The pipeline's particle filter fuses fixes over time and
   from two cameras, so the full-pipeline effect of a matcher change has to be measured with `run.py` on laps 2 and 6.
2. **One reference keyframe** (the nearest one) per query. The pipeline uses the two nearest keyframes
   (`--n-kf 2`) of a denser map, which is why ALIKED solves 85 % here and 94 % in the pipeline's own statistics.
   This affects all matchers equally.
3. **One lap, one camera** (`fl`, lap 6). `rl` was not tested. With 113 queries, differences below about
   5 points are within noise; the RoMa vs ALIKED gap in the dark (13 points) and under the daylight stand-in
   (11.5 points) is not.
4. **Synthetic conditions are not real daylight.** `@day_proxy` cannot create daylight textures or real shadows.
   The test shows robustness to image degradation and to losing the night-specific lamps; it cannot prove the
   daylight case.
5. **The blur is one-sided** (the image is averaged over zoom factors 1 … 1 + z), which shifts features as if the
   car were slightly further ahead. On the real car this corresponds to the difference between the exposure
   stamp and the middle of the exposure: a constant timing offset that can be calibrated, not a matcher problem.
6. **Timings are per match on an A100**, not per frame of the full pipeline (which also decodes images, runs two
   cameras and the filter). The RESULTS17.10morning.md runtime (201 ms) was measured on an RTX 2070.

---

## 6. Problems met (reproducibility notes)

1. **`lomatch` declares a broken dependency** (`dataclasses>=0.8`, which does not exist for modern Python).
   Install with `pip install --no-deps lomatch` and then `pip install einops tyro`.
2. **RoMa by default needs an optional compiled CUDA kernel** (`local_corr`, from `romatch[fused-local-corr]`).
   Without it, RoMa fails at the first match. `exp_roma.py` now uses the pure-PyTorch path by default
   (`use_custom_corr=False`); `--roma-fused-corr` turns the kernel on if it is installed.
3. **ZIH cluster:** Barnard has no GPUs (use Alpha Centauri or Capella). `srun` on Alpha needs `-N 1` and
   at most 6 CPUs per GPU. A conda environment created with `-n` landed in `/tmp` on the login node, invisible to
   the GPU nodes. It now lives in the workspace (`conda create -p $WS/envs/camloc`).
4. **Weights** for ALIKED, LightGlue, RoMa, Tiny RoMa (XFeat), LoMa (DaD, DeDoDe) and DINOv2 are downloaded on first
   use. `TORCH_HOME` points into the workspace so the jobs reuse them.

---

## 7. Combined estimate: ALIKED, with RoMa only when ALIKED fails

From `out/exp_roma/per_query.csv`: use the ALIKED fix when it succeeds, otherwise the RoMa fix.

| Perturbation | ALIKED solved | ALIKED + RoMa fallback | RoMa needed on | Within 1 m, ALIKED → combined |
|---|---|---|---|---|
| clean | 85.0 % | **92.9 %** | 15.0 % of frames | 77.0 → 80.5 % |
| dark | 73.5 % | **86.7 %** | 26.5 % | 67.3 → **76.1 %** |
| lamps off | 82.3 % | **92.9 %** | 17.7 % | 77.0 → 81.4 % |
| day proxy | 81.4 % | **92.9 %** | 18.6 % | 72.6 → 77.9 % |
| blur | 83.2 % | **92.9 %** | 16.8 % | 61.1 → 64.6 % |

- RoMa solves every query that ALIKED solves, so the combination reaches RoMa's solve rate while ALIKED handles
  most frames.
- The largest gain is in the dark: **+13 points** solved, **+9 points** within 1 m.
- Runtime: calling RoMa on every failed frame would add on average about 70 ms per frame (clean) to 125 ms (dark)
  on an A100: too much for the 100 ms target. The filter already bridges short gaps, so RoMa should only be called
  after several frames without an accepted fix (or at a lower resolution).
- "Within 1 m" rises less than "solved": some rescued fixes are more than 1 m off. The filter's outlier gate is
  meant for these.

**Decision:** build the RoMa fallback, triggered only after a run of failed frames, and judge it end to end with
`run.py` on laps 2 and 6 (clean and perturbed test files) against the unchanged pipeline.

---

## 8. Next steps

| Priority | Work | Purpose | Evaluated on |
|---|---|---|---|
| 1 | **Baseline robustness of the unchanged pipeline:** `run.py` (fl + rl, `configs/fl_rl.json`, `maps_no26`) on laps 2 and 6 test files: clean, faster, dark, lamps off, dropped frames, fl dead (`scripts/robustness_gpu.sh`) | Reference for the fallback; failure-analysis slide | Laps 2 and 6 |
| 2 | **RoMa fallback in the keyframe map:** store the keyframe images (or their stamps) in `kf_*.npz`, and in `localize` run RoMa on the nearest keyframe when the ALIKED fix fails, looking up the stored 3D keypoint pixels in the dense warp (as in `exp_roma.py`) | More fixes in dark / changed light | Laps 2 and 6 with `run.py`, clean and `@dark` / `@day_proxy` test files |
| 3 | **Runtime budget:** run RoMa only after N failed frames, or at most every k-th frame; lower resolution (`--roma-coarse 420 --roma-upsample 672`) | Keep the 100 ms target on average | Laps 2 and 6 |
| 4 | Repeat the comparison for `rl` | The pipeline uses fl + rl | Lap 6 |
| 5 | Keep ALIKED + LightGlue as the main matcher; drop LoMa-B and Tiny RoMa for now | No gain worth the cost | - |

---

## 9. Reproduce

```bash
export WS=<workspace>                       # dataset, conda env and torch cache
source $WS/miniforge/bin/activate
conda create -p $WS/envs/camloc python=3.12 -y
conda activate $WS/envs/camloc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git
pip install romatch
pip install --no-deps lomatch
pip install einops tyro
export TORCH_HOME=$WS/torch_cache

BAG=$WS/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
python tools/export_cache.py $BAG cache --img-every 0
python tools/extract_gt.py $BAG gt.csv
python scripts/split_laps.py gt.csv cache
python scripts/exp_roma.py $BAG cache gt.csv --matchers aliked,loma_b,tiny_roma,roma --out out/exp_roma
```

On ZIH Alpha Centauri as a batch job: `#SBATCH -N 1 --ntasks=1 --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=04:00:00`.
Runtime of the full comparison on one A100: about 20 minutes.
