#!/bin/bash
# Same robustness suite as scripts/robustness_gpu.sh, with the RoMa fallback switched on (run.py --roma-fallback).
# Meant to run as a second GPU job next to the baseline job, on the same maps and the same test files:
#
#   BAG=/path/to/c0_2026-10-13_22-53-42_camloc_0.mcap ROOT=$WS/camloc_testsets bash scripts/robustness_roma_gpu.sh
#
#   1. waits for the held-out maps maps_no26/kf_{fl,rl}.npz (built by the baseline job)
#   2. adds the keyframe images to them: maps_no26/kf_{fl,rl}_img.npz (scripts/add_kf_images.py, one bag pass each)
#   3. runs the suite with --tag roma: waits for each test file the baseline job builds, then runs run.py on it
#      -> $ROOT/results_roma.csv / .md, and $ROOT/compare.md (baseline vs RoMa, when the baseline has finished)
set -e
cd "$(dirname "$0")/.."
BAG=${BAG:-c0_2026-10-13_22-53-42_camloc_0.mcap}
ROOT=${ROOT:-$HOME/camloc_testsets}
SUITE=${SUITE:-quick}
LAPS=${LAPS:-2,6}
PY=${PY:-python}
CAMS=${CAMS:-fl,rl}
N=${N:-3}                       # RoMa after N image sets without an accepted fix
ROMA_ARGS=${ROMA_ARGS:-}        # e.g. "--roma-model tiny" or "--roma-max-calls 2"
step() { echo "== $(date +%H:%M) $*"; }

[ -f "$BAG" ] || { echo "dataset not found: $BAG"; exit 1; }
step "1/3 waiting for the held-out maps"
for c in ${CAMS//,/ }; do
  until [ -f maps_no26/kf_$c.npz ] && [ -f gt.csv ] && [ -f cache/laps.csv ]; do sleep 60; done
done
sleep 30                        # let the last np.savez finish writing
step "2/3 keyframe images"
for c in ${CAMS//,/ }; do
  [ -f maps_no26/kf_${c}_img.npz ] || $PY scripts/add_kf_images.py "$BAG" cache gt.csv maps_no26/kf_$c.npz --cam $c
done
step "3/3 suite '$SUITE' on laps $LAPS with the RoMa fallback (N=$N) -> $ROOT"
$PY scripts/run_testsets.py "$BAG" gt.csv --laps $LAPS --suite $SUITE --root "$ROOT" --skip-run-existing --tag roma --wait-built 14400 \
  --cmd "$PY run.py {mcap} {csv} --cams $CAMS --maps maps_no26 --pf-config configs/fl_rl.json --roma-fallback $N $ROMA_ARGS"
[ -f "$ROOT/results.csv" ] && $PY scripts/compare_results.py "$ROOT" --tags ,roma --names baseline,roma || true
step "done: $ROOT/results_roma.md"
