#!/bin/bash
# Robustness suite on a GPU machine. Run from the repo root of WangFu3's pipeline with the harness files
# copied in (testgen/, scripts/make_testset.py, scripts/run_testsets.py). His code is not modified.
#
#   BAG=/path/to/c0_2026-10-13_22-53-42_camloc_0.mcap bash scripts/robustness_gpu.sh
#   SUITE=single LAPS=2,6 bash scripts/robustness_gpu.sh          # more scenarios
#
# Steps (each skipped if its output exists):
#   1. cache/, gt.csv, cache/laps.csv            (his scripts)
#   2. held-out maps maps_no26/kf_{fl,rl}.npz    (his build_kf_map.py --exclude-laps 2,6, ~11 min each)
#      -> laps 2 and 6 are unseen by the maps, like the hidden run
#   3. build the test files, run his run.py on each, score, write $ROOT/results.csv / results.md
set -e
cd "$(dirname "$0")/.."
BAG=${BAG:-c0_2026-10-13_22-53-42_camloc_0.mcap}
ROOT=${ROOT:-$HOME/camloc_testsets}
SUITE=${SUITE:-quick}
LAPS=${LAPS:-2,6}
PY=${PY:-python}
CAMS=${CAMS:-fl,rl}
step() { echo "== $(date +%H:%M) $*"; }

[ -f "$BAG" ] || { echo "dataset not found: $BAG"; exit 1; }
step "1/3 cache, ground truth, lap split"
[ -f cache/state.csv ]  || $PY tools/export_cache.py "$BAG" cache --img-every 10
[ -f gt.csv ]           || $PY tools/extract_gt.py "$BAG" gt.csv
[ -f cache/laps.csv ]   || $PY scripts/split_laps.py gt.csv cache
step "2/3 held-out maps (without laps 2,6)"
mkdir -p maps_no26
for c in ${CAMS//,/ }; do
  [ -f maps_no26/kf_$c.npz ] || $PY scripts/build_kf_map.py "$BAG" cache gt.csv maps_no26/kf_$c.npz --cam $c --exclude-laps 2,6
done
step "3/3 suite '$SUITE' on laps $LAPS -> $ROOT"
$PY scripts/run_testsets.py "$BAG" gt.csv --laps $LAPS --suite $SUITE --root "$ROOT" --skip-run-existing \
  --cmd "$PY run.py {mcap} {csv} --cams $CAMS --maps maps_no26 --pf-config configs/fl_rl.json"
step "done: $ROOT/results.md"
