#!/bin/bash
# Held-out tuning + final run, unattended. The selection rule is fixed in advance:
# lowest position RMSE on training laps 2 and 6, using maps built WITHOUT those laps.
# Validation laps 3 and 7 are evaluated only in the last step, after the choice is frozen.
set -e
cd "$(dirname "$0")/.."
P=/home/kunting/test/Download/ENTER/envs/camloc/bin/python
BAG=c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p maps_no26 out/lolo out/final
step() { echo "STEP $(date +%H:%M) $*"; }

for c in fl cl rl; do
  step "1/5 held-out map $c (without laps 2,6)"
  [ -f maps_no26/kf_$c.npz ] || $P scripts/build_kf_map.py $BAG cache gt.csv maps_no26/kf_$c.npz --cam $c --exclude-laps 2,6 2>&1 | grep -v -i warn
done

step "2/5 online run fl,cl,rl with held-out maps"
[ -f out/lolo/run3_log.csv ] || $P run.py $BAG out/lolo/run3.csv --work cache --maps maps_no26 --cams fl,cl,rl 2>&1 | grep -v -i warn

step "3/5 select cameras + filter parameters on laps 2,6 only"
python3 scripts/sweep_pf.py cache gt.csv out/lolo/run3_log.csv --tune-laps 2,6 --cams fl fl,cl fl,rl fl,cl,rl

for c in cl rl; do
  step "4/5 final map $c (all training laps)"
  [ -f maps/kf_$c.npz ] || $P scripts/build_kf_map.py $BAG cache gt.csv maps/kf_$c.npz --cam $c 2>&1 | grep -v -i warn
done

CAMS=$(python3 -c "import json; print(json.load(open('out/lolo/run3_log_best.json'))['cams'])")
step "5/5 final online run, cams=$CAMS, frozen params -> validation"
cp out/lolo/run3_log_best.json out/final/selected_config.json
$P run.py $BAG out/final/final.csv --work cache --maps maps --cams $CAMS --pf-config out/final/selected_config.json 2>&1 | grep -v -i warn
python3 scripts/eval_laps.py gt.csv out/final/final.csv cache > out/final/final_eval.txt
python3 tools/plot_result.py cache gt.csv out/final/final.csv out/final/final_overview.png \
  --title "Final: cams $CAMS, params chosen on held-out training laps 2,6"
step "DONE"
cat out/final/final_eval.txt
