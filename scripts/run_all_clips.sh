#!/usr/bin/env bash
set -euo pipefail

cd ~/multi-view-vehicle-data-engine
source /opt/pytorch/bin/activate

# Each entry: "<camera> <clip-name> <start-frame> <end-frame>"
RUNS=( "c0 c0_5 420 650" "c1 c1_1 7470 7700" "c2 c2_0 5330 5560" \
       "c3 c3_3 7310 7540" "c4 c4_2 8850 8980" "c4 c4_3 8720 8850" )

for r in "${RUNS[@]}"; do
  set -- $r
  python scripts/run_one_clip.py --clip "$2.mp4" --camera "$1" --clip-name "$2" \
    --start "$3" --end "$4" --sam-model large --out "${2}_tracks.json"
done
