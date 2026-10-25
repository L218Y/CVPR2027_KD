#!/usr/bin/env bash
# Three-seed confirmation: ce / cbce / cbce_kd at 500 epochs, packed onto two GPUs.
#
# The negative claim rests on the teacher adding nothing beyond class re-weighting,
# so the two contrasts need seed-level spread, not just a single run each.
#
#   bash scripts/run_confirm.sh            # all nine runs
#   EPOCHS=500 bash scripts/run_confirm.sh
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-500}"
TAG="${TAG:-confirm}"
mkdir -p logs_screen

run() { # gpu arm seed
  local gpu="$1" arm="$2" seed="$3" name="${TAG}_${arm}_s${seed}" rc
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:tools \
    python3 tools/train_kd.py --cfg "configs/min/$arm.yaml" \
      --seed "$seed" --epochs "$EPOCHS" --tag "$name" \
      > "logs_screen/$name.log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] $name rc=$rc"
  return "$rc"
}

# Five waves; each wave keeps both GPUs busy and the pairing is fixed here rather
# than derived, so a rerun reproduces the same schedule.
run 0 ce 3407            & A=$!; run 1 cbce 3407        & B=$!; wait "$A" "$B"
run 0 cbce_kd 3407       & A=$!; run 1 ce 3408          & B=$!; wait "$A" "$B"
run 0 cbce 3408          & A=$!; run 1 cbce_kd 3408     & B=$!; wait "$A" "$B"
run 0 ce 3409            & A=$!; run 1 cbce 3409        & B=$!; wait "$A" "$B"
run 0 cbce_kd 3409

echo "[$(date -u +%H:%M:%S)] confirmation complete"
