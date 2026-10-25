#!/usr/bin/env bash
# Run the named arms two at a time, one per GPU.
#
#   bash scripts/run_arms.sh ce cbce uniform_kd cbce_kd correctness_kd
#   EPOCHS=500 SEED=3408 TAG=confirm bash scripts/run_arms.sh ce cbce_kd
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-120}"
SEED="${SEED:-3407}"
TAG="${TAG:-screen}"
mkdir -p logs_screen

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <arm> [arm ...]" >&2
  exit 2
fi

run() { # gpu arm tag
  local gpu="$1" arm="$2" tag="$3" rc
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:tools \
    python3 tools/train_kd.py --cfg "configs/min/$arm.yaml" \
      --seed "$SEED" --epochs "$EPOCHS" --tag "$tag" \
      > "logs_screen/$tag.log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] $tag finished rc=$rc"
  return "$rc"
}

i=1
for arm in "$@"; do
  if [ $((i % 2)) -eq 1 ]; then
    run 0 "$arm" "${TAG}_$arm" & A=$!
  else
    run 1 "$arm" "${TAG}_$arm" & B=$!
    wait "$A" "$B"
  fi
  i=$((i + 1))
done
# odd number of arms: wait for the last single run
if [ $((i % 2)) -eq 0 ]; then
  wait "$A"
fi

echo "[$(date -u +%H:%M:%S)] all requested arms done"
