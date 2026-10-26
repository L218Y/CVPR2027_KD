#!/usr/bin/env bash
# Stage-2 screening: five arms x 120 epochs x one seed on two GPUs.
#
# Two arms run concurrently, one per GPU; the fifth starts when a GPU frees up.
# Every arm writes its own log, so a crash is attributable without re-running.
#
#   bash scripts/run_screen.sh                 # 120 epochs, seed 3407
#   EPOCHS=200 SEED=3408 bash scripts/run_screen.sh
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-120}"
SEED="${SEED:-3407}"
mkdir -p logs_screen

run() { # gpu cfg tag
  local gpu="$1" cfg="$2" tag="$3" rc
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:tools \
    python3 tools/train_kd.py --cfg "configs/min/$cfg.yaml" \
      --seed "$SEED" --epochs "$EPOCHS" --tag "$tag" \
      > "logs_screen/$tag.log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] $tag finished rc=$rc"
  return "$rc"
}

echo "[$(date -u +%H:%M:%S)] stage-2 screening: 5 arms, ${EPOCHS} epochs, seed ${SEED}"

run 0 ce screen_ce & A=$!
run 1 cbce screen_cbce & B=$!
wait "$A" "$B"

run 0 uniform_kd screen_uniform_kd & A=$!
run 1 cbce_kd screen_cbce_kd & B=$!
wait "$A" "$B"

run 0 correctness_kd screen_correctness_kd

echo "[$(date -u +%H:%M:%S)] screening complete"
python3 - <<'PY'
import glob, json
for path in sorted(glob.glob("output/min/*/*screen*/metrics.json")):
    d = json.load(open(path))
    print(f"{d['experiment']:52s} ep={d['epochs']:4d} last10={d['last10_miou']:.2f} best={d['best_miou']:.2f}")
PY
