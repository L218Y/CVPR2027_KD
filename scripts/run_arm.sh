#!/usr/bin/env bash
# One comparison arm, launched in the background with its own log.
#   bash scripts/run_arm.sh <gpu> <config> <tag> [seed]
set -euo pipefail
cd /data/lab/CVPR2027_KD
GPU=$1; CFG=$2; TAG=$3; SEED=${4:-3407}
mkdir -p logs_clean
PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU nohup python3 -u tools/train_kd.py \
  --cfg "$CFG" --tag "$TAG" --seed "$SEED" --workers 6 \
  > "logs_clean/${TAG}_s${SEED}.log" 2>&1 &
echo "launched $TAG on gpu $GPU pid $!"
