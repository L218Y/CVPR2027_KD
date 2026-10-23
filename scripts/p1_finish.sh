#!/usr/bin/env bash
# P0/P1 tail controller.
#
# Seed 3408 is started by the caller on the GPU that is already free; this
# controller waits for the uniform teacher, measures recoverability per class
# and lighting condition, then releases the other GPU for seed 3407 and finally
# uses the first free GPU for seed 3409.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tools:${PYTHONPATH:-}"
mkdir -p logs

LOG="logs/p1_finish.log"
say() { echo "p1finish: $* $(date -u +%H:%M:%SZ)" >> "$LOG"; }

UNIFORM=output/MFNet_CMNeXt-B2_it_uniform_s3407
busy() { pgrep -f "train_kd.py --cfg $1" >/dev/null 2>&1; }

say "start"

# 1. Wait for the uniform teacher to finish, then measure r per class.
while busy configs/kd_mfnet_uniform.yaml || [ ! -f "$UNIFORM/best.pth" ]; do sleep 60; done
say "uniform ready, running diagnostics"
CUDA_VISIBLE_DEVICES=0 python3 scripts/diag_dn.py --cfg configs/rgb_mfnet.yaml \
  --ckpt "$UNIFORM/last.pth" --out results/diag_uniform >> logs/diag_r.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 scripts/r_by_class.py --run "$UNIFORM" \
  --out results/diag_uniform >> logs/diag_r.log 2>&1
bash scripts/sync_results.sh "P0 recoverability diagnostic" >> "$LOG" 2>&1
say "r diagnostic done"

# 2. Seed 3407 on the GPU the uniform run releases.
bash scripts/run_and_sync.sh configs/kd_mfnet_v2.yaml "p1 v2 s3407 final" --seed 3407 --tag v2 >> "$LOG" 2>&1 &
S3407=$!

# 3. Seed 3409 once the GPU running 3408 is free.
while busy configs/kd_mfnet_v2.yaml; do sleep 60; done
bash scripts/run_and_sync.sh configs/kd_mfnet_v2.yaml "p1 v2 s3409 final" --seed 3409 --tag v2 >> "$LOG" 2>&1 &
S3409=$!

wait $S3407 $S3409
python3 scripts/verify_stats.py >> "$LOG" 2>&1
bash scripts/sync_results.sh "P0+P1 complete" >> "$LOG" 2>&1
say "finished"
