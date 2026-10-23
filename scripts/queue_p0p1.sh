#!/usr/bin/env bash
# P0 infrastructure + P1: two GPUs, dependency-aware.
#
#   GPU0: teacher -> uniform (needed to measure r per class) -> v2 seed A
#   GPU1: rgb -> (wait teacher) -> diagnostics -> v2 seed B
# A third v2 seed runs on the first free GPU afterwards.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tools:${PYTHONPATH:-}"
mkdir -p logs results/diag

TEACHER=output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth

wait_for() {  # wait_for <file>
  while [ ! -f "$1" ]; do sleep 120; done
}

run() {  # run <gpu> <cfg> <note> [extra...]
  local gpu="$1" cfg="$2" note="$3"; shift 3
  echo "queue: GPU$gpu $note start $(date -u +%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_and_sync.sh "$cfg" "$note" "$@" >> "logs/${note// /_}.log" 2>&1
  echo "queue: GPU$gpu $note done rc=$? $(date -u +%H:%M:%SZ)"
}

( run 0 configs/teacher_mfnet.yaml "p0 teacher" --seed 3407
  run 0 configs/kd_mfnet_uniform.yaml "p0 uniform" --seed 3407
  run 0 configs/kd_mfnet_v2.yaml "p1 v2 s3407" --seed 3407 --tag v2 ) &

( run 1 configs/rgb_mfnet.yaml "p0 rgb" --seed 3407
  wait_for "$TEACHER"
  PYTHONPATH=. python3 scripts/diag_split.py --cfg configs/kd_mfnet_uniform.yaml \
      --ckpt output/MFNet_CMNeXt-B2_it_uniform_s3407/best.pth --out results/diag_uniform >> logs/diag.log 2>&1
  PYTHONPATH=. python3 scripts/diag_split.py --cfg configs/rgb_mfnet.yaml \
      --ckpt output/MFNet_CMNeXt-B2_i_rgb_s3407/best.pth --out results/diag_rgb >> logs/diag.log 2>&1
  bash scripts/sync_results.sh "P0 diagnostics" || true
  run 1 configs/kd_mfnet_v2.yaml "p1 v2 s3408" --seed 3408 --tag v2 ) &

wait
run 0 configs/kd_mfnet_v2.yaml "p1 v2 s3409" --seed 3409 --tag v2
python3 scripts/verify_stats.py || true
bash scripts/sync_results.sh "P0+P1 complete" || true
echo "queue: finished $(date -u +%H:%M:%SZ)"
