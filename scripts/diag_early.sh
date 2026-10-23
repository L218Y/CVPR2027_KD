#!/usr/bin/env bash
# Early P0: day/night stratification plus calibration/hallucination for the
# RGB baseline and the teacher, as soon as the RGB run releases the GPU.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tools:${PYTHONPATH:-}"
mkdir -p logs results/diag_rgb results/diag_teacher

while pgrep -f "cfg configs/rgb_mfnet.yaml" >/dev/null 2>&1; do sleep 30; done

RGB=output/MFNet_CMNeXt-B2_i_rgb_s3407/best.pth
TEA=output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth

CUDA_VISIBLE_DEVICES=1 python3 scripts/diag_split.py --cfg configs/rgb_mfnet.yaml \
  --ckpt "$RGB" --out results/diag_rgb >> logs/diag_early.log 2>&1
CUDA_VISIBLE_DEVICES=1 python3 scripts/diag_split.py --cfg configs/teacher_mfnet.yaml \
  --ckpt "$TEA" --out results/diag_teacher >> logs/diag_early.log 2>&1
CUDA_VISIBLE_DEVICES=1 python3 tools/analyze.py --cfg configs/rgb_mfnet.yaml \
  --ckpt "$RGB" --out results/diag_rgb >> logs/diag_early.log 2>&1
CUDA_VISIBLE_DEVICES=1 python3 tools/analyze.py --cfg configs/teacher_mfnet.yaml \
  --ckpt "$TEA" --out results/diag_teacher >> logs/diag_early.log 2>&1

bash scripts/sync_results.sh "early P0 day/night + calibration" || true
echo "diag_early: finished $(date -u +%H:%M:%SZ)"
