#!/usr/bin/env bash
# Diagnostics that need the GPU: wait for the seed sweep to release it, then
# stratify the existing checkpoints by day/night and measure r per class.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tools:${PYTHONPATH:-}"
mkdir -p logs results/diag

while pgrep -f "tools/train_kd.py" >/dev/null 2>&1; do sleep 120; done

python3 scripts/diag_split.py --cfg configs/rgb_mfnet.yaml \
  --ckpt output/MFNet_CMNeXt-B2_i_rgb_s3407/best.pth --out results/diag_rgb >> logs/diag.log 2>&1
python3 scripts/diag_split.py --cfg configs/kd_mfnet_uniform.yaml \
  --ckpt output/MFNet_CMNeXt-B2_it_uniform_s3407/best.pth --out results/diag_uniform >> logs/diag.log 2>&1
python3 scripts/diag_split.py --cfg configs/kd_mfnet.yaml \
  --ckpt output/MFNet_CMNeXt-B2_it_recoverability_s3407/best.pth --out results/diag_rapd >> logs/diag.log 2>&1
python3 scripts/diag_split.py --cfg configs/teacher_mfnet.yaml \
  --ckpt output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth --out results/diag_teacher >> logs/diag.log 2>&1

bash scripts/sync_results.sh "day/night diagnostics" || true
echo "diag: finished ($(date -u +%H:%M:%SZ))"
