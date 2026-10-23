#!/usr/bin/env bash
# P0/P1 tail controller.
#
# The first queue launcher hard-coded the v2 config at start time, so the seed
# 3408 run inherited the pre-diagnosis settings (disagreement gate) that the
# night-pedestrian analysis later argued against. This controller keeps that
# run as an explicit gate ablation, then re-runs 3408 with the final config and
# slots the recoverability diagnostic in front of the remaining seeds.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tools:${PYTHONPATH:-}"
mkdir -p logs results

LOG="logs/p1_finish.log"
say() { echo "p1finish: $* $(date -u +%H:%M:%SZ)" >> "$LOG"; }

TEACHER=output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth
UNIFORM=output/MFNet_CMNeXt-B2_it_uniform_s3407/best.pth
OLD_V2=output/MFNet_CMNeXt-B2_it_recoverability_s3408

say "start"

# 1. Park the stale gate run under its own name once it exits.
while pgrep -f "train_kd.py configs/kd_mfnet_v2.yaml --seed 3408" >/dev/null 2>&1; do sleep 60; done
if [ -d "$OLD_V2" ] && [ ! -d "${OLD_V2}_v2gate" ]; then
  mv "$OLD_V2" "${OLD_V2}_v2gate"
  mv "results/$(basename "$OLD_V2")" "results/$(basename "${OLD_V2}_v2gate")" 2>/dev/null
  python3 - <<'PY' >> logs/p1_finish.log 2>&1
import json
p = "results/MFNet_CMNeXt-B2_it_recoverability_s3408_v2gate/metrics.json"
d = json.load(open(p))
d["experiment"] = "MFNet_CMNeXt-B2_it_recoverability_s3408_v2gate"
d["run_variant"] = "v2-gate"
d["note"] = "GATE=disagreement, RARE_PROTECT=0.5 (pre-diagnosis config)"
json.dump(d, open(p, "w"), indent=2)
PY
  say "renamed stale gate run"
  bash scripts/sync_results.sh "v2 gate ablation (pre-diagnosis config)" >> logs/p1_finish.log 2>&1
fi

# 2. Seed 3408 with the final config.
bash scripts/run_and_sync.sh configs/kd_mfnet_v2.yaml "p1 v2 s3408 final" --seed 3408 --tag v2 >> "$LOG" 2>&1 &
S3408=$!

# 3. Recoverability diagnostic as soon as the uniform teacher exists.
while [ ! -f "$UNIFORM" ]; do sleep 60; done
while pgrep -f "train_kd.py configs/kd_mfnet_uniform.yaml" >/dev/null 2>&1; do sleep 60; done
say "uniform ready, running r diagnostic"
CUDA_VISIBLE_DEVICES=0 python3 scripts/diag_dn.py --cfg configs/rgb_mfnet.yaml \
  --ckpt "$TEACHER" --out results/diag_teacher >> logs/diag_r.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 scripts/diag_dn.py --cfg configs/rgb_mfnet.yaml \
  --ckpt "output/MFNet_CMNeXt-B2_it_uniform_s3407/last.pth" --out results/diag_uniform >> logs/diag_r.log 2>&1
python3 scripts/r_by_class.py --run output/MFNet_CMNeXt-B2_it_uniform_s3407 \
  --out results/diag_uniform >> logs/diag_r.log 2>&1
bash scripts/sync_results.sh "P0 recoverability diagnostic" >> "$LOG" 2>&1
say "r diagnostic done"

# 4. Remaining seed on the free GPU.
wait $S3408
bash scripts/run_and_sync.sh configs/kd_mfnet_v2.yaml "p1 v2 s3409 final" --seed 3409 --tag v2 >> "$LOG" 2>&1

python3 scripts/verify_stats.py >> "$LOG" 2>&1
bash scripts/sync_results.sh "P0+P1 complete" >> "$LOG" 2>&1
say "finished"
