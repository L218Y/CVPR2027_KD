#!/usr/bin/env bash
# Multi-seed repetition of the decisive comparisons (teacher, RGB, uniform, RAPD).
# A teacher is retrained per seed so that teacher variance is included.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p logs configs/seeds
while pgrep -f "tools/train_kd.py" >/dev/null 2>&1; do sleep 120; done

gen() {
python3 - "$@" <<'PY'
import sys, yaml, copy
base, out, seed, teacher = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = yaml.safe_load(open(base))
if "KD" in cfg and cfg["KD"].get("ENABLE"):
    cfg["KD"]["TEACHER_PATH"] = f"output/MFNet_CMNeXt-B2_it_teacher_s{seed}/best.pth"
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
}

for seed in 3408 3409; do
  gen configs/teacher_mfnet.yaml configs/seeds/teacher_s$seed.yaml $seed ""
  gen configs/rgb_mfnet.yaml     configs/seeds/rgb_s$seed.yaml     $seed ""
  gen configs/kd_mfnet_uniform.yaml configs/seeds/uniform_s$seed.yaml $seed t
  gen configs/kd_mfnet.yaml         configs/seeds/rapd_s$seed.yaml    $seed t

  echo "queue: seed $seed teacher"
  bash scripts/run_and_sync.sh configs/seeds/teacher_s$seed.yaml "seed$seed teacher" --seed $seed >> logs/seed${seed}_teacher.log 2>&1
  echo "queue: seed $seed rgb"
  bash scripts/run_and_sync.sh configs/seeds/rgb_s$seed.yaml "seed$seed rgb" --seed $seed >> logs/seed${seed}_rgb.log 2>&1
  echo "queue: seed $seed uniform"
  bash scripts/run_and_sync.sh configs/seeds/uniform_s$seed.yaml "seed$seed uniform" --seed $seed >> logs/seed${seed}_uniform.log 2>&1
  echo "queue: seed $seed rapd"
  bash scripts/run_and_sync.sh configs/seeds/rapd_s$seed.yaml "seed$seed rapd" --seed $seed >> logs/seed${seed}_rapd.log 2>&1
done
python3 scripts/summarize.py || true
bash scripts/sync_results.sh "seed sweep" || true
echo "queue: seeds finished ($(date -u +%H:%M:%SZ))"
