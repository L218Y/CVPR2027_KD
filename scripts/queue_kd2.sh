#!/usr/bin/env bash
# Remaining runs: the uniform control must come before further ablations, since
# it decides whether the gain comes from distillation at all.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p logs
while pgrep -f "tools/train_kd.py" >/dev/null 2>&1; do sleep 120; done

run() {
  local cfg="$1" note="$2"; shift 2
  echo "queue: starting $note ($(date -u +%H:%M:%SZ))"
  bash scripts/run_and_sync.sh "$cfg" "$note" "$@" >> "logs/${note// /_}.log" 2>&1
  echo "queue: $note exited with $? ($(date -u +%H:%M:%SZ))"
}

run configs/kd_mfnet_uniform.yaml "S1 uniform kd"
run configs/kd_mfnet_eps02.yaml   "S4 rapd eps 0.2" --tag eps02
python3 scripts/summarize.py || true
bash scripts/sync_results.sh "final summary" || true
echo "queue2: finished ($(date -u +%H:%M:%SZ))"
