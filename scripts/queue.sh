#!/usr/bin/env bash
# Sequential experiment queue for the single-GPU machine.
#
#   nohup bash scripts/queue.sh > logs/queue.log 2>&1 &
#
# Waits until no trainer is running, then executes the plan below. Every job
# publishes its own results through run_and_sync.sh, so the repository is up to
# date after each stage even if a later stage fails.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p logs

wait_for_gpu() {
  while pgrep -f "tools/train_kd.py" >/dev/null 2>&1; do
    echo "queue: waiting for the running trainer to finish ($(date -u +%H:%M:%SZ))"
    sleep 120
  done
}

run() {
  local cfg="$1" note="$2"
  shift 2
  echo "queue: starting $note ($(date -u +%H:%M:%SZ))"
  bash scripts/run_and_sync.sh "$cfg" "$note" "$@" >> "logs/${note// /_}.log" 2>&1
  echo "queue: $note exited with $? ($(date -u +%H:%M:%SZ))"
}

wait_for_gpu

run configs/rgb_mfnet.yaml                "S0 rgb baseline"
run configs/kd_mfnet_uniform.yaml         "S1 uniform kd"
run configs/kd_mfnet.yaml                 "S2 rapd"
run configs/kd_mfnet_conservative.yaml    "S3 conservative only"
run configs/kd_mfnet_eps02.yaml           "S4 rapd eps 0.2" --tag eps02

echo "queue: all runs done, summarizing"
python3 scripts/summarize.py || true
bash scripts/sync_results.sh "final summary" || true
echo "queue: finished ($(date -u +%H:%M:%SZ))"
