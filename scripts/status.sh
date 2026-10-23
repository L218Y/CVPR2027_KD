#!/usr/bin/env bash
# One-screen progress report: queue state, current run, ETA, GPU and results.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== queue =="
tail -3 ../queue.log 2>/dev/null || echo "no queue log"
echo
echo "== current run =="
pid=$(pgrep -f "tools/train_kd.py" | head -1)
if [ -n "${pid}" ]; then
  log=$(ls -t ../*.log logs/*.log 2>/dev/null | head -1)
  echo "pid $pid  log $log"
  tr '\r' '\n' < "$log" | tail -2 | sed 's/^/  /'
else
  echo "idle"
fi
echo
echo "== gpu =="
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "no gpu"
echo
echo "== results =="
[ -f results/SUMMARY.md ] && cat results/SUMMARY.md || ls results 2>/dev/null | head
