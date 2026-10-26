#!/usr/bin/env bash
# Fallback publisher: poll for finished runs and publish them even if the
# training wrapper was killed or the job was launched by hand.
#
#   nohup bash scripts/watch_publish.sh 180 > logs/watch_publish.log 2>&1 &
#
# Every INTERVAL seconds it calls scripts/autopublish.sh. That script is
# idempotent: it only picks up metrics.json files newer than its state file,
# skips release assets that already exist, and serialises concurrent publishers
# with flock. So running this alongside run_and_publish.sh is safe.
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-180}"
mkdir -p logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watch_publish started (every ${INTERVAL}s)"
while true; do
  bash scripts/autopublish.sh --status 0 --note "watchdog" >>logs/watch_publish.log 2>&1
  sleep "$INTERVAL"
done
