#!/usr/bin/env bash
# Run one training job and always publish its results when it ends.
#
#   bash scripts/run_and_sync.sh configs/teacher_mfnet.yaml "mfnet teacher"
#
# The sync step runs even if training crashes, so the failure and its log are
# visible on GitHub instead of only on the machine.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CFG="${1:?usage: run_and_sync.sh <config.yaml> [note]}"
NOTE="${2:-$(basename "$CFG" .yaml)}"

START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "run: $CFG started at $START"

python3 tools/train_kd.py --cfg "$CFG"
STATUS=$?

python3 scripts/collect_results.py --status "$STATUS" --note "$NOTE" || true
bash scripts/sync_results.sh "$NOTE" || echo "sync failed (exit $?)" >&2

echo "run: finished with status $STATUS"
exit "$STATUS"
