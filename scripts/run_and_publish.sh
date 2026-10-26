#!/usr/bin/env bash
# Run a command and publish its results when it ends, even if it crashes.
#
#   bash scripts/run_and_publish.sh --note "md teacher" \
#        python3 tools/train_kd.py --cfg configs/min/md_teacher.yaml --seed 3407
#
# The publish step is the same scripts/autopublish.sh used everywhere else:
# collect -> git push of small files -> per-run GitHub release with checkpoints.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

NOTE=""
if [ "${1:-}" = "--note" ]; then
  NOTE="${2:-}"; shift 2
fi
if [ "$#" -eq 0 ]; then
  echo "usage: $0 [--note TEXT] <command> [args...]" >&2
  exit 2
fi

START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "run: $* started at $START"

PUBLISHED=0
publish() {
  local status="$1"
  [ "$PUBLISHED" = "1" ] && return 0
  PUBLISHED=1
  bash scripts/autopublish.sh --status "$status" --note "${NOTE:-$*}" || true
}
trap 'publish 130' INT TERM

"$@"
STATUS=$?
echo "run: finished with status $STATUS"
publish "$STATUS"
exit "$STATUS"
