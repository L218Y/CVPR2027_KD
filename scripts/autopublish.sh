#!/usr/bin/env bash
# Publish an experiment when it ends.
#
#   bash scripts/autopublish.sh --status 0 --note "md teacher 120ep"
#   bash scripts/autopublish.sh --all --dry-run
#
# What it does, in order:
#   1. collect metrics/logs into results/ (scripts/collect_results.py)
#   2. commit the small, reviewable files and push them to GitHub
#   3. create one GitHub release per finished run and upload its checkpoints
#      (best/last/probe), metrics.json and train.log as release assets
#   4. write results/releases/<tag>.json + results/RELEASES.md and verify every
#      uploaded asset's size against the local file
#
# It is safe to call after a crash: the exit status is recorded, the log is
# still published, and a failed upload does not fail the training job.
#
# Token (never written to the repo, never echoed):
#   $GITHUB_TOKEN, else $PAT_FILE (default ~/.config/cvpr_kd/pat), else gh auth.
# If no token is available it still collects and commits locally, then writes
# results/PUBLISH_PENDING.md with the exact commands needed to finish.
#
# Environment overrides:
#   REPO=L218Y/CVPR2027_KD  REMOTE=origin  BRANCH=main
#   PAT_FILE=~/.config/cvpr_kd/pat  AUTOPUBLISH_ALL=0
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

REPO="${REPO:-L218Y/CVPR2027_KD}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
PAT_FILE="${PAT_FILE:-$HOME/.config/cvpr_kd/pat}"
STATE="$REPO_DIR/logs/.autopublish_state.json"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR" "$REPO_DIR/results/releases"
LOG="$LOG_DIR/autopublish.log"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

STATUS=0
NOTE=""
TAG_FILTER=""
DRY=0
NO_RELEASE=0
ALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --status) STATUS="${2:-0}"; shift 2 ;;
    --note) NOTE="${2:-}"; shift 2 ;;
    --tag) TAG_FILTER="${2:-}"; shift 2 ;;
    --all) ALL=1; shift ;;
    --no-release) NO_RELEASE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) say "unknown argument: $1"; exit 2 ;;
  esac
done
[ "${AUTOPUBLISH_ALL:-0}" = "1" ] && ALL=1

# Serialise concurrent publishers (two GPUs finishing together).
exec 9>"${TMPDIR:-/tmp}/cvpr2027_autopublish.lock"
if ! flock -w 600 9; then
  say "another autopublish is still running; giving up"
  exit 0
fi

TOKEN="${GITHUB_TOKEN:-}"
TOKEN_SOURCE="GITHUB_TOKEN"
if [ -z "$TOKEN" ] && [ -s "$PAT_FILE" ]; then
  TOKEN="$(tr -d '\r\n' < "$PAT_FILE")"
  TOKEN_SOURCE="$PAT_FILE"
fi
if [ -z "$TOKEN" ] && command -v gh >/dev/null 2>&1; then
  TOKEN="$(gh auth token 2>/dev/null || true)"
  [ -n "$TOKEN" ] && TOKEN_SOURCE="gh auth token"
fi

# ------------------------------------------------------------ changed runs ---
# A run is "new" if its metrics.json is newer than the state file, or if no
# state exists. --all overrides. TAG_FILTER matches the output directory name.
CHANGED="$(python3 - "$STATE" "$ALL" "$TAG_FILTER" <<'PY'
import json, os, sys, time
from pathlib import Path
state_path, all_flag, tag_filter = Path(sys.argv[1]), sys.argv[2] == "1", sys.argv[3]
cutoff = 0.0
if state_path.is_file():
    try:
        cutoff = float(json.loads(state_path.read_text()).get("last_publish_ts", 0.0))
    except Exception:
        cutoff = 0.0
runs = []
for m in sorted(Path("output").rglob("metrics.json")):
    rel = m.parent.relative_to(Path("output"))
    if tag_filter and tag_filter not in str(rel):
        continue
    if all_flag or m.stat().st_mtime > cutoff:
        runs.append(str(rel))
print("\n".join(runs))
PY
)"
if [ -z "$CHANGED" ]; then
  say "no new run directories since the last publish"
else
  say "new runs: $(echo "$CHANGED" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- collect ---
if [ -n "$CHANGED" ]; then
  say "collecting results (status=$STATUS note='$NOTE')"
  python3 scripts/collect_results.py --status "$STATUS" --note "$NOTE" >>"$LOG" 2>&1 || \
    say "collect_results failed (continuing)"
  python3 scripts/summarize.py >>"$LOG" 2>&1 || true
else
  say "nothing new to collect"
fi

# --------------------------------------------------------------- git push ---
if [ "$DRY" = "1" ]; then
  say "dry-run: would git add results configs docs paper scripts tools tests semseg"
else
  git config user.name  >/dev/null 2>&1 || git config user.name  "CVPR2027-KD Bot"
  git config user.email >/dev/null 2>&1 || git config user.email "cvpr2027-kd@users.noreply.github.com"

  git add -A -- results configs docs paper logs_clean scripts tools tests semseg \
      STATE.md PROGRESS.md MANIFEST.md README_KD.md 2>/dev/null || true
  # Never publish weights, datasets, caches or local logs through the tree.
  git reset -q -- output data checkpoints logs logs_screen '*.pth' '*.tar.gz' 2>/dev/null || true

  # Refuse to commit anything above 50 MB: those belong in a release.
  BIG="$(git diff --cached --name-only -z | while IFS= read -r -d '' f; do
           [ -f "$f" ] || continue
           s=$(stat -c %s "$f" 2>/dev/null || echo 0)
           [ "$s" -gt 52428800 ] && echo "$s $f"
         done)"
  if [ -n "$BIG" ]; then
    say "refusing to commit files >50MB:"; echo "$BIG" | tee -a "$LOG"
    git reset -q -- $(echo "$BIG" | awk '{print $2}') 2>/dev/null || true
  fi

  if git diff --cached --quiet; then
    say "git: nothing new to commit"
  else
    MSG="results: $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$STATUS"
    [ -n "$NOTE" ] && MSG="$MSG ($NOTE)"
    git commit -q -m "$MSG" && say "git: committed $MSG"
  fi

  if [ -n "$TOKEN" ]; then
    # A shallow clone cannot rebase reliably; the repo is small, so unshallow.
    if [ -f .git/shallow ]; then
      GIT_TERMINAL_PROMPT=0 git fetch -q --unshallow "$REMOTE" 2>/dev/null || true
    fi
    ASKPASS="$(mktemp /tmp/cvpr_kd_askpass.XXXXXX)"
    TOKFILE="$(mktemp /tmp/cvpr_kd_token.XXXXXX)"
    trap 'rm -f "$ASKPASS" "$TOKFILE"' EXIT
    printf '%s\n' "$TOKEN" > "$TOKFILE"; chmod 600 "$TOKFILE"
    printf '#!/bin/sh\ncat "%s"\n' "$TOKFILE" > "$ASKPASS"; chmod 700 "$ASKPASS"
    export GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0
    git -c rebase.autoStash=true pull -q --rebase "$REMOTE" "$BRANCH" 2>>"$LOG" || true
    for attempt in 1 2 3; do
      if git push "$REMOTE" "HEAD:$BRANCH" >>"$LOG" 2>&1; then
        say "git: pushed to $REMOTE/$BRANCH"; break
      fi
      say "git: push rejected (attempt $attempt), rebasing"
      git -c rebase.autoStash=true pull -q --rebase "$REMOTE" "$BRANCH" 2>>"$LOG" || true
      sleep 5
    done
    # The token is only needed for git; releases use a curl config file.
    printf '%s\n' "$TOKEN" > "$TOKFILE"
  else
    say "git: no token; commit is local only"
  fi
fi

# --------------------------------------------------------------- releases ---
if [ "$NO_RELEASE" = "1" ] || [ -z "$CHANGED" ]; then
  say "release: skipped"
else
  if [ "$DRY" = "1" ]; then
    say "dry-run: would upload $(echo "$CHANGED" | wc -l) run(s) to GitHub releases"
  elif [ -z "$TOKEN" ]; then
    PENDING="$REPO_DIR/logs/PUBLISH_PENDING.md"
    {
      echo "# Pending release uploads"
      echo
      echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ). No GitHub token was found."
      echo
      echo "Add a token with repo scope at \`~/.config/cvpr_kd/pat\` (chmod 600), then run:"
      echo
      echo '```bash'
      echo "cd $REPO_DIR"
      echo "bash scripts/autopublish.sh --all --note 'flush pending'"
      echo '```'
      echo
      echo "Runs waiting for checkpoints:"
      echo
      for r in $CHANGED; do echo "- \`$r\`"; done
    } > "$PENDING"
    say "release: no token; wrote $PENDING"
  else
    export AUTOPUBLISH_REPO="$REPO" AUTOPUBLISH_TOKEN_FILE="$(mktemp /tmp/cvpr_kd_token.XXXXXX)"
    printf '%s\n' "$TOKEN" > "$AUTOPUBLISH_TOKEN_FILE"; chmod 600 "$AUTOPUBLISH_TOKEN_FILE"
    trap 'rm -f "$AUTOPUBLISH_TOKEN_FILE"' EXIT
    printf '%s\n' "$CHANGED" | python3 scripts/publish_release.py \
        --repo "$REPO" --token-file "$AUTOPUBLISH_TOKEN_FILE" --results results >>"$LOG" 2>&1
    RC=$?
    [ "$RC" = "0" ] && say "release: done" || say "release: failed (rc=$RC); see $LOG"
  fi
fi

# ------------------------------------------------------------------ state ---
if [ "$DRY" = "0" ]; then
  python3 - "$STATE" "$STATUS" "$NOTE" <<'PY'
import json, sys, time
from pathlib import Path
state = Path(sys.argv[1])
payload = {"last_publish_ts": time.time(), "last_status": int(sys.argv[2]),
           "last_note": sys.argv[3], "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
state.write_text(json.dumps(payload, indent=1))
PY
fi
say "finished (status=$STATUS)"
exit 0
