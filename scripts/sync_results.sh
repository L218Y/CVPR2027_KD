#!/usr/bin/env bash
# Commit collected results and push them to GitHub.
#
# The token is read from $PAT_FILE (default ~/.config/cvpr_kd/pat) and is fed
# to git through a temporary askpass helper.  It is never written into the
# repository, the git config or the process arguments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PAT_FILE="${PAT_FILE:-$HOME/.config/cvpr_kd/pat}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
NOTE="${1:-}"

if [ ! -s "$PAT_FILE" ]; then
  echo "sync: token file '$PAT_FILE' not found or empty" >&2
  exit 2
fi

git config user.name  >/dev/null 2>&1 || git config user.name  "CVPR2027-KD Bot"
git config user.email >/dev/null 2>&1 || git config user.email "cvpr2027-kd@users.noreply.github.com"

git add -A
git reset -q -- PAT .secrets 2>/dev/null || true

if git diff --cached --quiet; then
  echo "sync: nothing new to commit"
else
  MSG="results: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
  [ -n "$NOTE" ] && MSG="$MSG ($NOTE)"
  git commit -q -m "$MSG"
fi

ASKPASS="$(mktemp /tmp/cvpr_kd_askpass.XXXXXX)"
trap 'rm -f "$ASKPASS"' EXIT
printf '#!/bin/sh\ncat "%s"\n' "$PAT_FILE" > "$ASKPASS"
chmod 700 "$ASKPASS"

echo "sync: pushing to $REMOTE/$BRANCH"
GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 git push "$REMOTE" "HEAD:$BRANCH"
echo "sync: done"
