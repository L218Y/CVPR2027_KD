#!/usr/bin/env bash
# Publish every irreplaceable artifact to a GitHub release before the lab is
# powered down. Regular git cannot carry these: the teacher checkpoints are
# 235 MB and GitHub hard-blocks files above 100 MB, so they go to release
# assets (2 GB per asset) instead of the tree.
#
# Upload order is deliberate: best checkpoints first, then the dataset archive,
# then last checkpoints. If the upload is interrupted, the valuable half is
# already safe.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

REPO="L218Y/CVPR2027_KD"
TAG="artifacts-2026-10-24"
TOKEN=$(cat "$HOME/.config/cvpr_kd/pat")
API="https://api.github.com/repos/$REPO"
LOG="$REPO_DIR/logs/release_upload.log"
mkdir -p logs
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }

auth() { printf 'Authorization: Bearer %s' "$TOKEN"; }

# 1. Archive the dataset once (292 MB of PNGs; the 984 MB download also holds
#    duplicates and is publicly re-fetchable, so it is not archived).
DATA_TAR="$REPO_DIR/../MFNet_dataset.tar.gz"
if [ ! -f "$DATA_TAR" ]; then
  say "archiving dataset"
  tar -C "$REPO_DIR/../data" -czf "$DATA_TAR" MFNet || { say "tar failed"; exit 1; }
fi
say "dataset archive $(du -h "$DATA_TAR" | cut -f1)"

# 2. Create the release (idempotent).
REL_ID=$(curl -s -H "$(auth)" "$API/releases/tags/$TAG" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id") or "")' 2>/dev/null)
if [ -z "$REL_ID" ]; then
  BODY=$(python3 - <<'PY'
import json
body = """Checkpoints, dataset and per-class diagnostics for the CVPR 2027
cross-modal privileged distillation study, snapshotted before the lab machine
was powered down.

Contents: CMNeXt-B2 teacher (RGB+thermal, 235 MB per checkpoint), RGB students,
uniform-KD students, RAPD v2 students (rare-class protect, seeds 3407/3408/3409),
the frozen MFNet split, and the MFNet dataset archive.

Checkpoints are the `best.pth` (best validation mIoU) and `last.pth` (epoch 500)
of each run. `results/` in the repository holds metrics.json, day/night
diagnostics, per-class recoverability and the analysis.
"""
print(json.dumps({"tag_name": "artifacts-2026-10-24", "name": "Artifacts 2026-10-24",
                  "body": body, "draft": False, "prerelease": False}))
PY
)
  say "creating release $TAG"
  REL_ID=$(curl -s -X POST -H "$(auth)" -H "Accept: application/vnd.github+json" \
    -d "$BODY" "$API/releases" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id") or ""); ' 2>/dev/null)
fi
[ -z "$REL_ID" ] && { say "release creation failed"; exit 1; }
say "release id $REL_ID"

UPLOAD="https://uploads.github.com/repos/$REPO/releases/$REL_ID/assets"

upload() {
  local f="$1" name
  name=$(basename "$f")
  if curl -s -H "$(auth)" "$API/releases/$REL_ID/assets?per_page=100" \
      | grep -q "\"name\": \"$name\""; then
    say "skip (present) $name"
    return 0
  fi
  local t0=$(date +%s)
  curl -s -X POST -H "$(auth)" -H "Content-Type: application/octet-stream" \
    --retry 3 --retry-delay 10 --max-time 5400 \
    --data-binary @"$f" "$UPLOAD?name=$name" > /tmp/asset_$name.json 2>>"$LOG"
  local rc=$? dt=$(( $(date +%s) - t0 ))
  local state=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("state","?"), d.get("size",0))' /tmp/asset_$name.json 2>/dev/null)
  say "uploaded $name rc=$rc ${dt}s state=$state"
}

# 3. Upload in priority order.
for d in output/*/; do
  [ -f "$d/best.pth" ] && upload "$d/best.pth"
done
upload "$DATA_TAR"
for d in output/*/; do
  [ -f "$d/probe.pth" ] && upload "$d/probe.pth"
done
for d in output/*/; do
  [ -f "$d/last.pth" ] && upload "$d/last.pth"
done

# 4. Verify every asset's size against the local file.
curl -s -H "$(auth)" "$API/releases/$REL_ID/assets?per_page=100" \
  | python3 -c '
import json, os, sys
assets = json.load(sys.stdin)
bad = 0
for a in assets:
    local = None
    for root, _, files in os.walk("output"):
        if a["name"] in files:
            local = os.path.join(root, a["name"])
            break
    if local is None:
        path = os.path.join("..", a["name"])
        local = path if os.path.exists(path) else None
    ok = local and os.path.getsize(local) == a["size"]
    bad += 0 if ok else 1
    print("%-28s remote %10d  %s" % (a["name"], a["size"], "ok" if ok else "MISMATCH"))
print("assets:", len(assets), "mismatches:", bad)
' | tee -a "$LOG"

say "done"
