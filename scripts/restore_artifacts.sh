#!/usr/bin/env bash
# Rebuild output/ and data/ from the GitHub release.
#
# Checkpoints are stored as release assets because they exceed GitHub's 100 MB
# file limit. This script pulls them back, restores the run directories they
# belong to, and verifies every file against results/ARTIFACTS.sha256.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

REPO="L218Y/CVPR2027_KD"
TAG="artifacts-2026-10-24"
TOKEN="${GITHUB_TOKEN:-}"
[ -f "$HOME/.config/cvpr_kd/pat" ] && TOKEN="${TOKEN:-$(cat "$HOME/.config/cvpr_kd/pat")}"

auth=()
[ -n "$TOKEN" ] && auth=(-H "Authorization: Bearer $TOKEN")

mkdir -p output data
mapfile -t URLS < <(curl -s "${auth[@]}" \
  "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
  | python3 -c 'import json,sys
for a in json.load(sys.stdin)["assets"]:
    print(a["name"], a["browser_download_url"])')

for row in "${URLS[@]}"; do
  name="${row%% *}"
  url="${row#* }"
  if [ -f "../$name" ] || find output -name "$name" -print -quit | grep -q .; then
    echo "have $name"
    continue
  fi
  case "$name" in
    MFNet_dataset.tar.gz) dest="data/$name" ;;
    *_best.pth|*_last.pth|*_probe.pth)
      run="${name%_*}"
      ext="${name##*_}"
      dest="output/$run/$ext"
      mkdir -p "output/$run" ;;
    *) dest="$name" ;;
  esac
  echo "fetch $name -> $dest"
  curl -sL "${auth[@]}" -o "$dest" "$url"
done

echo "verifying checksums"
sha256sum -c results/ARTIFACTS.sha256 --quiet
echo "artifacts restored"
