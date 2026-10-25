#!/usr/bin/env bash
# Rebuild output/ and data/ from the GitHub release.
#
# Checkpoints are stored as release assets because they exceed GitHub's 100 MB
# file limit. This script pulls them back, restores the run directories they
# belong to, unpacks the prepared dataset and verifies every file against
# results/ARTIFACTS.sha256.
#
# Portable to bash 3.2 (the macOS default): no mapfile/readarray.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

REPO="L218Y/CVPR2027_KD"
TAG="artifacts-2026-10-24"
TOKEN="${GITHUB_TOKEN:-}"
[ -f "$HOME/.config/cvpr_kd/pat" ] && TOKEN="${TOKEN:-$(cat "$HOME/.config/cvpr_kd/pat")}"

auth=()
[ -n "$TOKEN" ] && auth=(-H "Authorization: Bearer $TOKEN")

# `data` used to be committed as an absolute-path symlink, which makes a clone
# created at that very path self-referential. Replace it with a real directory.
if [ -L data ]; then
  echo "replacing committed symlink data -> $(readlink data) with a real directory"
  rm -f data
fi
mkdir -p output data

listing="$(mktemp)"
trap 'rm -f "$listing"' EXIT
curl -s "${auth[@]}" "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
  | python3 -c 'import json,sys
for a in json.load(sys.stdin).get("assets", []):
    print(a["name"], a["browser_download_url"])' > "$listing"

if [ ! -s "$listing" ]; then
  echo "no assets returned for $TAG (private repo? set GITHUB_TOKEN)" >&2
  exit 1
fi

while read -r name url; do
  [ -z "$name" ] && continue
  case "$name" in
    MFNet_dataset.tar.gz) dest="data/$name" ;;
    *_best.pth|*_last.pth|*_probe.pth)
      run="$(echo "$name" | sed -E 's/_(best|last|probe)\.pth$//')"
      ext="$(echo "$name" | sed -E 's/^.*_//')"
      dest="output/$run/$ext"
      mkdir -p "output/$run" ;;
    *) dest="$name" ;;
  esac
  if [ -f "$dest" ]; then
    echo "have $dest"
    continue
  fi
  echo "fetch $name -> $dest"
  curl -sL "${auth[@]}" -o "$dest" "$url"
done < "$listing"

echo "verifying checksums"
# ARTIFACTS.sha256 was written from inside results/ for the checkpoints but from
# inside data/ for the dataset archive (hence a `../` prefix). Normalise the
# paths so one check from the repo root covers every entry.
norm="$(mktemp)"
trap 'rm -f "$listing" "$norm"' EXIT
awk '{p=$2; sub(/^\.\.\//, "data/", p); print $1 "  " p}' results/ARTIFACTS.sha256 > "$norm"
sha256sum -c "$norm" --quiet

if [ -f data/MFNet_dataset.tar.gz ] && [ ! -d data/MFNet/rgb ]; then
  echo "unpacking prepared dataset"
  tar -xzf data/MFNet_dataset.tar.gz -C data
fi

if [ ! -f data/MFNet/train.txt ]; then
  cp results/splits/train.txt results/splits/test.txt data/MFNet/ 2>/dev/null || true
fi

echo "artifacts restored: $(ls -1 output | wc -l | tr -d ' ') runs, $(ls -1 data/MFNet/rgb 2>/dev/null | wc -l | tr -d ' ') rgb images"
