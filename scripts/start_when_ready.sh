#!/usr/bin/env bash
# Wait for the dataset archive, verify it, unpack it, then start the P0 probes
# and the fallback publisher. Safe to run in the background:
#
#   nohup bash scripts/start_when_ready.sh > logs/start_when_ready.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs logs_p0 data

TARGET="data/MFNet_dataset.tar.gz"
EXPECTED=282482751
echo "[$(date -u +%H:%M:%S)] waiting for $TARGET ($EXPECTED bytes)"

if [ "$(stat -c %s "$TARGET" 2>/dev/null || echo 0)" -lt "$EXPECTED" ] && \
   ! pgrep -f "curl .*MFNet_dataset.tar.gz" >/dev/null; then
  echo "[$(date -u +%H:%M:%S)] starting dataset download"
  nohup curl -L --retry 20 --retry-delay 5 --connect-timeout 30 \
    --speed-limit 20000 --speed-time 30 -C - \
    -o "$TARGET" "https://github.com/L218Y/CVPR2027_KD/releases/download/artifacts-2026-10-24/MFNet_dataset.tar.gz" \
    >> logs/dataset_dl.log 2>&1 &
  echo $! > logs/dataset_dl.pid
fi

for _ in $(seq 1 240); do
  SIZE="$(stat -c %s "$TARGET" 2>/dev/null || echo 0)"
  [ "$SIZE" -ge "$EXPECTED" ] && break
  sleep 20
done
SIZE="$(stat -c %s "$TARGET" 2>/dev/null || echo 0)"
if [ "$SIZE" -ne "$EXPECTED" ]; then
  echo "[$(date -u +%H:%M:%S)] timeout: $TARGET is $SIZE bytes, expected $EXPECTED"
  exit 1
fi

WANT="$(awk '/MFNet_dataset/{print $1}' results/ARTIFACTS.sha256 | head -1)"
GOT="$(sha256sum "$TARGET" | awk '{print $1}')"
if [ -n "$WANT" ] && [ "$WANT" != "$GOT" ]; then
  echo "[$(date -u +%H:%M:%S)] checksum mismatch: want $WANT got $GOT"
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] checksum ok; unpacking"
tar -xzf "$TARGET" -C data || { echo "tar failed"; exit 1; }

cp -f results/splits/train.txt data/MFNet/train.txt
cp -f results/splits/test.txt  data/MFNet/test.txt
N="$(ls data/MFNet/rgb 2>/dev/null | wc -l)"
echo "[$(date -u +%H:%M:%S)] rgb images: $N (train $(wc -l < data/MFNet/train.txt), test $(wc -l < data/MFNet/test.txt))"
[ "$N" -ge 1176 ] || { echo "not enough images"; exit 1; }

chmod +x scripts/*.sh scripts/publish_release.py 2>/dev/null || true

echo "[$(date -u +%H:%M:%S)] starting watchdog publisher"
nohup bash scripts/watch_publish.sh 180 > logs/watch_publish.log 2>&1 &
echo $! > logs/watch_publish.pid

echo "[$(date -u +%H:%M:%S)] starting P0 probes"
nohup bash scripts/run_p0_probes.sh > logs_p0/queue.log 2>&1 &
echo $! > logs_p0/queue.pid
echo "[$(date -u +%H:%M:%S)] launched P0 pid $(cat logs_p0/queue.pid), watchdog pid $(cat logs/watch_publish.pid)"
