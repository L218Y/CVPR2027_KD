#!/usr/bin/env bash
# Launch queued arms as GPUs free up, without needing a live session.
#
# One arm per free GPU: a GPU is only used when it has room for another ~12 GB
# job, and a per-arm marker prevents relaunching the same arm after a restart.
#
#   nohup setsid bash scripts/queue_remaining.sh > logs_clean/queue.log 2>&1 &
set -u
cd /data/lab/CVPR2027_KD || exit 1
mkdir -p .queue logs_clean

QUEUE=(
  "1:configs/kd_mfnet_uniform_rgbteacher.yaml:p1_uniformrgbteacher:3407"
  "0:configs/kd_mfnet_correctness_fg.yaml:p1_correctnessfg:3407"
)

for item in "${QUEUE[@]}"; do
  IFS=: read -r gpu cfg tag seed <<< "$item"
  done_marker=".queue/${tag}_s${seed}.done"
  start_marker=".queue/${tag}_s${seed}.started"
  if [ -f "$done_marker" ] || [ -f "$start_marker" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $tag s$seed (already handled)"
    continue
  fi

  echo "[$(date -u +%H:%M:%S)] waiting for GPU $gpu to reach <12100 MiB for $tag s$seed"
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [ "${used:-99999}" -lt 12100 ]; then
      break
    fi
    sleep 60
  done

  echo "[$(date -u +%H:%M:%S)] GPU $gpu free (${used} MiB), launching $tag s$seed"
  bash scripts/run_arm.sh "$gpu" "$cfg" "$tag" "$seed" >> logs_clean/queue.log 2>&1
  touch "$start_marker"
  sleep 90   # let the allocator claim its memory before the next check
done
echo "[$(date -u +%H:%M:%S)] queue drained"
