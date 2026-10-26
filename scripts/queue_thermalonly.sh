#!/usr/bin/env bash
# Queue LR8 and LR2 thermal-only training after LR4 finishes.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while pgrep -f "train_mm.py --cfg configs/min/lr4_thermalonly.yaml" >/dev/null; do
  sleep 30
done
bash scripts/run_thermalonly.sh 8
bash scripts/run_thermalonly.sh 2
echo "thermal-only queue done"
