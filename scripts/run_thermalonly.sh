#!/usr/bin/env bash
# Train a true thermal-only expert at one MFNet budget.
#   bash scripts/run_thermalonly.sh 4
set -u
s="${1:?usage: $0 2|4|8}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/tools:$PWD/scripts"
mkdir -p logs_p0
torchrun --nproc_per_node=2 --master_port=$((29630 + s)) \
  tools/train_mm.py --cfg "configs/min/lr${s}_thermalonly.yaml" \
  > "logs_p0/lr${s}_thermalonly.log" 2>&1
echo "thermal-only LR${s} finished with $?"
