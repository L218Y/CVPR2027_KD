#!/usr/bin/env bash
# Train and evaluate the person-aware gate against a true thermal-only expert.
#   bash scripts/run_thermalonly_gates.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/tools:$PWD/scripts"
mkdir -p logs_p0 results/protocol

RGB_CKPT="output/min/ce120/MFNet_CMNeXt-B2_i_rgb_s3407_p0ce120_s3407/best.pth"
for s in 4 8 2; do
  B_CKPT="$(find "output/min/lr${s}_thermalonly" -name '*epoch120*.pth' ! -name '*checkpoint*' | head -1)"
  if [ -z "$B_CKPT" ]; then echo "missing thermal-only LR${s} checkpoint"; continue; fi
  for seed in 3407 3408 3409; do
    tag="thermalonly_lr${s}_s${seed}"
    out="output/gate/unet_${tag}"
    log="logs_p0/gate_${tag}.log"
    python3 scripts/train_unet_gate.py \
      --cfg-a configs/min/ce120.yaml --ckpt-a "$RGB_CKPT" \
      --cfg-b "configs/min/lr${s}_thermalonly.yaml" --ckpt-b "$B_CKPT" \
      --b-modals thermal \
      --train-root "data/MFNet_lr${s}_gateval" --test-root "data/MFNet_lr${s}" \
      --out-dir "$out" --seed "$seed" > "$log" 2>&1
    GATE="$out/best_gate.pth"
    python3 scripts/eval_unet_gate_variants.py \
      --cfg-a configs/min/ce120.yaml --ckpt-a "$RGB_CKPT" \
      --cfg-b "configs/min/lr${s}_thermalonly.yaml" --ckpt-b "$B_CKPT" \
      --b-modals thermal --gate "$GATE" --root "data/MFNet_lr${s}" \
      --out "results/protocol/UNETGATE_${tag}.json" >> "$log" 2>&1
    python3 scripts/eval_unet_gate_variants.py \
      --cfg-a configs/min/ce120.yaml --ckpt-a "$RGB_CKPT" \
      --cfg-b "configs/min/lr${s}_thermalonly.yaml" --ckpt-b "$B_CKPT" \
      --b-modals thermal --gate "$GATE" --root "data/MFNet_lr${s}" --degrade \
      --out "results/protocol/UNETGATE_degraded_${tag}.json" >> "$log" 2>&1
    echo "finished $tag"
  done
done
