#!/usr/bin/env bash
# Evaluate finished P1 arms under the single protocol, best and last.
cd /data/lab/CVPR2027_KD || exit 1
T=output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth

cfg_for () {
  case "$1" in
    ce)          echo configs/rgb_mfnet.yaml ;;
    uniform)     echo configs/kd_mfnet_uniform.yaml ;;
    correctness) echo configs/kd_mfnet_correctness.yaml ;;
    classbalance) echo configs/kd_mfnet_classbalance.yaml ;;
    both)        echo configs/kd_mfnet_both.yaml ;;
    cbce)        echo configs/rgb_mfnet_cbce.yaml ;;
  esac
}

for tag in "$@"; do
  cfg=$(cfg_for "$tag")
  if [ -z "$cfg" ]; then echo "unknown arm $tag"; continue; fi
  dir=$(find output -maxdepth 1 -type d -name "*_s3407_p1_${tag}" | head -1)
  if [ -z "$dir" ]; then echo "skip $tag (no run dir)"; continue; fi
  for ck in best last; do
    if [ ! -f "$dir/$ck.pth" ]; then echo "skip $tag $ck"; continue; fi
    echo "== $tag $ck"
    python3 scripts/eval_protocol.py --cfg "$cfg" --ckpt "$dir/$ck.pth" \
      --teacher-ckpt "$T" --out "results/protocol/P1_${tag}_${ck}.json" 2>&1 \
      | grep -E '"miou_official"|night_person_iou|"epoch"' || true
  done
done
