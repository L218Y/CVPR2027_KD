#!/usr/bin/env bash
set -euo pipefail
cd /data/lab/CVPR2027_KD
T=output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth
run () { # cfg ckpt out teacher?
  if [ -n "${4:-}" ]; then
    python3 scripts/eval_protocol.py --cfg "$1" --ckpt "$2" --teacher-ckpt "$T" --out "results/protocol/$3.json" 2>&1 | grep -v Warning | tail -9
  else
    python3 scripts/eval_protocol.py --cfg "$1" --ckpt "$2" --out "results/protocol/$3.json" 2>&1 | grep -v Warning | tail -9
  fi
}
run configs/teacher_mfnet.yaml $T EXIST_teacher_s3407_best ""
run configs/teacher_mfnet.yaml output/MFNet_CMNeXt-B2_it_teacher_s3407/last.pth EXIST_teacher_s3407_last ""
run configs/rgb_mfnet.yaml output/MFNet_CMNeXt-B2_i_rgb_s3407/last.pth EXIST_rgb_s3407_last p
run configs/kd_mfnet_uniform.yaml output/MFNet_CMNeXt-B2_it_uniform_s3407/best.pth EXIST_uniform_s3407_best p
run configs/kd_mfnet_uniform.yaml output/MFNet_CMNeXt-B2_it_uniform_s3407/last.pth EXIST_uniform_s3407_last p
for s in 3407 3408 3409; do
  run configs/kd_mfnet_v2.yaml output/MFNet_CMNeXt-B2_it_recoverability_s${s}_v2/best.pth EXIST_v2_s${s}_best p
  run configs/kd_mfnet_v2.yaml output/MFNet_CMNeXt-B2_it_recoverability_s${s}_v2/last.pth EXIST_v2_s${s}_last p
done
