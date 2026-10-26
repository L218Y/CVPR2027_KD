# Lab handoff (2026-10-26)

## Critical correction

Earlier runs labelled "thermal-only" (`configs/min/lr{2,4,8}_teacher.yaml`) were
**RGB-T fusion** models: their `MODALS` were `['img', 'thermal']` and their checkpoints
have 58.7M parameters. That label was wrong and must not be used in the paper.

True thermal-only experts were retrained with `MODALS: ['thermal']`
(`configs/min/lr{2,4,8}_thermalonly.yaml`, 25.8M parameters, `tools/train_mm.py`).
Their results are the valid budget curve.

## Completed true thermal-only experts (MFNet official 392-image test)

| budget | official mIoU | fg mIoU | night person IoU | checkpoint |
|---|---:|---:|---:|---|
| LR8 60x80 | 32.31 | 24.47 | 57.21 | `output/min/lr8_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch120_32.31.pth` |
| LR4 120x160 | 41.67 | 34.86 | 62.41 | `output/min/lr4_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch100_41.67.pth` |
| LR2 240x320 | 42.80 | 36.10 | 64.73 | `output/min/lr2_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch100_42.8.pth` |

RGB-only reference: 56.38 / 51.28 / 45.24. Full-res teacher: 61.28 / — / 65.74.
LR4 recovers 84% of the teacher's night-person gain; LR2 adds only 2.3 points for 4x
the pixels, so the knee remains LR4.

## Gates against the true thermal-only expert

LR4, three seeds, person-or variant:
57.39, 57.45, 57.60 official; 61.42, 61.97, 63.39 night person.
Mean 57.48 +/- 0.11 official, 52.50 +/- 0.12 fg, 62.26 +/- 1.02 night person.

LR4 full (unconstrained) gate: 57.58, 57.75, 57.91; mean 57.74.
The person constraint costs ~0.26 overall but keeps every non-person class within
0.06 IoU; the full gate changes car_stop by -1.96 and guardrail by +1.60. This is now
a safety-guarantee result, not a large accuracy win over the full gate.

LR8 seed 3407 only:
- clean person-or 56.2x official, night person 58.38; full 56.48.
- degraded RGB: RGB 47.68, full 51.27, person-or 49.34; thermal-only 41.67.
LR2 gates were not run before shutdown.

Raw JSON:
`results/protocol/THERMALONLY_lr{8,4,2}.json`,
`results/protocol/UNETGATE_thermalonly_lr4_s340{7,8,9}.json`,
`results/protocol/UNETGATE_degraded_thermalonly_lr4_s340{7,8,9}.json`,
`results/protocol/UNETGATE_thermalonly_lr8_s3407.json`.

## State of the paper

`paper/sections/*.tex` still contains the old (wrong) RGB-T-fusion "thermal-only"
numbers (e.g. 55.80) and old gate means 57.57 / 61.36. **Do not submit it as is.**
The correct replacement values are above. The methods text must say the second expert
is thermal-only (`f_b(t)`), which is now true; the old code was not.

Recommended paper change: use person-or as the main deployment (57.48 / 62.26, all
non-person classes within 0.06) and report the full gate (57.74) as an unconstrained
upper bound that violates the non-person fidelity guarantee. The title "person-aware"
remains defensible as a safety constraint, not an accuracy trick.

## Resume commands

```bash
cd /data/lab/CVPR2027_KD
export PYTHONPATH="$PWD:$PWD/tools:$PWD/scripts"
# finish LR8 gates and run LR2 gates (script loops 4,8,2; delete finished LR4 JSONs or
# edit the loop to start at 8 if you do not want to repeat LR4):
bash scripts/run_thermalonly_gates.sh
# hard-aug ablation with true thermal-only B, one seed:
python3 scripts/train_unet_gate.py \
  --cfg-a configs/min/ce120.yaml \
  --ckpt-a output/min/ce120/MFNet_CMNeXt-B2_i_rgb_s3407_p0ce120_s3407/best.pth \
  --cfg-b configs/min/lr4_thermalonly.yaml \
  --ckpt-b output/min/lr4_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch100_41.67.pth \
  --b-modals thermal --p-hard 0 \
  --train-root data/MFNet_lr4_gateval --test-root data/MFNet_lr4 \
  --out-dir output/gate/unet_thermalonly_lr4_nohard --seed 3407
# compute table: patch scripts/measure_compute.py to cfg_b=lr4_thermalonly,
# model_b=build_model(..., ['thermal']), load epoch100, fb=flops(b,[th]), joint pb=b([th]).
```

Checkpoints are archived in GitHub release `thermalonly-2026-10-26` (assets:
`thermalonly_lr{8,4,2}_*.pth`, `rgb_ce120_best.pth`, `gate_lr4_s340{7,8,9}.pth`).
They are not in git. If the lab disk is lost, retrain each with:

```bash
bash scripts/run_thermalonly.sh 8
bash scripts/run_thermalonly.sh 4
bash scripts/run_thermalonly.sh 2
```

## Dataset/layout

`data/MFNet_lr{2,4,8}` and `data/MFNet_lr{2,4,8}_gateval` remain on the lab disk;
they are not in git. `tools/make_lr_dataset.py` rebuilds them.
