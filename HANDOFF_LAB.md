# Lab handoff (2026-10-27)

## Critical correction

Earlier runs labelled "thermal-only" (`configs/min/lr{2,4,8}_teacher.yaml`) were
**RGB-T fusion** models: their `MODALS` were `['img', 'thermal']` and their checkpoints
have 58.7M parameters. That label was wrong and must not be used in the paper.

True thermal-only experts use `MODALS: ['thermal']`
(`configs/min/lr{2,4,8}_thermalonly.yaml`, 25.8M parameters, `tools/train_mm.py`).
All gate, early-fusion, and compute numbers below use these true experts.

## Completed true thermal-only experts (MFNet official 392-image test)

| budget | official mIoU | fg mIoU | night person IoU | checkpoint |
|---|---:|---:|---:|---|
| LR8 60x80 | 32.31 | 24.47 | 57.21 | `output/min/lr8_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch120_32.31.pth` |
| LR4 120x160 | 41.67 | 34.86 | 62.41 | `output/min/lr4_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch100_41.67.pth` |
| LR2 240x320 | 42.80 | 36.10 | 64.73 | `output/min/lr2_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch100_42.8.pth` |

RGB-only reference: 56.38 / 51.28 / 45.24. Full-res RGB-T teacher: 61.28 / — / 65.74.
LR4 recovers 84% of the teacher's night-person gain; LR2 adds only 0.06 overall and
1.03 night-person points for 4x pixels, so the knee is LR4.

Raw JSON: `results/protocol/THERMALONLY_lr{8,4,2}.json`.

## Gates, three seeds each, true thermal-only B

| budget | official | fg | night person | full gate official |
|---|---:|---:|---:|---:|
| LR8 | 57.19 +/- 0.07 | 52.19 +/- 0.08 | 58.63 +/- 0.23 | 57.20 |
| LR4 | 57.48 +/- 0.11 | 52.50 +/- 0.12 | 62.26 +/- 1.02 | 57.74 |
| LR2 | 57.54 +/- 0.06 | 52.57 +/- 0.07 | 63.29 +/- 0.64 | 57.84 |

LR4 person-or seeds: 57.39, 57.45, 57.60 official; 61.42, 61.97, 63.39 night.
LR4 full seeds: 57.58, 57.75, 57.91.
The person constraint costs ~0.26 overall but keeps every non-person class within
0.08 (seven of eight <=0.06). Full gate changes car_stop -1.96 and guardrail +1.60.
This is a safety-guarantee result, not a large accuracy win over the full gate.

Raw JSON: `results/protocol/UNETGATE_thermalonly_lr{8,4,2}_s340{7,8,9}.json` and
`..._degraded_...json`.

## Hard-augmentation ablation (LR4, seed 3407)

person-or: hard 57.39 clean / 49.34 degraded; nohard 57.47 / 49.27 (neutral).
full gate: hard 57.58 / 51.27; nohard 57.72 / 49.31 (hard aug helps only the
unconstrained gate under corruption, at -0.14 clean).

Raw JSON: `results/protocol/UNETGATE_nohard_thermalonly_lr4_s3407.json` and
`..._degraded_...json`.

## Single-model early fusion

`semseg/models/earlyfusion.py`, config `configs/min/lr4_earlyfusion.yaml`.
Six-channel input, first conv expanded from RGB pre-trained filters.
Final 120ep: 55.51 overall / 50.30 fg / 60.57 night person;
25.8M / 46.4G / 11.09 ms.
Checkpoint: `output/min/lr4_earlyfusion/MFNet_CMNeXt-B2_it/EarlyFusionCMNeXt_CMNeXt-B2_MFNet_epoch120_55.51.pth`.
Raw JSON: `results/protocol/EARLYFUSION_lr4.json`.

Gate vs early fusion: +1.97 overall / +1.69 night person at ~2.17x compute.
Gate vs single full fusion: +1.68 / +0.38 and 24.0 vs 29.8 ms (19% faster).

## Measured compute (`results/COMPUTE_thermalonly.json`)

| model | params M | GFLOPs | latency ms |
|---|---:|---:|---:|
| RGB | 25.78 | 46.20 | 11.08 |
| thermal-only | 25.78 | 46.20 | 11.03 |
| gate | 1.93 | 45.57 | 3.18 |
| early fusion | 25.79 | 46.38 | 11.09 |
| joint gate | 53.49 | 137.97 | 23.999 |

Joint gate = 2.17x RGB, 0.81x the 58.7M full-fusion single model (29.8 ms).

## PST900 transfer: negative, do not revive without new evidence

- PST900 RGB: 78.92 overall / 73.75 survivor.
- LR4 thermal-only at 70 epochs: 48.26 / 37.88.
- LR2 thermal-only at 120 epochs: 38.92 / 24.69 (worse than LR4).
- No gate trained on PST900. Use as a limitation: the budget knee is dataset-dependent;
  small low-contrast survivors are not recovered at 1/16.

Raw logs: `logs_p0/pst900_lr{2,4}_thermalonly.log`; results under `results/protocol/PST900_*`.

## Full-resolution thermal-only teacher (completed)

Trained 120 epochs with `configs/min/full_thermalonly.yaml` (GPU0, single process).
Best checkpoint: `output/min/full_thermalonly/MFNet_CMNeXt-B2_t/CMNeXt_CMNeXt-B2_MFNet_epoch80_46.16.pth`
(also `best.pth`). Official 392-image test: **46.16 overall / 65.61 night person**.
This validates the 84% recovery denominator: a full-resolution thermal-only B2 is
65.61, close to the RGB-T fusion teacher's 65.74, so RGB access does not inflate it.
Log: `logs_p0/full_thermalonly.log`; eval: `output/min/full_thermalonly/MFNet_CMNeXt-B2_t/eval_*.txt`.

## Paper state (local edits, 2026-10-27)

`paper/main.tex` is CVPR review style (`cvpr.sty` patched to literal anonymous text).
`paper/sections/*.tex` now uses the true numbers above. Tables 1-4:
budget/gates, variants, degradation, single-model compute. Figure 1 is the night-person
budget curve; Figure 2 is per-class delta. Figure 3 was removed as redundant with Table 3.
Nine pages total (eight content + references); clean compile, no overfull boxes.

Local figures were regenerated from `scripts/make_figures_static.py` because the result
JSONs are lab-only. The reproducible lab script is `scripts/make_figures.py`.

Do not reuse old 55.80/57.57/61.36/3.9x numbers.

## Resume commands

```bash
cd /data/lab/CVPR2027_KD
export PYTHONPATH="$PWD:$PWD/tools:$PWD/scripts"
# evaluate fullres teacher once best.pth exists:
python3 tools/val_mm.py --cfg configs/min/full_thermalonly.yaml
# regenerate figures from lab JSONs:
python3 scripts/make_figures.py
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
