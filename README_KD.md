# Recoverability-Aware Privileged Distillation (RAPD)

Train with multimodal sensors, deploy with RGB only. This repository is a fork of
[CMNeXt / DELIVER](https://github.com/jamycheung/DELIVER) (Apache-2.0) that adds a
privileged-modality distillation framework: a frozen multimodal teacher supervises an
RGB-only student, but the supervision is reweighted per pixel by how much of the
teacher's fused evidence is actually recoverable from RGB.

## Method

For every pixel the student feature `f_s` and teacher feature `f_t` are compared by a
light probe `g` (1x1 conv, 128 hidden channels), trained with a cosine objective:

```
r = (cos(g(f_s), f_t) + 1) / 2   in [0, 1]
```

`r` is detached, so the student cannot raise its own supervision weight. The distillation
target is a convex mixture of the teacher posterior `p_t` and a conservative target `p_u`
that backs off towards the running class prior `pi`:

```
p_u = (1 - eps) * p_t + eps * pi,        eps = eps_max * (1 - r)
L_kd = r * KL(p_t || p_s) + (1 - r) * KL(p_u || p_s)
```

Where the teacher saw privileged evidence the student never gets (`r -> 0`), the student is
asked for the class prior instead of an unreachable posterior; where the evidence is
recoverable (`r -> 1`) it distils the teacher exactly. A feature term
`1 - cos(proj(f_s), f_t)`, also weighted by `r`, is added with weight `LAMBDA_FEAT`. For the
first `WARMUP_EPOCHS` epochs the loss falls back to plain uniform KD so the probe is trained
on meaningful features before it starts reweighting anything.

Modes (`KD.MODE`): `recoverability` (full method), `uniform` (Hinton KD baseline),
`conservative_only` (prior back-off with `r = 0`), `hard` (teacher argmax, no softening).

## Install

```bash
python3 -m pip install timm fvcore opencv-python-headless tabulate tensorboard
# mmcv is optional: the fork falls back to local norm/conv builders
```

Backbone weights: `checkpoints/pretrained/segformer/mit_b2.pth` from
[huggingface.co/InSAI-Lab/CMNeXt](https://huggingface.co/InSAI-Lab/CMNeXt/tree/main/pretrained/segformers).

## Data

```bash
python3 scripts/prepare_mfnet.py --src data/mfnet_raw/MFNet --dst data/MFNet
```

Reads the official `train.txt` / `val.txt` lists (784 / 392 images for MFNet, augmented
duplicates dropped) and writes `data/MFNet/{rgb,ther,labels}` plus our own split files.

## Runs

```bash
bash scripts/run_and_sync.sh configs/teacher_mfnet.yaml "mfnet teacher"
bash scripts/run_and_sync.sh configs/kd_mfnet.yaml      "mfnet rapd"
```

`run_and_sync.sh` runs the trainer, then always collects metrics, logs and figures into
`results/` and pushes them to GitHub, even when training crashes. The token is read from
`~/.config/cvpr_kd/pat` through a temporary askpass helper and never enters the repository.

## Analysis

```bash
python3 tools/analyze.py --cfg configs/kd_mfnet.yaml \
    --ckpt output/MFNet_CMNeXt-B2_img_recoverability_s3407/best.pth \
    --out results/MFNet_CMNeXt-B2_img_recoverability_s3407 --save-rmaps 8
python3 scripts/summarize.py        # writes results/SUMMARY.md with PGR
```

Metrics: mIoU / mAcc / aAcc, per-class IoU, expected calibration error, hallucination rate
(pixels predicted as a class absent from the image ground truth), and
PGR = (mIoU_student - mIoU_rgb) / (mIoU_teacher - mIoU_rgb).

## Layout

```
semseg/kd/losses.py       RecoverabilityKD, UniformKD
semseg/kd/probe.py        FeatureProbe, RecoverabilityHead
tools/train_kd.py         unified trainer (teacher / kd modes)
tools/analyze.py          calibration, hallucination, r-maps
scripts/prepare_mfnet.py  MFNet -> data/MFNet
scripts/summarize.py      results/SUMMARY.md with PGR
scripts/collect_results.py, sync_results.sh, run_and_sync.sh
configs/teacher_mfnet.yaml, kd_mfnet.yaml, teacher_nyu.yaml, kd_nyu.yaml, kd_deliver.yaml
tests/test_kd.py
```

## Results

`results/SUMMARY.md` is regenerated after every run and is the single source of truth for
the numbers in the paper. Runs planned:

| id | dataset | teacher | student | mode |
|---|---|---|---|---|
| T1 | MFNet | CMNeXt-B2 RGB-T | — | teacher |
| S0 | MFNet | — | CMNeXt-B2 RGB | no distillation |
| S1 | MFNet | T1 | CMNeXt-B2 RGB | uniform |
| S2 | MFNet | T1 | CMNeXt-B2 RGB | recoverability |
| S3 | MFNet | T1 | CMNeXt-B2 RGB | conservative_only |
| S4 | MFNet | T1 | CMNeXt-B2 RGB | eps_max sweep |
| T2 | NYUv2 | CMNeXt-B2 RGB-D | — | teacher |
| S5 | NYUv2 | T2 | CMNeXt-B2 RGB | recoverability |

## Verify

```bash
PYTHONPATH=. python3 tests/test_kd.py     # loss equivalence, masking, gradients
```
