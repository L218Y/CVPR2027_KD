# Evaluation and comparison protocol (P1)

Everything in this file is fixed before the P1 runs are looked at. Any number
reported for P1 comes from these two scripts, not from the trainer's own
validation loop, not from `scripts/diag_dn.py`, and not from a checkpoint chosen
per arm after seeing results.

## Data

| item | value |
|---|---|
| dataset | MFNet as distributed by its authors, `_flip` copies dropped |
| usable pairs | 1176 (train 784 / eval 392) |
| split | official `train.txt` / `val.txt` |
| train sha256 | `4bb4c7a698c6e2dfad0f40089830f58546523ac513e24e0dae8c2aaa7c857d91` |
| eval sha256 | `446e9dbcee6a66bb95331dc704a2e184b723e5dd8aded861dbf76c9968879180` |

`semseg/datasets/mfnet.py` maps `split='val'` onto `test.txt` (written from the
official `val.txt`). The same 392 images therefore both select the best epoch and
are reported, which is why:

* **primary comparison uses the fixed epoch-500 checkpoint (`last.pth`)**, which
  involves no selection at all;
* `best.pth` is reported only as a secondary number, labelled as
  validation-selected, and never mixed with `last` in the same column.

An independent final test set does not exist for this dataset in this workspace.
Nothing here should be described as a held-out test result.

## Evaluation

`scripts/eval_protocol.py` is the only source of reported metrics:

* deterministic: `model.eval()`, no TTA, no flip, no multi-scale, single forward;
* resolution 480x640 for both training and evaluation, matching the configs;
* one confusion matrix per image is stored, plus day/night and per-class tables;
* `miou_official` is the trainer's definition (a class with empty union scores 0
  and is averaged over all 9 classes); `miou_foreground` averages classes 1..8;
  both are reported so the choice cannot be gamed after the fact;
* person precision/recall and the guardrail IoU are reported because the
  motivating claim is about night pedestrians and because guardrail is where
  negative transfer was visible;
* when `--teacher-ckpt` is given, each pixel is cross-tabulated as
  teacher-right/student-right, teacher-only, student-only, both-wrong.

## Statistics

`scripts/compare_protocol.py` compares runs on identical images with an
image-level paired bootstrap (2000 resamples, seed 0). The bootstrap rebuilds
the global confusion matrix from resampled images and recomputes mIoU, so the CI
belongs to the reported statistic. Images, not pixels, are the resampling unit.

A difference is called significant only when the 95% interval excludes 0. With
392 images the interval is wide; a single-seed difference of ~1 mIoU is not
evidence on its own, which is why the key arms are repeated with paired seeds.

## Training

All arms share config, data order and budget:

* `CMNeXt-B2`, MiT-B2 ImageNet init, 500 epochs, batch 8, AdamW 6e-5, poly 0.9,
  warmup 10 epochs, AMP, 480x640, seed 3407 for the main comparison;
* `TRAIN.DETERMINISTIC: true` for every P1 arm (cudnn deterministic, benchmark
  off) so a rerun reproduces the same optimisation path;
* the teacher is the frozen `MFNet_CMNeXt-B2_it_teacher_s3407/best.pth`
  (sha256 `f1ea918197b26affa7b7f6631554c09242526127af8b9fac1680a092b97b0d8d`);
  a missing teacher now raises instead of silently training against random
  weights;
* the model-selection set is the same for all arms, so it cannot favour one.

## Arms

| arm | student | supervision |
|---|---|---|
| `p1_ce` | RGB | CE on every valid pixel (reference) |
| `p1_uniform` | RGB | CE + plain Hinton KD, tau 2, lambda 1 |
| `p1_correctness` | RGB | CE + KD only where the teacher's argmax equals the label |
| `p1_classbalance` | RGB | CE + KD weighted by capped inverse class frequency |
| `p1_both` | RGB | CE + KD weighted by both |
| `p1_cbce` | RGB | class-balanced CE only, no teacher (control for re-weighting) |

`LAMBDA_FEAT` is 0 in every P1 arm: the earlier `uniform` vs `recoverability`
comparison changed the back-off, the rare-class protection *and* the feature
term at once, so it could not attribute the difference. The feature term is
deliberately out of scope here.

Task-aware gates weight the per-pixel KL by `w = valid * teacher_correct *
class_weight` and divide by `sum(w)`, so the KD scale does not drift with the
selected fraction and a batch with no selected pixel contributes exactly zero
KD. The CE term is unchanged and still covers every valid pixel.

## Reproduction

```bash
export PYTHONPATH=.
python3 tools/train_kd.py --cfg configs/rgb_mfnet.yaml            --tag p1_ce      --seed 3407
python3 tools/train_kd.py --cfg configs/kd_mfnet_uniform.yaml     --tag p1_uniform --seed 3407
python3 tools/train_kd.py --cfg configs/kd_mfnet_correctness.yaml --tag p1_correctness --seed 3407
python3 tools/train_kd.py --cfg configs/kd_mfnet_classbalance.yaml --tag p1_classbalance --seed 3407
python3 tools/train_kd.py --cfg configs/kd_mfnet_both.yaml        --tag p1_both    --seed 3407

for tag in ce uniform correctness classbalance both; do
  name=$(ls -d output/*_s3407_${tag} | head -1)
  python3 scripts/eval_protocol.py --cfg configs/rgb_mfnet.yaml \
      --ckpt "$name/last.pth" --teacher-ckpt output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth \
      --out "results/protocol/P1_${tag}_last.json"
done

python3 scripts/compare_protocol.py --ref CE=results/protocol/P1_ce_last.json \
    --arm Uniform=results/protocol/P1_uniform_last.json \
    --out results/protocol/compare_p1.md
```

## Known limitations of this protocol

* One dataset, one backbone, one teacher. Nothing here supports a claim beyond
  MFNet / CMNeXt-B2.
* The 392-image eval set is also the selection set, so absolute `best` numbers are
  optimistic; `last` is the honest comparison.
* Image-level bootstrap captures evaluation noise, not training-seed noise; the
  latter needs the paired-seed repeats.
* `class_balance` weights are computed from the training split only, with
  `beta 0.5` and `cap 5.0`; the cap is a modelling choice, not tuned here.
