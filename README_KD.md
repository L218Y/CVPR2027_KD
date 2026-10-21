# Recoverability-Aware Privileged Distillation (RAPD)

Training-time multimodal, inference-time RGB-only semantic segmentation.

A frozen multimodal teacher (RGB + depth/thermal/event/LiDAR) supervises an
RGB-only student.  Instead of matching the teacher everywhere, a per-pixel
recoverability score `r` estimates how much of the teacher's fused evidence is
explainable from RGB.  Where `r` is low the teacher's over-confident target is
replaced by a conservative target that backs off towards the class prior, so the
student is never forced to reproduce evidence it cannot see.

## Layout

```
semseg/kd/losses.py      RecoverabilityKD, UniformKD
semseg/kd/probe.py       FeatureProbe, RecoverabilityHead
tools/train_kd.py        unified trainer (teacher / kd modes)
scripts/prepare_mfnet.py MFNet -> data/MFNet
scripts/collect_results.py, sync_results.sh, run_and_sync.sh
configs/teacher_mfnet.yaml, kd_mfnet.yaml
tests/test_kd.py
```

## Run

```bash
# 1) fused teacher
bash scripts/run_and_sync.sh configs/teacher_mfnet.yaml "mfnet teacher"

# 2) RGB-only student with RAPD
bash scripts/run_and_sync.sh configs/kd_mfnet.yaml "mfnet rapd"
```

`run_and_sync.sh` always publishes metrics/logs to `results/` and pushes them to
GitHub when the job ends, even if it crashes.

## Baseline

Forked from `jamycheung/DELIVER` (CMNeXt, Apache-2.0).  The only change to the
baseline is that `mmcv` is optional, so the code runs on modern PyTorch/CUDA.
