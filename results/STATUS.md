# Experiment status

- updated: 2026-10-22 09:40:40 UTC
- host: 7ec8c19e901c (Linux-6.8.0-138-generic-x86_64-with-glibc2.35)
- gpu: NVIDIA GeForce RTX 4090, 24564 MiB
- commit: fa78f45
- exit status: 0
- note: S3 conservative only

| experiment | dataset | kd | seed | best mIoU | epoch | time |
|---|---|---|---|---|---|---|
| MFNet_CMNeXt-B2_i_rgb_s3407 | MFNet | recoverability | 3407 | 58.95 | 400 | 01:50:49 |
| MFNet_CMNeXt-B2_it_conservative_only_s3407 | MFNet | conservative_only | 3407 | 61.01 | 275 | 03:02:30 |
| MFNet_CMNeXt-B2_it_recoverability_s3407 | MFNet | recoverability | 3407 | 60.49 | 350 | 05:07:21 |
| MFNet_CMNeXt-B2_it_teacher_s3407 | MFNet | recoverability | 3407 | 60.07 | 225 | 03:44:32 |
