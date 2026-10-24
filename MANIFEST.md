# Artifact manifest

Everything that cannot be regenerated from source is published as a GitHub
release so the project survives the lab machine being powered down.

Release: <https://github.com/L218Y/CVPR2027_KD/releases/tag/artifacts-2026-10-24>
Size: 17 assets, 1.79 GB. Checksums: [`results/ARTIFACTS.sha256`](results/ARTIFACTS.sha256).

## Why a release and not the repository

GitHub rejects any single file above 100 MB, and the RGB-T teacher checkpoints
are 235 MB. Git LFS would work but its free storage quota (1 GB) is below the
1.5 GB of checkpoints, so release assets (2 GB per file) are the only option
that holds everything in one place. The repository itself stays at a few
megabytes and keeps every result that is small enough to diff.

## What is in the release

| asset | size | what it is |
|---|---|---|
| `MFNet_CMNeXt-B2_it_teacher_s3407_best.pth` | 235 MB | teacher, best val mIoU 61.28 @225 |
| `MFNet_CMNeXt-B2_it_teacher_s3407_last.pth` | 235 MB | teacher, epoch 500 |
| `MFNet_CMNeXt-B2_i_rgb_s3407_best.pth` | 103 MB | RGB baseline, best 61.72 @275 |
| `MFNet_CMNeXt-B2_i_rgb_s3407_last.pth` | 103 MB | RGB baseline, epoch 500 |
| `MFNet_CMNeXt-B2_it_uniform_s3407_best.pth` | 103 MB | uniform KD, best 59.95 @375 |
| `MFNet_CMNeXt-B2_it_uniform_s3407_last.pth` | 103 MB | uniform KD, epoch 500 |
| `MFNet_CMNeXt-B2_it_uniform_s3407_probe.pth` | 0.5 MB | frozen recoverability probe |
| `MFNet_CMNeXt-B2_it_recoverability_s3407_v2_{best,last,probe}.pth` | 207 MB | RAPD v2, best 60.10 @375 |
| `MFNet_CMNeXt-B2_it_recoverability_s3408_v2_{best,last,probe}.pth` | 207 MB | RAPD v2, best 60.68 @300 |
| `MFNet_CMNeXt-B2_it_recoverability_s3409_v2_{best,last,probe}.pth` | 207 MB | RAPD v2, seed 3409 |
| `MFNet_dataset.tar.gz` | 282 MB | MFNet train+val images, labels and split files |

"RAPD v2" is `KD.MODE=recoverability` with `GATE=none`, `RARE_PROTECT=1.0`,
`EPS_MAX=0.5`, `TAU=2`, `LAMBDA_KD=1`, `LAMBDA_FEAT=0.1`, 500 epochs, seed as named.

## What is not in the release, and how to get it

- **MFNet source archive** (984 MB, `MFNet.zip`): public, re-downloadable with
  `gdown 1NFdIigejYmCHFrdN2MSe1vxHHAs739SA`. The extracted copy in
  `MFNet_dataset.tar.gz` is what all runs actually used.
- **ImageNet-pretrained MiT-B2** (`checkpoints/pretrained/segformer/mit_b2.pth`,
  95 MB): from `InSAI-Lab/CMNeXt` on Hugging Face, downloaded by
  `scripts/prepare_mfnet.py`.
- **Per-run training logs**: tracked in the repository under `logs_clean/`
  (progress bars stripped) and `results/*/train.log`.
- **TensorBoard event files**: not archived; every scalar they contain is in
  `results/*/metrics.json`.

## Restore

```bash
git clone https://github.com/L218Y/CVPR2027_KD && cd CVPR2027_KD
bash scripts/restore_artifacts.sh          # downloads and checksums every asset
tar -C data -xzf data/MFNet_dataset.tar.gz # 784 train / 392 val images
```

`restore_artifacts.sh` verifies each file against `results/ARTIFACTS.sha256`
and refuses to continue on a mismatch.

## Where the numbers live

- `results/ANALYSIS.md` — inventory of all 15 completed runs, cross-machine
  variance, same-machine comparisons, thermal ablation.
- `results/P0_DAYNIGHT.md` — day/night and per-class breakdown.
- `results/VERIFIED.md`, `results/SUMMARY.md` — canonical tables rebuilt from
  `metrics.json` by `scripts/verify_stats.py` and `scripts/summarize.py`.
- `results/diag_uniform/r_by_class.json` — recoverability per class and lighting.
- `results/diag_v2_s340{7,8}`, `results/diag_teacher` — day/night per checkpoint.
