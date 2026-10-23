# Machine state snapshot

**Snapshot time**: 2026-10-23 ~09:35 UTC
**Host**: single RTX 4090 24 GB, `/data/lab/CVPR2027_KD` (repo at `code/`)
**Reason**: lab compute is being shut down briefly; this file records what is on
disk, what was in flight, and how to resume.

## What is running at snapshot time

`bash scripts/queue_seeds.sh` → seed **3409**:

| run | state |
|---|---|
| 3409 teacher | finished (best 60.01, last-10 59.38) |
| 3409 rgb | finished |
| 3409 uniform | **in flight, ~ep 58/500 — will not finish** |
| 3409 rapd | not started |

`scripts/queue_diag.sh` was waiting for the GPU; it never started. Update: the
diagnostics and the v2 runs supersede it, re-launch it after restart if the old
checkpoints are still needed.

## What is on disk (not in git, needs a persistent volume)

| path | size | notes |
|---|---|---|
| `code/output/*/best.pth` | 99 MB each, 12 runs | checkpoints; sha256 in `results/CHECKSUMS.txt` |
| `code/output/*/probe.pth` | 520 KB each | **also copied** to `results/probes/` and pushed |
| `code/output/*/last.pth` | 99 MB each | crash-recovery copies |
| `data/mfnet_raw`, `data/MFNet` | ~5 GB | MFNet raw + converted dataset |
| `data/MFNet.zip`, `data/MF_Dataset.zip` | ~6 GB | original archives |
| `code/checkpoints/pretrained/segformer/mit_b2.pth` | 95 MB | backbone init weights |

If the volume does **not** survive the shutdown, everything above must be
recreated: `scripts/prepare_mfnet.py` for the data, the HuggingFace URL in
`README_KD.md` for the backbone, and the teacher runs must be retrained
(3.7 h each) before any distillation run.

## What is safely in git

- all code (`semseg/kd`, `tools/`, `scripts/`, `configs/`, `tests/`)
- `results/` — per-run `metrics.json`, `SUMMARY.md`, `STATUS.md`, `summary.csv`,
  `LIVE.md`, `VERIFIED.md`, `CHECKSUMS.txt`
- `results/probes/*.pth` — probe weights, enough to recompute r-maps without the
  teacher being re-run for the probe
- `logs_clean/` — per-run logs with progress bars stripped (164 KB total)
- `PROGRESS.md`, `STATE.md`, `README_KD.md`

## Verified numbers at snapshot (last-10 validation mean, from `results/VERIFIED.md`)

| seed | teacher | rgb | uniform | rapd ε=0.5 |
|---|---|---|---|---|
| 3407 | 59.29 | 58.29 | 58.44 | 59.24 |
| 3408 | 57.43 | 58.17 | 60.24 | 59.55 |
| 3409 | 59.38 | — | incomplete | — |

Key finding: RAPD ≈ uniform on the mean (+0.06, n=2), both ≈ +1.1 over RGB;
RAPD is far more stable across seeds (range 0.31 vs 1.80) and improves
`guardrail` by ~5.6 IoU on both seeds.

## How to resume after the restart

```bash
cd /data/lab/CVPR2027_KD/code
git pull --rebase origin main

# 1) verify the checkpoints survived
sha256sum -c results/CHECKSUMS.txt

# 2) re-install the environment if the container was rebuilt
python3 -m pip install timm fvcore opencv-python-headless tabulate tensorboard

# 3) diagnostics first (15 min, decides the v2 design)
bash scripts/queue_diag.sh

# 4) then the v2 queue (see PROGRESS.md for the plan)
```

The token for automatic pushes lives at `~/.config/cvpr_kd/pat` (mode 600). If
the home directory was wiped, re-upload it before running anything that pushes.

## Open decisions

1. Whether to finish 3409 (v1) or drop it — v1 is superseded by v2, the three
   teachers are already trained and reusable.
2. Compute budget for DeLiVER (4×4090 recommended, A100 80 GB allows the
   original 1024² protocol).
