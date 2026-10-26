# Cheap Heat: final protocol numbers

Gate protocol: trained on a 627/157 split of `train.txt` with hard-condition augmentation, checkpoint selected on the 157 held-out hard split, official 392-image test evaluated exactly once. The deployment rule only lets the thermal branch override pixels where either model predicts `person`.

## Sensor-budget Pareto

| thermal | pixels | official | fg | night person |
|---|---:|---:|---:|---:|
| none (CE120) | 0 | 56.38 | 51.28 | 45.24 |
| LR8 60x80 | 1/64 | 57.38 ± 0.04 | 52.40 ± 0.05 | 58.46 ± 0.14 |
| LR4 120x160 | 1/16 | 57.57 ± 0.08 | 52.61 ± 0.09 | 61.36 ± 0.40 |
| LR2 240x320 | 1/4 | 57.45 ± 0.00 | 52.47 ± 0.00 | 61.01 ± 0.00 |
| full-res teacher | 1 | 61.28 | — | 65.74 |

LR8/LR4 are 3 gate-training seeds with the same frozen experts; LR2 is one seed.

## Clean vs degraded

| method | clean official | clean night person | degraded official | degraded night person |
|---|---:|---:|---:|---:|
| RGB only | 56.38 | 45.24 | 47.95 | 32.74 |
| thermal only | 55.80 | 61.88 | 50.70 | 62.02 |
| full gate | 56.20 | 61.17 | 51.10 | 60.95 |
| person gate | 57.53 | 61.17 | 49.49 | 60.95 |

## Hard-augmentation ablation (LR4)

| gate training | clean official | clean night person | degraded official | degraded night person |
|---|---:|---:|---:|---:|
| hard aug | 57.53 | 61.17 | 51.10 | 60.95 |
| no hard aug | 57.54 | 61.08 | 50.23 | 61.11 |

## Per-class change (LR4 person gate vs RGB-only)

| class | RGB-only | gate | delta |
|---|---:|---:|---:|
| car | 83.91 | 83.93 | +0.02 |
| person | 55.79 | 65.24 | +9.45 |
| bike | 69.73 | 69.89 | +0.16 |
| curve | 30.12 | 30.13 | +0.01 |
| car_stop | 35.76 | 36.14 | +0.37 |
| guardrail | 33.59 | 33.58 | -0.00 |
| color_cone | 55.34 | 55.59 | +0.24 |
| bump | 45.97 | 45.97 | +0.00 |

## Caveats

- Low-resolution thermal is simulated by bicubic down/up sampling, not a real sensor.
- The gate uses two frozen B2 models; compute is not yet matched to a single model.
- MFNet only; no second dataset or real low-resolution capture yet.
- The gate is selected on a held-out split of `train.txt`; the 392-image test is touched once.

## Compute (RTX 4090, 480x640, batch 1)

| component | params (M) | GFLOPs | latency (ms) |
|---|---:|---:|---:|
| RGB expert | 25.8 | 46.2 | 11.01 |
| thermal expert | 58.7 | 74.3 | 29.78 |
| gate | 1.9 | 45.6 | 3.25 |
| joint pipeline | 86.4 | 166.0 | 42.57 |

The budget claim is about **sensor pixels**, not compute: the two-expert pipeline is
about 3.9x the RGB-only latency (42.6 vs 11.0 ms). A compute-matched single-model
variant and a native low-resolution thermal branch are future work.

