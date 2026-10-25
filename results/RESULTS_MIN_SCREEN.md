# Minimal screen (120 epochs, seed 3407)

Fixed final epoch, no TTA, no model selection. Student is RGB-only at test time.

| arm | epoch | official | foreground | day | night | night person | last-10 (train) | Δfg vs ce | Δnight-person vs ce |
|---|---|---|---|---|---|---|---|---|---|
| ce | 120 | 55.32 | 50.09 | 54.83 | 51.86 | 45.43 | 54.30 | +0.00 | +0.00 |
| cbce | 120 | 56.30 | 51.28 | 53.66 | 53.40 | 42.90 | 54.61 | +1.20 | -2.53 |
| uniform_kd | 120 | 55.55 | 50.34 | 56.16 | 51.33 | 45.66 | 54.64 | +0.25 | +0.24 |
| cbce_kd | 120 | 56.73 | 51.71 | 55.57 | 53.32 | 45.38 | 56.12 | +1.62 | -0.04 |
| correctness_kd | 120 | 57.32 | 52.41 | 55.27 | 53.83 | 41.08 | 55.83 | +2.32 | -4.35 |

The two contrasts that matter are pre-registered in `docs/MINIMAL_EXPERIMENTS.md` §6:

- `cbce − ce`: how much of the gain is plain class re-weighting.
  Observed **+1.19 foreground** (51.28 - 50.09).
- `cbce_kd − cbce`: how much the frozen teacher adds on top of identical CE
  treatment. Observed **+0.43 foreground**; image-level paired bootstrap
  **+0.50 [-0.56, +1.58]**, i.e. the interval crosses zero.

`results/protocol/MIN_decisive_3407_e120.md` holds the decisive interval. Note the
two numbers differ on purpose: the table reports the observed difference, the
bootstrap reports a resampled mean; earlier reports in this repo mixed the two
without saying which was which.

Where the gain actually sits (per-class IoU, foreground):

| arm | car | person | bike | car_stop | guardrail | bump |
|---|---|---|---|---|---|---|
| ce | 83.64 | 55.73 | 70.04 | 33.00 | 29.27 | 41.92 |
| cbce | 81.51 | 50.98 | 65.08 | 40.45 | 33.85 | 55.48 |
| uniform_kd | 83.38 | 56.18 | 70.00 | 34.24 | 29.67 | 43.31 |
| cbce_kd | 82.54 | 54.54 | 67.57 | 39.03 | 25.73 | 55.42 |
| correctness_kd | 81.05 | 51.29 | 67.54 | 39.61 | 31.40 | 57.29 |

Every arm that wins does so on `bump` and `car_stop` (re-weighting), and pays for
it on `car`, `person` and `bike`. Adding the teacher to the re-weighted arm moves
one class: person 50.98 -> 54.54, at the cost of guardrail 33.85 -> 25.73.

Single seed, 120 epochs: these numbers rank candidates, they do not establish
significance, and they must not be compared against the 500-epoch tables. The
image-level interval covers evaluation sampling only; the seed-to-seed spread
measured earlier in this repo was 1.5-2.0 mIoU, which is larger than every
difference in the table except the re-weighting effect.
