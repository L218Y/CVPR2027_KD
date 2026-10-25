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
- `cbce_kd − cbce`: how much the teacher adds on top of identical CE treatment (+0.43 foreground).

Single seed, 120 epochs: these numbers rank candidates, they do not establish significance, and they must not be compared against the 500-epoch tables.
