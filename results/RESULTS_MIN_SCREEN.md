# Minimal screen, partial (3 of 5 arms)

120 epochs, seed 3407, fixed final epoch, no TTA, no model selection. Student is
RGB-only at test time (verified: `student_modals=['img']`, 348-tensor checkpoints).

| arm | official | foreground | day | night | night person | guardrail |
|---|---|---|---|---|---|---|
| ce | 55.32 | 50.09 | — | 51.86 | 45.43 | 29.27 |
| cbce | 56.30 | 51.28 | 53.66 | 53.40 | 42.90 | 33.85 |
| cbce_kd | 56.73 | 51.71 | 55.57 | 53.32 | 45.38 | 25.73 |

Pending: `uniform_kd`, `correctness_kd`.

## The two pre-registered contrasts

- `cbce - ce` = **+1.19 foreground**: what class re-weighting alone buys.
- `cbce_kd - cbce` = **+0.43 foreground, 95% CI [-0.56, +1.58]**: what the
  frozen multimodal teacher adds on top of identical CE treatment.

The interval crosses zero. At this budget there is **no measurable aggregate
contribution from the teacher**; the gain that does exist comes from re-weighting.

The increment is class-specific rather than uniform: night person 42.90 -> 45.38
(+2.48) while guardrail 33.85 -> 25.73 (-8.12). The teacher helps the class it is
strong at and the cost is paid by another rare class.

Single seed, 120 epochs: this ranks candidates and bounds the effect. It does not
establish equivalence, and it must not be compared with the 500-epoch tables.
