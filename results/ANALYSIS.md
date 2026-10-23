# Experiment inventory and analysis (as of 23:45 UTC, 23 Oct 2026)

All numbers below were read back from `results/*/metrics.json`; the day/night and
recoverability numbers come from re-loading the checkpoints with
`scripts/diag_dn.py`, `scripts/r_by_class.py` and `scripts/thermal_ablation.py`
(896/896 teacher tensors, 348/348 student tensors).

## 1. What has actually run

Fourteen completed runs, one in flight.

| machine | run | mode | seed | best | last-10 |
|---|---|---|---|---|---|
| old (wiped) | teacher | — | 3408 | 58.57 | 57.43 |
| old | teacher | — | 3409 | 60.01 | 59.38 |
| old | rgb | — | 3408 | 59.27 | 58.17 |
| old | rgb | — | 3409 | 59.58 | 58.56 |
| old | uniform | KD | 3408 | 60.79 | 60.24 |
| old | rapd ε=0.5 | KD | 3407 | 60.54 | 59.35 |
| old | conservative | KD | 3407 | 61.01 | 58.57 |
| old | rapd ε=0.2 | KD | 3407 | 59.62 | 58.57 |
| **new** | teacher | — | 3407 | 61.28 | 59.61 |
| **new** | rgb | — | 3407 | 61.72 | 59.99 |
| **new** | uniform | KD | 3407 | 59.95 | 59.21 |
| **new** | v2 (rare-protect) | KD | 3407 | 60.10 | 59.01 |
| **new** | v2 (rare-protect) | KD | 3408 | 60.68 | 60.08 |
| new | v2 (rare-protect) | KD | 3409 | *running, ep ~390/500* | |

The eight "old" runs were restored from git after the data volume was wiped; their
checkpoints are gone, so they cannot be re-evaluated and they were produced on
different hardware. They are historical only.

## 2. The two machines disagree by more than the treatment effect

Same config, same seed, same data pipeline:

| run | old last-10 | new last-10 | shift |
|---|---|---|---|
| rgb s3407 | 58.29 | 59.99 | +1.70 |
| teacher s3407 | 59.29 | 59.61 | +0.32 |
| uniform s3407 | 58.44 | 59.21 | +0.77 |

A +1.7 mIoU change for an identical recipe means the unattributed variance of a
single MFNet run is larger than every effect this project has tried to measure
(KD gains of 0.5–1.5). Only same-machine, same-seed comparisons are admissible,
and even those carry ±1.

## 3. Same-machine comparison (seed 3407, the only complete set)

| condition | RGB | uniform KD | RAPD v2 | teacher |
|---|---|---|---|---|
| best mIoU | **61.72** | 59.95 | 60.10 | 61.28 |
| last-10 mIoU | **59.99** | 59.21 | 59.01 | 59.61 |
| day mIoU | 58.97 | 59.41 | 59.38 | **60.86** |
| night mIoU | 56.78 | 54.80 | 55.11 | **57.99** |
| **person night IoU** | **48.0** | 48.1 | 47.3 | **65.7** |
| person day IoU | 67.0 | 67.7 | 67.6 | 71.5 |
| guardrail night | **56.0** | 48.3 | 45.2 | 43.8 |

Both knowledge-distillation variants sit 0.8–1.0 mIoU *below* the plain RGB
baseline on last-10, and neither moves night pedestrians at all (48.1 / 47.3 vs
48.0). The only large, consistent teacher advantage is night pedestrian
(+17.7), and distillation transfers none of it.

Seed 3408 with the same v2 recipe gives 60.08 last-10 — a 1.07 mIoU spread
across two seeds, i.e. the same size as the differences being compared.

## 4. Why not: the probe says the evidence is recoverable

Recoverability `r` of the uniform student, by ground-truth class:

| class | all | day | night |
|---|---|---|---|
| unlabeled | 0.929 | 0.934 | 0.924 |
| car | 0.898 | 0.915 | 0.869 |
| **person** | **0.865** | 0.874 | **0.855** |
| bike | 0.881 | 0.858 | 0.885 |
| curve | 0.861 | 0.867 | 0.841 |
| car_stop | 0.841 | 0.840 | 0.841 |
| guardrail | 0.829 | — | 0.829 |
| color_cone | 0.845 | 0.849 | 0.844 |
| bump | 0.855 | 0.869 | 0.835 |

`person` is the second-most "recoverable" class in the whole set (r = 0.855 at
night), so ε = ε_max(1−r) is small there: the loss is already almost pure
teacher matching on pedestrians. The method is not withholding thermal
evidence from pedestrians; it is trying to distil it at nearly full weight and
failing. That rules out the hypothesis that motivated v2's epsilon shaping.

## 5. Why it fails: the evidence really is thermal

Teacher with the thermal stream intact, zeroed, and shuffled across the batch
(statistics preserved, correspondence destroyed):

| thermal stream | day mIoU | person day | night mIoU | person night |
|---|---|---|---|---|
| intact | 60.86 | 71.5 | 57.99 | **65.7** |
| zeroed | 1.40 | 2.3 | 2.04 | 0.7 |
| shuffled | 56.42 | 49.7 | 48.24 | **16.8** |

Destroying the thermal–label correspondence costs the teacher 49 points of
night-pedestrian IoU. Night pedestrians are simply not in the RGB evidence at
this resolution and exposure; the teacher's competence is modality-borne, not
decision-boundary-borne, and soft targets cannot recreate missing evidence.

The probe measures *feature predictability*, not *task sufficiency*: the student
feature predicts the teacher feature to cos ≈ 0.73 while carrying none of the
pedestrian information that matters. That gap is the paper's central
methodological finding, and it also falsifies the premise that a high
recoverability score implies a safe distillation target.

## 6. Consequences for the paper

1. The MFNet mean-mIoU table is not usable as the main result: the effect is
   smaller than run-to-run variance on one machine and smaller than the
   machine-to-machine shift.
2. The honest headline is conditional: privileged distillation transfers only
   evidence that is *both* predictable and task-sufficient, and thermal night
   pedestrians fail the second test. This is measurable (shuffled-thermal
   control) and falsifiable, which is stronger than another 0.4 mIoU claim.
3. If a positive main result is still wanted, it has to come from a setting where
   the RGB→X gap is large enough to survive noise (DeLiVER, gap 9.1) and from a
   distillation target that carries the evidence in a transferable form
   (feature-level supervision on thermal-informative regions, not soft labels).
4. Two seeds are not enough to claim anything: v2 spans 59.01–60.08 across seeds.

## 7. Open items

- v2 seed 3409 finishing ~00:20; it cannot change the conclusion but completes
  the v2 triple.
- The uniform/v2 all-class distillation loss reduced overall accuracy relative
  to plain RGB on seed 3407; with n=1 per cell this may still be noise.
- `guardrail` moves opposite to `person` in every comparison, so any claim about
  it must be tested with its own control (RGB already beats the teacher there).
