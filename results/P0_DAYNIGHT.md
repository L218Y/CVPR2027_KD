# P0 — day/night stratification (MFNet val, 392 images: 194 day / 198 night)

Checkpoints: `best.pth` of `MFNet_CMNeXt-B2_it_teacher_s3407` and
`MFNet_CMNeXt-B2_i_rgb_s3407`. Both reloaded with 896/896 and 348/348 tensors
respectively, so the numbers below are the checkpoints' own numbers, not the
logged ones copied over.

## Aggregate

| metric | teacher (RGB-T) | RGB student | Δ (teacher − student) |
|---|---|---|---|
| best-epoch mIoU | 61.28 @225 | **61.72** @275 | −0.44 |
| last-10 mean mIoU | 59.61 | **59.99** | −0.38 |
| final epoch mIoU | 59.55 | 59.82 | −0.27 |
| day mIoU | **60.86** | 58.97 | +1.89 |
| night mIoU | **57.99** | 56.78 | +1.21 |

The teacher does **not** dominate the RGB student in aggregate on this machine.
It is better in both lighting conditions and still loses overall, because the
mean over nine classes is dominated by rare classes the teacher regresses on
(guardrail −15.5, cone −2.4, car_stop −3.7), not by the condition averages.

Both "best" values are single-epoch peaks on a 392-image split (teacher
neighbours 60.76 / 60.65; student neighbours 56.72 / 59.45), so best-epoch
numbers carry roughly ±1 mIoU of selection noise; last-10 is the robust
comparator.

## Per class (IoU, pooled over the whole split)

| class | teacher | RGB | Δ |
|---|---|---|---|
| unlabeled | 97.57 | 97.47 | +0.10 |
| car | **85.28** | 84.65 | +0.63 |
| person | **68.92** | 58.33 | **+10.59** |
| bike | 70.48 | 71.14 | −0.66 |
| curve | **38.39** | 32.73 | +5.66 |
| car_stop | 37.13 | **40.87** | −3.74 |
| guardrail | 39.03 | **54.58** | −15.55 |
| cone | 57.54 | **59.89** | −2.35 |
| bump | **57.21** | 55.83 | +1.38 |

## Day versus night

| class | teacher day | teacher night | RGB day | RGB night |
|---|---|---|---|---|
| person | 71.49 | **65.74** | 66.95 | **47.99** |
| car | 90.35 | 77.56 | 89.84 | 76.85 |
| guardrail | 0.00 | 43.77 | 0.00 | 55.96 |
| bump | 65.85 | 43.52 | 66.12 | 39.25 |
| mIoU | 60.86 | 57.99 | 58.97 | 56.78 |

`person` is the only class where the thermal modality buys something large and
robust: +4.5 in daylight, **+17.7 at night**. Guardrail runs the other way —
thermal is *worse* than RGB at night by 12.2 — so any previous claim that RAPD
helps guardrail cannot be attributed to privileged evidence.

## Why pooled mIoU exceeds both condition means

Per-class IoU is recomputed on the union of pixels, so it is not a weighted
average of the two condition values. `guardrail` exists almost only at night
(day IoU 0.00 over a handful of pixels, night 43.8), which lifts the pooled
mean by +4.3/9 ≈ 0.48 for the teacher, while `bike` gains +14.3 for the same
reason. Naively averaging the day and night columns therefore understates the
pooled value; the two are different estimators and must not be combined.
