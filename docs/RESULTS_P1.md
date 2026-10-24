# P1 results — task-level selective distillation on MFNet

Status: four of six arms finished at seed 3407 (`CE`, `Uniform`, `Correctness`,
`ClassBalance`); `Both` and the `BalancedCE` control are still training, and
paired-seed repeats for `CE`/`ClassBalance` are queued. Numbers below are final
for the arms that completed and are never extrapolated.

All numbers come from `scripts/eval_protocol.py` on the fixed 392-image split and
are never mixed with the trainer's own validation. Generated tables are in
[`../results/REPORT_P1.md`](../results/REPORT_P1.md); the protocol is in
[`PROTOCOL_P1.md`](PROTOCOL_P1.md).

## Main comparison (fixed epoch-500 checkpoint, no model selection)

| arm | official mIoU | foreground mIoU | night mIoU | person | night person | guardrail | delta vs CE [95% CI] |
|---|---|---|---|---|---|---|---|
| CE | 58.29 | 53.40 | 54.66 | 57.92 | 47.91 | 39.12 | — |
| Uniform | 60.01 | 55.33 | 55.24 | 58.77 | 48.16 | 46.91 | +1.63 [+0.25, +3.00] |
| Correctness | 60.22 | 55.56 | 54.97 | 58.52 | 47.66 | 45.75 | +1.83 [+0.06, +3.69] |
| ClassBalance | 61.35 | 56.83 | 56.62 | 59.09 | 48.81 | 45.26 | +2.93 [+1.27, +4.72] |

Image-level paired bootstrap, 2000 resamples, same images for every arm. All three
KD arms beat the CE reference; ordering says class-frequency weighting contributes
more than the teacher-correctness gate.

Per-class IoU shows where this comes from — `ClassBalance` vs `CE`: bump 44.0 → 57.3,
car_stop 34.6 → 39.1, guardrail 39.1 → 45.3, person 57.9 → 59.1. Day person moves
66.2 → 67.9 while **night person moves only 47.91 → 48.81** against a teacher at
65.74.

## The transferable surplus is real but stays unconverted

`REPORT_P1.md` cross-tabulates teacher and student per pixel. Among the 506,795
night-person pixels:

| arm | both right | teacher only | student only | night person IoU |
|---|---|---|---|---|
| CE | 53.57% | **21.95%** | 3.54% | 47.91 |
| Uniform | 53.67% | **21.84%** | 3.33% | 48.16 |
| Correctness | 53.00% | **22.51%** | 3.21% | 47.66 |
| ClassBalance | 56.42% | **19.09%** | 4.07% | 48.81 |

Three things follow, and they matter more than the mIoU ranking:

1. **The opportunity exists.** The teacher is right on ~22% of night-person pixels
   where the student is wrong, and the reverse happens on ~3.5%. That is a ~6:1
   one-sided pool of pixels, so the earlier framing "the privileged evidence is not
   recoverable from RGB" is not what the data shows. There is a large deficit the
   teacher demonstrably solves.
2. **Plain KD does not touch it.** Uniform and correctness leave the teacher-only
   rate at 21.8–22.5%; night person IoU moves by ±0.25.
3. **Only the class-frequency weighting moves it**, and only by ~2.9 points of the
   22 available (night person 47.91 → 48.81). That is the single clearest signal in
   this batch: what converts teacher surplus is *how much KD gradient* that class
   receives, not *which pixels* are declared reliable.

The reverse case is equally clear: at night, guardrail has teacher-only 2.18–3.95%
versus student-only 12.9–14.9%. There the teacher is the worse teacher, and copying
it is harmful. The correctness gate was supposed to help exactly here, and on
guardrail it does raise IoU from 39.1 to ~45.8, but the same gate slightly *lowers*
night person (47.91 → 47.66), so it is trading one class for another rather than
selecting better targets overall.

## What the numbers do not support yet

* `BalancedCE` is still running. It is the control that decides whether +2.93 mIoU
  is privileged transfer or plain foreground re-weighting of the supervised term.
  Until it lands, no attribution is claimed.
* One seed per arm, and the CE reference landed 2 mIoU below the previously
  published RGB run (61.72 best / 59.99 last-10). The seed spread that document
  itself reports (59.3–61.7) covers exactly this range, which is why paired
  repeats of `CE` and `ClassBalance` are queued at seeds 3408/3409. Single-seed
  deltas of ~1 mIoU are not evidence, and the image-level CI only bounds
  evaluation noise, not training noise.
* Every arm stays 17 points short of the teacher on night person. No claim is made
  that the motivating gap is closed.
* Guardrail has few pixels (75k, and zero in day), so its IoU is unstable between
  checkpoints of the same arm (CE: 39.1 at epoch 500, 48.8 at its best epoch).

## What this implies for the next experiment

The bottleneck is not target selection. Given that the teacher's target is
available and good on those pixels, and that logit-level reweighting moves them by
less than one IoU point, the plausible blockers are now:

* **Capacity of the transferred quantity.** A 9-class posterior at 1/4 resolution,
  bilinearly upsampled, may not carry the small-person boundary information that
  makes the teacher right.
* **Representation, not supervision.** The student's RGB features at night may not
  contain the cue; more KD weight cannot create it. This is what the planned
  task-probe (real vs predicted fused features through the same head) would settle.
* **Optimisation budget for rare, hard pixels.** Night person is 0.88% of pixels;
  even 8× KD upweighting gives it little total gradient over 500 epochs.

That ordering suggests the next arm should test the *transferred quantity*
(multi-scale or feature-level targets with a task-calibrated adapter) rather than
another logit-gate variant.

## Reproduction

```bash
export PYTHONPATH=.
bash scripts/run_arm.sh 0 configs/kd_mfnet_classbalance.yaml p1_classbalance 3407
bash scripts/eval_p1.sh ce uniform correctness classbalance
python3 scripts/report_p1.py --out results/REPORT_P1.md
```
