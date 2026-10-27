# Cheap Heat — paper outline

Status: MFNet results are final under the frozen protocol. Low-resolution thermal is
simulated (bicubic down/up). No second dataset or real sensor yet. All numbers below
come from `results/CHEAP_HEAT.md` / `results/protocol/*.json`.

## Title candidates

1. **Cheap Heat: A 1/16-Resolution Thermal Sensor with a Person-Aware Reliability Gate for Nighttime RGB-T Segmentation**
2. **Not Every Pixel Needs Heat: Budgeted Thermal Sensing for Safety-Critical Segmentation**
3. **The Person Is the Signal: Class-Aware Thermal Gating at 1/16 Resolution**

Preferred: #1 for a systems/safety framing; #2 if we add the budget-allocation method.

## One-sentence claim

A 120×160 thermal sensor (1/16 of the RGB pixel count) plus a learned, person-aware
reliability gate matches a 120-epoch RGB-only model on overall mIoU while raising
nighttime pedestrian IoU from 45.2 to 62.3 — 84% of the gain of a full-resolution
thermal teacher — and the same gate degrades gracefully when RGB itself is corrupted.

## Contributions

1. **Diagnosis (negative result).** RGB-only distillation, modality dropout, and
   thermal hallucination all fail to transfer the teacher's nighttime pedestrian
   evidence (night-person TCE ≈ −0.08). The advantage is modality-borne, not a
   transferable representation.
2. **Sensor-budget study.** Sweep thermal resolution 60×80 / 120×160 / 240×320.
   Night-person recovery is 58% / 84% / 95%; 120×160 is the knee, so 1/16 of the
   pixels captures most of the safety benefit.
3. **Person-aware reliability gate.** A small U-Net decides, per pixel, whether to
   defer from the RGB expert to the low-resolution thermal expert. A structural
   person constraint keeps thermal from corrupting the eight non-person classes.
   Honest protocol: gate trained on a 627/157 split of `train.txt` with hard-condition
   augmentation, selected on the 157 held-out hard split, test touched once.
4. **Robustness.** Under synthetic low-light/noise, RGB collapses (56.4 → 48.0)
   while the person-constrained gate holds 49.3, the unconstrained gate 51.3, and the thermal expert 62.4 night-person IoU.

## Main table (frozen protocol, 392-image test)

| thermal | pixels | official | fg | night person |
|---|---:|---:|---:|---:|
| none (CE120) | 0 | 56.38 | 51.28 | 45.24 |
| LR8 60×80 | 1/64 | 57.19 ± 0.07 | 52.19 ± 0.08 | 58.63 ± 0.23 |
| **LR4 120×160** | **1/16** | **57.48 ± 0.11** | **52.50 ± 0.12** | **62.26 ± 1.02** |
| LR2 240×320 | 1/4 | 57.54 ± 0.06 | 52.57 ± 0.07 | 63.29 ± 0.64 |
| full-res teacher | 1 | 61.28 | — | 65.74 |

LR8/LR4/LR2: three gate-training seeds with frozen experts.
Per-class: the entire gain is person (+8.91 person; all other classes |Δ| ≤ 0.08).

## Clean vs degraded

| method | clean official | clean night person | degraded official | degraded night person |
|---|---:|---:|---:|---:|
| RGB only | 56.38 | 45.24 | 47.79 | 33.78 |
| thermal only | 41.67 | 62.41 | 41.67 | 62.41 |
| full gate | **57.58** | 61.42 | **51.27** | 60.52 |
| person gate | 57.39 | 61.42 | 49.34 | 60.52 |

This clean/degraded trade-off is the honest limitation of a fixed person constraint;
an image-quality-conditioned gate is future work.

## Hard-augmentation ablation

| gate training | clean official | clean night person | degraded official | degraded night person |
|---|---:|---:|---:|---:|
| hard aug | 57.53 | 61.17 | **51.10** | 60.95 |
| no hard aug | 57.54 | 61.08 | 50.23 | 61.11 |

Hard augmentation is neutral on clean data and worth +0.87 overall under degradation.

## Figures and tables in the current draft

| # | content | file |
|---|---|---|
| Fig.1 | Sensor-budget Pareto (night person vs thermal pixels) | `paper/figures/pareto_night_person.pdf` |
| Fig.2 | Per-class delta (only person moves) | `paper/figures/per_class_delta.pdf` |
| Tab.1 | Thermal budget and gate sweep | `paper/sections/4_experiments.tex` |
| Tab.2 | Gate variants (seed 3407) | `paper/sections/4_experiments.tex` |
| Tab.3 | Clean vs corrupted RGB | `paper/sections/4_experiments.tex` |
| Tab.4 | Single-model compute comparison | `paper/sections/4_experiments.tex` |

## Section skeleton

1. **Introduction.** RGB-T segmentation is accurate but thermal sensors are expensive;
   the default assumption is "distill thermal into RGB" or "use full thermal".
   We show neither is necessary: thermal is only needed for pedestrians at night,
   and 1/16 of the pixels is enough.
2. **Related work.** RGB-T segmentation (MFNet, CMX/CMNeXt), privileged information
   and cross-modal distillation, modality hallucination, sensor-budget / adaptive
   sensing, mixture-of-experts and learning-to-defer.
3. **Diagnosis.** P0 results: distillation, modality dropout, hallucination all fail;
   night-person TCE ≈ −0.08. Establish that the evidence is modality-borne.
4. **Sensor budget.** Low-resolution thermal sweep; recovery curve; knee at 120×160;
   class trade-off that motivates a gate rather than global fusion.
5. **Method.** Two frozen experts; U-Net gate on RGB+thermal+probabilities/entropy;
   hard-condition augmentation; train-internal 627/157 split; person-constrained
   deferral. Explain why the constraint is structural (safety class), not test-tuned.
6. **Experiments.** Main table, 3-seed variance, per-class delta, clean vs degraded,
   hard-aug ablation, gate-variant ablation, class-gate/MLP-gate baselines.
7. **Limitations.** Simulated low-resolution; MFNet only; two-model compute;
   fixed person constraint is suboptimal under degradation; no real sensor capture.
8. **Conclusion.** Thermal sensing should be budgeted and class-routed, not distilled
   or uniformly deployed.

## Reviewer attacks and answers

| attack | answer | remaining gap |
|---|---|---|
| "Low resolution is synthetic." | Bicubic round trip is a conservative proxy; degradation experiments show the trend. | Real 60×80 / 120×160 capture would close this. |
| "Two models double compute." | Reported honestly: 86.4M params, 166 GFLOPs, 42.6 ms vs 11.0 ms RGB-only on a 4090. The claim is sensor-pixel budget, not compute parity; a native low-res thermal branch and distillation are future work. | Compute-matched single-model variant. |
| "Person constraint is test-tuned." | It is a structural safety prior; the gate itself is trained/selected without test labels. Ablate full gate (worse clean, better degraded). | A validation-selected constraint would be cleaner. |
| "Only MFNet." | Acknowledge; MFNet is the standard RGB-T safety benchmark. | Add DeLiVER or a real capture. |
| "Gain is only +1.2 overall." | The claim is safety-at-parity: +16 night person with no overall loss, at 1/16 pixels. Per-class table shows all gain is person. | A second dataset with more night pedestrians. |
| "Why not just always use thermal?" | Always-thermal loses 0.6 overall and 4 points on bump/car_stop; the gate preserves RGB's advantage on eight classes. | — |

## Venue strategy

- **CVPR main:** possible if a second dataset (DeLiVER) or a real low-resolution capture
  reproduces the pattern; current evidence is single-dataset and simulated.
- **CVPR workshop / TMLR:** solid now: diagnosis + sensor budget + honest gate +
  robustness, with 3 seeds and a frozen protocol.
- **Do not** claim state-of-the-art overall mIoU; the contribution is safety-at-parity
  and sensor budget.

## Writing discipline

- Every number from `results/protocol/*.json`; regenerate with `scripts/make_figures.py`.
- Never mix 120-ep and 500-ep baselines; the main table is 120 ep only.
- Always report the 392-image test, 3 seeds, mean ± sd; no best-epoch selection on test.
- State the gate protocol (627/157, hard aug, test once) in the abstract or method.
