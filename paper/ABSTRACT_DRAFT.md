# Abstract draft (200 words)

Thermal cameras make nighttime pedestrian segmentation reliable, but their cost and
bandwidth keep RGB-Thermal (RGB-T) perception out of low-cost robots and vehicles.
We ask a budget question: how many thermal pixels are actually needed, and where
should they be used? On MFNet, we first show that the thermal teacher's nighttime
pedestrian advantage cannot be transported to an RGB-only student by logit
distillation, modality dropout, or thermal hallucination (night-person TCE ≈ −0.08):
the evidence is modality-borne. We then sweep the thermal sensor from 60×80 to
240×320 and find a clear knee at 120×160: 1/16 of the RGB pixel count recovers 84%
of the full-resolution teacher's nighttime pedestrian gain. Used globally, however,
low-resolution thermal loses on eight other classes. We therefore introduce a
person-aware reliability gate: a small U-Net, trained on a held-out split of the
training set with hard-condition augmentation, decides per pixel when to defer from
an RGB expert to the 120×160 thermal expert, and a structural constraint restricts
thermal overrides to pedestrians. With three seeds and a frozen 392-image test, the
gate raises overall mIoU from 56.38 to 57.48 ± 0.11 and nighttime pedestrian IoU from
45.24 to 62.26 ± 1.02; all other classes change by at most 0.08 IoU. Under synthetic
low-light and noise, RGB collapses to 47.79 mIoU while the gate holds 49.34 (unconstrained 51.27).
Thermal sensing should be budgeted and class-routed, not distilled or uniformly
deployed.

## Contributions (for the introduction)

1. A negative diagnosis: three transfer channels (logit distillation, modality
   dropout, hallucination) fail to move nighttime pedestrian evidence into RGB.
2. A sensor-budget curve for RGB-T segmentation, with a 120×160 knee and a
   per-class account of why global low-resolution fusion is not enough.
3. A person-aware reliability gate with an honest train-internal selection protocol
   and three-seed evidence: safety improves by +16.1 night-person IoU at +1.2 overall.
4. A degradation study showing the gate and the thermal expert remain useful when RGB
   is corrupted, while RGB-only collapses.

## Claims we will NOT make

- State-of-the-art overall mIoU on MFNet.
- Real-sensor validation (low-resolution thermal is simulated by bicubic round trip).
- Single-model compute parity (the gate uses two frozen B2 experts).
- Generalization beyond MFNet (no second dataset yet).
