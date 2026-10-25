"""Training-time privileged feature injection.

The logit-level route is exhausted: on MFNet, any soft-target weighting we tried
(including the recoverability back-off) was indistinguishable from uniform KD, and
uniform KD was indistinguishable from no distillation at all. What the teacher
uniquely knows never crossed the logit bottleneck.

This module tests the other bottleneck. During training only, the frozen teacher's
deepest feature map is projected into the student's feature space and added to the
student's own features:

    s[-1] <- s[-1] + g * W(t[-1])

``g`` is a scalar initialised at zero, so training starts exactly as the RGB
baseline and only picks up the side channel if it pays for itself. ``W`` is a 1x1
convolution. At inference the whole module is absent, so the deployed network is
still RGB-only and identical in cost to the baseline.

The obvious failure mode is that the student leans on the side channel and
collapses when it is removed. ``drop_prob`` therefore zeroes the injection in a
fraction of iterations, so the network must remain useful without it; the caller
also reports ``inject_frac`` so a run that never injected is visible.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PrivilegedInjector(nn.Module):
    def __init__(self, teacher_dim: int, student_dim: int, gate_init: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Conv2d(teacher_dim, student_dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(student_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, teacher_feat: Tensor, student_feat: Tensor) -> Tensor:
        z = self.norm(self.proj(teacher_feat))
        if z.shape[-2:] != student_feat.shape[-2:]:
            z = F.interpolate(z, size=student_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.gate * z
