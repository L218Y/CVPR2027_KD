"""Task-level selective distillation.

Feature cosine similarity answers "can the student regress the teacher's fused
feature", which is not the same question as "is the teacher's posterior a better
target than the label here".  This module weights the logit target by signals
that are defined on the task itself: whether the teacher is actually correct on
that pixel, and how rare the class is in the training split.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GATES = ("none", "correctness", "class_balance", "both")


def class_balance_weights(counts: Tensor, beta: float = 0.5, cap: float = 5.0) -> Tensor:
    """Per-class weights proportional to (uniform_freq / freq) ** beta, capped.

    ``cap`` bounds how much a tiny class such as guardrail can dominate; without
    it the weight of the rarest class exceeds two orders of magnitude.
    """
    counts = counts.float()
    present = counts > 0
    freq = counts / counts.sum().clamp_min(1.0)
    ref = 1.0 / present.sum().clamp_min(1)
    w = (ref / freq.clamp_min(1e-12)).pow(beta).clamp(max=cap)
    return torch.where(present, w, torch.zeros_like(w))


class TaskAwareKD(nn.Module):
    """Distil only where the teacher is task-reliable.

    ``w = valid * teacher_correct * class_weight`` and the loss is the weighted
    mean of the per-pixel KL, so the KD scale does not drift with the fraction of
    selected pixels.  A batch where nothing is selected contributes zero KD; the
    supervised CE term is untouched.
    """

    def __init__(
        self,
        tau: float = 2.0,
        gate: str = "both",
        class_weights: Optional[Tensor] = None,
        ignore_index: int = 255,
        foreground_only: bool = False,
    ) -> None:
        super().__init__()
        assert gate in GATES, f"unknown gate {gate}"
        self.tau = tau
        self.gate = gate
        self.ignore_index = ignore_index
        self.foreground_only = foreground_only
        if class_weights is None:
            class_weights = torch.ones(1)
        self.register_buffer("class_weights", class_weights.float())

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        r: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        ignore_index: int = 255,
        **_: dict,
    ) -> dict:
        t = teacher_logits.detach()
        p_t = F.softmax(t / self.tau, dim=1)
        log_p_t = F.log_softmax(t / self.tau, dim=1)
        log_p_s = F.log_softmax(student_logits / self.tau, dim=1)
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=1, keepdim=True)

        valid = labels != ignore_index
        if self.foreground_only:
            # the unlabeled class is ~92% of pixels and the teacher is almost
            # always right there, so a correctness gate that includes it selects
            # ~99% of pixels and stops being a gate at all.
            valid = valid & (labels != 0)
        w = valid.float().unsqueeze(1)
        if self.gate in ("correctness", "both"):
            correct = (t.argmax(dim=1) == labels) & valid
            w = w * correct.float().unsqueeze(1)
        if self.gate in ("class_balance", "both"):
            idx = labels.clamp(0, self.class_weights.numel() - 1)
            w = w * self.class_weights[idx].unsqueeze(1).to(w.dtype)

        den = w.sum()
        num = (w * kl).sum()
        kd = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num)) * (self.tau ** 2)
        return {
            "total": kd,
            "kd": kd.detach(),
            "feat": student_logits.new_zeros(()),
            "r_mean": student_logits.new_ones(()),
            "select_frac": (den / valid.float().sum().clamp_min(1.0)).detach(),
        }
