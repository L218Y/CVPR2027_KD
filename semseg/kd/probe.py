"""Light-weight modules used to estimate privileged-evidence recoverability.

``FeatureProbe`` learns a linear map from the student's RGB feature to the
teacher's fused feature and reports the cosine agreement per pixel.  The
agreement is the recoverability score: if a linear probe on RGB features can
already explain the teacher's fused feature, that evidence is recoverable and
safe to distil; otherwise it is privileged and must not be forced onto the
student.

``RecoverabilityHead`` is a deployment-time head: it predicts the same score
from RGB only, so the final model can flag pixels where it is guessing.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FeatureProbe(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, teacher_dim, 1),
        )

    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.proj(student_feat)
        cos = F.cosine_similarity(z, teacher_feat, dim=1, eps=1e-6)
        r = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)
        mse = F.mse_loss(
            F.normalize(z, dim=1, eps=1e-6),
            F.normalize(teacher_feat, dim=1, eps=1e-6),
        )
        return r, mse

    @torch.no_grad()
    def score(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        z = self.proj(student_feat)
        cos = F.cosine_similarity(z, teacher_feat, dim=1, eps=1e-6)
        return ((cos + 1.0) * 0.5).clamp(0.0, 1.0)


class RecoverabilityHead(nn.Module):
    """Predicts the probe's recoverability score from RGB features alone."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return torch.sigmoid(self.head(feat))
