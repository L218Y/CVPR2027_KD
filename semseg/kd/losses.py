"""Privileged-knowledge distillation losses.

The central idea is that a multimodal teacher's evidence is not uniformly
recoverable from RGB.  For every pixel we estimate a recoverability score
``r`` in [0, 1].  Where ``r`` is high the student is asked to match the
teacher distribution; where ``r`` is low the teacher's over-confident target
is replaced by a conservative target that keeps the teacher's ranking but
backs off towards the class prior.  Matching an unrecoverable target is what
produces confident hallucinations, so we never do it.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _masked_mean(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return x.mean()
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    m = m.unsqueeze(1).to(x.dtype)
    if m.shape[-2:] != x.shape[-2:]:
        m = F.interpolate(m, size=x.shape[-2:], mode="nearest")
    return (x * m).sum() / m.sum().clamp_min(1.0)


class UniformKD(nn.Module):
    """Plain Hinton distillation, used as the M2 baseline."""

    def __init__(self, tau: float = 2.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
        ignore_index: int = 255,
        **_: dict,
    ) -> dict:
        t = teacher_logits.detach()
        p_t = F.softmax(t / self.tau, dim=1)
        log_p_s = F.log_softmax(student_logits / self.tau, dim=1)
        kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1, keepdim=True)
        mask = None if labels is None else (labels != ignore_index)
        loss = _masked_mean(kl, mask) * (self.tau ** 2)
        return {
            "total": loss,
            "kd": loss.detach(),
            "feat": student_logits.new_zeros(()),
            "r_mean": student_logits.new_ones(()),
        }


class RecoverabilityKD(nn.Module):
    """Recoverability-aware privileged distillation.

    Parameters
    ----------
    tau:
        Distillation temperature.
    eps_max:
        Maximum probability mass moved from the teacher to the class prior in
        the conservative target.
    mode:
        ``recoverability`` (full method), ``uniform`` (ablation: r == 1),
        ``conservative_only`` (ablation: always use the conservative target),
        ``hard`` (ablation: hard-label distillation).
    """

    def __init__(
        self,
        tau: float = 2.0,
        eps_max: float = 0.5,
        mode: str = "recoverability",
        prior_momentum: float = 0.99,
        gate: str = "none",
        rare_protect: float = 0.0,
    ) -> None:
        super().__init__()
        assert mode in {"recoverability", "uniform", "conservative_only", "hard"}
        assert gate in {"none", "disagreement"}
        self.gate = gate
        self.rare_protect = rare_protect
        self.tau = tau
        self.eps_max = eps_max
        self.mode = mode
        self.prior_momentum = prior_momentum

    @torch.no_grad()
    def _update_prior(self, labels: Tensor, n_classes: int, ignore_index: int) -> Tensor:
        if not hasattr(self, "prior") or self.prior.numel() != n_classes:
            self.register_buffer("prior", torch.full((n_classes,), 1.0 / n_classes))
            self.prior = self.prior.to(labels.device)
        valid = labels[labels != ignore_index]
        if valid.numel() == 0:
            return self.prior
        hist = torch.bincount(valid, minlength=n_classes).float()
        hist = hist / hist.sum().clamp_min(1.0)
        hist = hist.to(self.prior.device)
        self.prior.mul_(self.prior_momentum).add_(hist, alpha=1 - self.prior_momentum)
        return self.prior / self.prior.sum().clamp_min(1e-8)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        r: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        ignore_index: int = 255,
        student_feat: Optional[Tensor] = None,
        teacher_feat: Optional[Tensor] = None,
        proj: Optional[nn.Module] = None,
        lambda_feat: float = 0.0,
    ) -> dict:
        t = teacher_logits.detach()
        n_classes = t.shape[1]
        if labels is not None:
            prior = self._update_prior(labels, n_classes, ignore_index)
        else:
            prior = torch.full((n_classes,), 1.0 / n_classes, device=t.device)

        p_t = F.softmax(t / self.tau, dim=1)
        log_p_s = F.log_softmax(student_logits / self.tau, dim=1)
        log_p_t = F.log_softmax(t / self.tau, dim=1)

        if self.mode == "hard":
            hard = t.argmax(dim=1)
            loss_map = F.cross_entropy(
                student_logits, hard, reduction="none", ignore_index=ignore_index
            ).unsqueeze(1)
            mask = None if labels is None else (labels != ignore_index)
            kd = _masked_mean(loss_map, mask)
            return {"total": kd, "kd": kd.detach(), "feat": student_logits.new_zeros(()), "r_mean": student_logits.new_zeros(())}

        if r is None or self.mode == "uniform":
            r_map = torch.ones_like(p_t[:, :1])
        else:
            r_map = r.detach()
            if r_map.dim() == 3:
                r_map = r_map.unsqueeze(1)
            if r_map.shape[-2:] != p_t.shape[-2:]:
                r_map = F.interpolate(r_map, size=p_t.shape[-2:], mode="bilinear", align_corners=False)
            r_map = r_map.clamp(0.0, 1.0)

        eps = self.eps_max * (1.0 - r_map)

        if self.gate == "disagreement":
            # only back off where the student already disagrees with the teacher:
            # if it predicts the teacher's class confidently, the privileged
            # evidence is evidently not needed at that pixel.
            with torch.no_grad():
                p_s_soft = F.softmax(student_logits / self.tau, dim=1)
                y_t = p_t.argmax(dim=1, keepdim=True)
                agree = p_s_soft.gather(1, y_t)
                eps = eps * (1.0 - agree).clamp(0.0, 1.0)

        if self.rare_protect > 0:
            # keep the teacher target intact on rare classes: the class prior
            # would otherwise flatten classes such as pedestrian or bump.
            with torch.no_grad():
                y_t = p_t.argmax(dim=1, keepdim=True)
                pv = prior.view(1, -1, 1, 1).expand(y_t.shape[0], -1, y_t.shape[2], y_t.shape[3])
                rel = pv.gather(1, y_t) / prior.max().clamp_min(1e-8)
                eps = eps * rel.pow(self.rare_protect)
        p_u = (1.0 - eps) * p_t + eps * prior.view(1, -1, 1, 1)
        log_p_u = torch.log(p_u.clamp_min(1e-8))

        kl_t = (p_t * (log_p_t - log_p_s)).sum(dim=1, keepdim=True)
        kl_u = (p_u * (log_p_u - log_p_s)).sum(dim=1, keepdim=True)

        if self.mode == "conservative_only":
            per_pixel = kl_u
        else:
            per_pixel = r_map * kl_t + (1.0 - r_map) * kl_u

        mask = None if labels is None else (labels != ignore_index)
        kd = _masked_mean(per_pixel, mask) * (self.tau ** 2)

        feat = student_logits.new_zeros(())
        if lambda_feat > 0 and student_feat is not None and teacher_feat is not None and proj is not None:
            s = proj(student_feat)
            cos = F.cosine_similarity(s, teacher_feat.detach(), dim=1).unsqueeze(1)
            feat_map = 1.0 - cos
            if r is not None and self.mode == "recoverability":
                w = F.interpolate(r_map, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
                feat_map = feat_map * w
            feat = _masked_mean(feat_map, mask)

        total = kd + lambda_feat * feat if lambda_feat > 0 else kd
        return {
            "total": total,
            "kd": kd.detach(),
            "feat": feat.detach() if torch.is_tensor(feat) else feat,
            "r_mean": r_map.mean().detach(),
        }
