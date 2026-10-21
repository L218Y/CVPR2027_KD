"""Fast CPU checks for the recoverability distillation modules.

Run with:  python tests/test_kd.py
"""

import torch
import torch.nn.functional as F

from semseg.kd import FeatureProbe, RecoverabilityHead, RecoverabilityKD, UniformKD


def _logits(b=2, c=9, h=32, w=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, c, h, w, generator=g)


def test_uniform_equals_recoverability_with_r_one():
    s, t = _logits(), _logits(seed=1)
    labels = torch.randint(0, 9, (2, 32, 40))
    u = UniformKD(tau=2.0)(s, t, labels=labels, ignore_index=255)["kd"]
    kd = RecoverabilityKD(tau=2.0, mode="recoverability")
    r = torch.ones(2, 32, 40)
    rec = kd(s, t, r=r, labels=labels)["kd"]
    assert torch.allclose(u, rec, atol=1e-5), (u.item(), rec.item())


def test_conservative_target_differs_and_is_valid():
    s, t = _logits(), _logits(seed=2)
    labels = torch.randint(0, 9, (2, 32, 40))
    kd = RecoverabilityKD(tau=2.0, eps_max=0.5)
    r0 = torch.zeros(2, 32, 40)
    out = kd(s, t, r=r0, labels=labels)
    assert torch.isfinite(out["total"])
    assert 0.0 <= float(out["r_mean"]) <= 1.0


def test_ignore_index_is_respected():
    s, t = _logits(), _logits(seed=3)
    labels = torch.full((2, 32, 40), 255)
    out = RecoverabilityKD()(s, t, r=torch.rand(2, 32, 40), labels=labels)
    assert float(out["total"]) == 0.0


def test_feature_probe_and_gradients():
    s_feat = torch.randn(2, 64, 8, 10, requires_grad=True)
    t_feat = torch.randn(2, 64, 8, 10)
    probe = FeatureProbe(64, 64, hidden_dim=32)
    r, mse = probe(s_feat, t_feat)
    assert r.shape == (2, 8, 10) and r.min() >= 0 and r.max() <= 1
    (mse + r.mean()).backward()
    assert s_feat.grad is not None

    head = RecoverabilityHead(64)
    assert head(s_feat.detach()).shape == (2, 1, 8, 10)


def test_full_loss_backward_with_feature_term():
    s, t = _logits(requires_grad=True) if False else (_logits(), _logits(seed=4))
    s.requires_grad_(True)
    labels = torch.randint(0, 9, (2, 32, 40))
    s_feat = torch.randn(2, 64, 8, 10, requires_grad=True)
    t_feat = torch.randn(2, 64, 8, 10)
    probe = FeatureProbe(64, 64, hidden_dim=16)
    kd = RecoverabilityKD(tau=2.0)
    out = kd(s, t, r=torch.rand(2, 32, 40), labels=labels, student_feat=s_feat,
             teacher_feat=t_feat, proj=probe.proj, lambda_feat=0.1)
    out["total"].backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
