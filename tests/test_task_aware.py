"""CPU checks for task-level selective distillation and its controls.

Run with: python3 tests/test_task_aware.py
"""

import torch
import torch.nn.functional as F

from semseg.kd import RecoverabilityKD, TaskAwareKD, class_balance_weights


def _logits(b=2, c=9, h=6, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, c, h, w, generator=g)


def _labels(b=2, h=6, w=8, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 9, (b, h, w), generator=g)


def _reference_kd(student, teacher, weights, tau):
    p_t = F.softmax(teacher / tau, dim=1)
    log_p_s = F.log_softmax(student / tau, dim=1)
    kl = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).sum(dim=1, keepdim=True)
    return (weights * kl).sum() / weights.sum().clamp_min(1e-12) * tau ** 2


def test_gate_keeps_only_correct_pixels():
    s, t, y = _logits(), _logits(seed=1), _labels()
    s.requires_grad_(True)
    kd = TaskAwareKD(tau=2.0, gate="correctness", class_weights=torch.ones(9))
    out = kd(s, t, labels=y)
    expected_w = (t.argmax(1) == y).float().unsqueeze(1)
    assert torch.allclose(out["total"], _reference_kd(s, t, expected_w, 2.0), atol=1e-6)
    g = torch.autograd.grad(out["total"], s)[0]
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_wrong_teacher_gives_zero_kd_and_finite_grad():
    s = _logits(seed=2)
    s.requires_grad_(True)
    y = torch.zeros(2, 6, 8, dtype=torch.long)
    t = torch.full((2, 9, 6, 8), -5.0)
    t[:, 7] = 5.0  # teacher always predicts class 7, label is 0
    out = TaskAwareKD(tau=2.0, gate="correctness")(s, t, labels=y)
    assert float(out["total"]) == 0.0
    assert float(out["select_frac"]) == 0.0
    g = torch.autograd.grad(out["total"], s)[0]
    assert torch.isfinite(g).all() and g.abs().sum() == 0


def test_gate_ignores_probe_score():
    s, t, y = _logits(seed=3), _logits(seed=4), _labels(seed=6)
    kd = TaskAwareKD(tau=2.0, gate="both", class_weights=torch.linspace(0.5, 4.0, 9))
    a = kd(s, t, r=torch.zeros(2, 6, 8), labels=y)["total"]
    b = kd(s, t, r=torch.ones(2, 6, 8), labels=y)["total"]
    assert torch.allclose(a, b)


def test_ignore_index_removed():
    s, t = _logits(seed=7), _logits(seed=8)
    y = _labels(seed=9).clone()
    y[:, :3, :] = 255
    out = TaskAwareKD(tau=2.0, gate="both", class_weights=torch.ones(9), ignore_index=255)(s, t, labels=y)
    assert 0.0 <= float(out["select_frac"]) <= 1.0
    full = TaskAwareKD(tau=2.0, gate="both", class_weights=torch.ones(9))(s, t, labels=y)
    assert torch.isfinite(full["total"])


def test_class_balance_weights_shape():
    counts = torch.tensor([1000.0, 500.0, 100.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w = class_balance_weights(counts, beta=0.5, cap=3.0)
    assert w[0] < w[1] < w[2]
    assert float(w[3]) == 3.0  # capped
    assert float(w[4]) == 0.0  # absent class has no weight
    assert torch.isfinite(w).all()


def test_balanced_ce_matches_weighted_reference():
    logits = _logits(seed=10)
    y = _labels(seed=11)
    w = torch.tensor([0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    got = F.cross_entropy(logits, y, weight=w, ignore_index=255)
    ref = F.cross_entropy(logits, y, weight=w, ignore_index=255, reduction="mean")
    assert torch.allclose(got, ref)


def test_conservative_only_forces_r_zero():
    s, t = _logits(seed=12), _logits(seed=13)
    y = _labels(seed=14)
    # separate instances: the running class prior is stateful, so calling the
    # same module twice would compare two different priors.
    kd0 = RecoverabilityKD(tau=2.0, eps_max=0.5, mode="conservative_only")
    kd1 = RecoverabilityKD(tau=2.0, eps_max=0.5, mode="conservative_only")
    with_r0 = kd0(s, t, r=torch.zeros(2, 6, 8), labels=y)["total"]
    with_r1 = kd1(s, t, r=torch.ones(2, 6, 8), labels=y)["total"]
    assert torch.allclose(with_r0, with_r1, atol=1e-6)
    # and it must differ from the plain recoverability objective at r = 1
    plain = RecoverabilityKD(tau=2.0, eps_max=0.5, mode="recoverability")
    assert not torch.allclose(with_r1, plain(s, t, r=torch.ones(2, 6, 8), labels=y)["total"], atol=1e-4)


def test_foreground_only_excludes_unlabeled():
    s, t = _logits(seed=20), _logits(seed=21)
    y = torch.zeros(2, 6, 8, dtype=torch.long)      # all unlabeled
    kd_fg = TaskAwareKD(tau=2.0, gate="correctness", foreground_only=True)
    out = kd_fg(s, t, labels=y)
    assert float(out["select_frac"]) == 0.0
    assert float(out["total"]) == 0.0
    y2 = y.clone()
    y2[:, :3, :] = 2                                # half person
    out2 = kd_fg(s, t, labels=y2)
    assert 0.0 < float(out2["select_frac"]) <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
