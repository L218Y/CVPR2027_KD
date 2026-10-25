"""A checkpoint built with a different modality list must be rejected.

This is a regression test for a silent failure: a two-stream (RGB+thermal)
checkpoint loaded into a single-stream RGB model used to report
``loaded 348/348 tensors (missing 0)`` while discarding the other 548 tensors,
so evaluation ran on a mostly-untrained network and still produced numbers.
"""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from train_kd import build_model, load_state  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "configs/min/ce.yaml"


def _models():
    cfg = yaml.safe_load(CFG.read_text())
    return cfg, build_model(cfg, 9, ["img"]), build_model(cfg, 9, ["img", "thermal"])


def test_dual_stream_checkpoint_is_rejected(tmp_path):
    _, single, dual = _models()
    path = tmp_path / "dual.pth"
    torch.save({"model_state_dict": dual.state_dict(), "epoch": 1}, path)

    assert len(dual.state_dict()) > len(single.state_dict())
    try:
        load_state(str(path), single, strict=True)
    except RuntimeError as exc:
        assert "tensors the model cannot take" in str(exc)
    else:
        raise AssertionError("a two-stream checkpoint must not load into a one-stream model")


def test_matching_checkpoint_loads_cleanly(tmp_path):
    _, single, _ = _models()
    path = tmp_path / "single.pth"
    torch.save({"model_state_dict": single.state_dict(), "epoch": 1}, path)
    msg = load_state(str(path), single, strict=True)
    assert "missing 0" in msg and "unexpected 0" in msg


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_dual_stream_checkpoint_is_rejected(Path(d))
        test_matching_checkpoint_loads_cleanly(Path(d))
    print("load_state tests passed")
