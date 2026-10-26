#!/usr/bin/env python3
"""Measure parameters, FLOPs and latency for the two-expert gate deployment."""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

from train_kd import build_model, load_state  # noqa: E402
from train_unet_gate import UNetGate, features  # noqa: E402
from fvcore.nn import FlopCountAnalysis  # noqa: E402

cfg_a = yaml.safe_load(open("configs/min/ce120.yaml"))
cfg_b = yaml.safe_load(open("configs/min/lr4_teacher.yaml"))
dev = torch.device("cuda")
a = build_model(cfg_a, 9, ["img"]).to(dev).eval()
b = build_model(cfg_b, 9, ["img", "thermal"]).to(dev).eval()
load_state("output/min/ce120/MFNet_CMNeXt-B2_i_rgb_s3407_p0ce120_s3407/last.pth", a)
load_state("output/min/lr4_teacher/MFNet_CMNeXt-B2_it_teacher_s3407_p0lr_s3407/last.pth", b)
g = UNetGate().to(dev).eval()
ck = torch.load("output/gate/unet_lr4_ce120/best_gate.pth", map_location="cpu", weights_only=False)
g.load_state_dict(ck["gate"])

rgb = torch.randn(1, 3, 480, 640, device=dev)
th = torch.randn(1, 3, 480, 640, device=dev)
x = torch.randn(1, 28, 480, 640, device=dev)

def flops(model, inp):
    return FlopCountAnalysis(model, inp).total() / 1e9

fa = flops(a, [rgb])
fb = flops(b, [rgb, th])
fg = flops(g, x)
pa = sum(p.numel() for p in a.parameters()) / 1e6
pb = sum(p.numel() for p in b.parameters()) / 1e6
pg = sum(p.numel() for p in g.parameters()) / 1e6

def latency(fn, iters=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters

with torch.no_grad():
    la = latency(lambda: a([rgb]))
    lb = latency(lambda: b([rgb, th]))
    # realistic joint path: run both experts, build gate input from their probs,
    # then run the gate.  This is what deployment does.
    def joint():
        pa = a([rgb]).softmax(1)
        pb = b([rgb, th]).softmax(1)
        feat = features(pa, pb, rgb, th)
        return g(feat)
    lj = latency(joint)
    # gate-only latency with precomputed 28-channel features
    lg = latency(lambda: g(x))

out = {
    "params_m": {"rgb_expert": pa, "thermal_expert": pb, "gate": pg, "total": pa + pb + pg},
    "flops_g": {"rgb_expert": fa, "thermal_expert": fb, "gate": fg, "total": fa + fb + fg},
    "latency_ms_batch1_480x640": {"rgb_expert": la, "thermal_expert": lb,
                                   "gate_only": lg, "joint_pipeline": lj},
    "device": torch.cuda.get_device_name(0),
}
Path("results/COMPUTE.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=2))
