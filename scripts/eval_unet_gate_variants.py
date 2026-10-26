#!/usr/bin/env python3
"""Evaluate deployment variants of a trained U-Net reliability gate.

Variants differ only in where the gate is allowed to override the RGB model:
  full        : any pixel where the two models disagree
  person_or   : only pixels where either model predicts person
  person_b    : only pixels where the low-res thermal model predicts person
  person_prob : only pixels where p_b(person) > 0.5

    python3 scripts/eval_unet_gate_variants.py --gate output/gate/unet_lr4_ce120/best_gate.pth ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

import semseg.datasets as D  # noqa: E402
from semseg.augmentations_mm import get_val_augmentation  # noqa: E402
from train_kd import build_model, load_state  # noqa: E402
from train_unet_gate import UNetGate, features, hist_of, summarise, condition_of, hard_aug  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--cfg-a", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--cfg-b", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--b-modals", nargs="+", default=["img", "thermal"])
    ap.add_argument("--conf-thresh", type=float, default=None,
                    help="if set, use full gate on images whose RGB confidence is below this")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_a = yaml.safe_load(Path(args.cfg_a).read_text())
    cfg_b = yaml.safe_load(Path(args.cfg_b).read_text())
    transform = get_val_augmentation(cfg_b["EVAL"]["IMAGE_SIZE"])
    ds = D.MFNet(args.root, "val", transform, ["img", "thermal"])
    n_classes = ds.n_classes

    model_a = build_model(cfg_a, n_classes, ["img"]).to(device)
    model_b = build_model(cfg_b, n_classes, args.b_modals).to(device)
    print("A:", load_state(args.ckpt_a, model_a, strict=True))
    print("B:", load_state(args.ckpt_b, model_b, strict=True))
    model_a.eval(); model_b.eval()

    gate = UNetGate().to(device)
    ck = torch.load(args.gate, map_location="cpu", weights_only=False)
    gate.load_state_dict(ck["gate"])
    gate.eval()

    variants = ("A", "B", "full", "person_or", "person_b", "person_prob")
    if args.conf_thresh is not None:
        variants = variants + ("adaptive",)
    h = {k: np.zeros((n_classes, n_classes), np.int64) for k in variants}
    cond = {k: {c: np.zeros((n_classes, n_classes), np.int64) for c in ("day", "night")} for k in variants}
    cursor = 0
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False, drop_last=False)
    with torch.no_grad():
        for sample, label in loader:
            sample = [x.to(device, non_blocking=True) for x in sample]
            label = label.to(device, non_blocking=True)
            if args.degrade:
                sample[0], sample[1] = hard_aug(sample[0], sample[1], p_hard=1.0, train=False)
            pa = model_a([sample[0]]).softmax(1)
            pb = model_b(([sample[1]] if args.b_modals == ["thermal"] else [sample[0], sample[1]])).softmax(1)
            score = gate(features(pa, pb, sample[0], sample[1])).squeeze(1)
            pa_p, pb_p = pa.argmax(1), pb.argmax(1)
            dis = pa_p != pb_p
            masks = {
                "A": torch.zeros_like(dis),
                "B": torch.ones_like(dis),
                "full": dis & (score > 0),
                "person_or": dis & (score > 0) & ((pa_p == 2) | (pb_p == 2)),
                "person_b": dis & (score > 0) & (pb_p == 2),
                "person_prob": dis & (score > 0) & (pb[:, 2] > 0.5),
            }
            if args.conf_thresh is not None:
                conf = pa.max(1).values.mean(dim=(1, 2))
                use_full = conf < args.conf_thresh
                masks["adaptive"] = torch.where(use_full.view(-1, 1, 1), masks["full"], masks["person_or"])
            lab = label.squeeze(1).cpu().numpy()
            for b in range(lab.shape[0]):
                c = condition_of(ds.files[cursor + b])
                for k, m in masks.items():
                    pred = torch.where(m[b], pb_p[b], pa_p[b]).cpu().numpy()
                    hh = hist_of(pred, lab[b], n_classes)
                    h[k] += hh
                    cond[k][c] += hh
            cursor += lab.shape[0]

    out = {}
    for k in variants:
        out[k] = {"overall": summarise(h[k]),
                  "conditions": {c: summarise(cond[k][c]) for c in ("day", "night")}}
        print(f"{k:12s} official {out[k]['overall']['miou_official']:6.2f} "
              f"fg {out[k]['overall']['miou_foreground']:6.2f} "
              f"night_person {out[k]['conditions']['night']['per_class_iou']['person']}")
    Path(args.out).write_text(json.dumps({"gate": args.gate, "root": args.root, "variants": out}, indent=1))


if __name__ == "__main__":
    main()
