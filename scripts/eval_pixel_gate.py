#!/usr/bin/env python3
"""Pixel-wise reliability gate between a strong RGB-only model and a low-res
RGB-Thermal model.

The class gate showed the aggregate trade-off is real but a per-class probability
swap leaves a lot of the oracle on the table, especially for night pedestrians.
This learns a tiny pixel-wise deferral rule on the *training* split: when the two
models disagree, predict which one is correct from their confidence, disagreement,
person probability, and RGB/thermal intensity.  Test labels are never used.

    python3 scripts/eval_pixel_gate.py \
        --cfg-a configs/min/ce.yaml --ckpt-a output/.../last.pth \
        --cfg-b configs/min/lr4_teacher.yaml --ckpt-b output/.../last.pth \
        --root data/MFNet_lr4 --out results/protocol/PIXGATE_lr4_ce500.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import semseg.datasets as D  # noqa: E402
from semseg.augmentations_mm import get_val_augmentation  # noqa: E402
from train_kd import build_model, load_state  # noqa: E402

CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
IGNORE = 255
FG = list(range(1, 9))


def condition_of(stem):
    return "night" if "N" in Path(str(stem)).name.upper()[:12] else "day"


def hist_of(pred, label, n_classes):
    keep = label != IGNORE
    idx = label[keep].astype(np.int64) * n_classes + pred[keep].astype(np.int64)
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def iou_table(hist):
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan) * 100.0


def summarise(hist):
    iou = iou_table(hist)
    return {
        "miou_official": float(np.nan_to_num(iou, nan=0.0).mean()),
        "miou_foreground": float(np.nan_to_num(iou[FG], nan=0.0).mean()),
        "per_class_iou": {c: (None if np.isnan(v) else round(float(v), 3)) for c, v in zip(CLASSES, iou)},
    }


def pixel_features(pa, pb, img, therm):
    eps = 1e-6
    pa = pa.permute(0, 2, 3, 1).reshape(-1, pa.shape[1])
    pb = pb.permute(0, 2, 3, 1).reshape(-1, pb.shape[1])
    p_max_a = pa.max(1).values
    p_max_b = pb.max(1).values
    ent_a = -(pa * torch.log(pa.clamp_min(eps))).sum(1)
    ent_b = -(pb * torch.log(pb.clamp_min(eps))).sum(1)
    kl = (pa * (torch.log(pa.clamp_min(eps)) - torch.log(pb.clamp_min(eps)))).sum(1)
    person_a, person_b = pa[:, 2], pb[:, 2]
    rgb = img.mean(1).reshape(-1)
    th = therm.mean(1).reshape(-1)
    return torch.cat([
        pa, pb, p_max_a[:, None], p_max_b[:, None], ent_a[:, None], ent_b[:, None],
        kl[:, None], person_a[:, None], person_b[:, None], rgb[:, None], th[:, None],
    ], 1)


def gate_net(in_dim=29):
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-a", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--cfg-b", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pixels-per-batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_a = yaml.safe_load(Path(args.cfg_a).read_text())
    cfg_b = yaml.safe_load(Path(args.cfg_b).read_text())
    transform = get_val_augmentation(cfg_b["EVAL"]["IMAGE_SIZE"])
    ds_train = D.MFNet(args.root, "train", transform, ["img", "thermal"])
    ds_test = D.MFNet(args.root, "val", transform, ["img", "thermal"])
    n_classes = ds_train.n_classes

    model_a = build_model(cfg_a, n_classes, ["img"]).to(device)
    model_b = build_model(cfg_b, n_classes, ["img", "thermal"]).to(device)
    print("A:", load_state(args.ckpt_a, model_a, strict=True))
    print("B:", load_state(args.ckpt_b, model_b, strict=True))
    model_a.eval()
    model_b.eval()

    # ---- train the gate on disagreement pixels only -------------------------
    feats, targets = [], []
    loader = DataLoader(ds_train, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False, drop_last=False)
    with torch.no_grad():
        for sample, label in loader:
            sample = [x.to(device, non_blocking=True) for x in sample]
            label = label.to(device, non_blocking=True)
            pa = model_a([sample[0]]).softmax(1)
            pb = model_b([sample[0], sample[1]]).softmax(1)
            f = pixel_features(pa, pb, sample[0], sample[1])
            lab = label.squeeze(1).reshape(-1)
            ya = pa.permute(0, 2, 3, 1).reshape(-1, n_classes).argmax(1) == lab
            yb = pb.permute(0, 2, 3, 1).reshape(-1, n_classes).argmax(1) == lab
            valid = (lab != IGNORE) & (ya != yb)
            if valid.any():
                idx = valid.nonzero(as_tuple=False).squeeze(1)
                if idx.numel() > args.pixels_per_batch:
                    idx = idx[torch.randperm(idx.numel(), device=idx.device)[:args.pixels_per_batch]]
                feats.append(f[idx].detach())
                targets.append(yb[idx].float())
    X = torch.cat(feats, 0)
    y = torch.cat(targets, 0)
    print(f"gate training pixels: {X.shape[0]} (b-correct frac {y.mean().item():.3f})")
    gate = gate_net(X.shape[1]).to(device)
    pos_weight = ((1 - y).sum() / y.sum().clamp_min(1)).detach()
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for ep in range(args.epochs):
        perm = torch.randperm(X.shape[0], device=X.device)
        tot = 0.0
        for i in range(0, X.shape[0], 8192):
            j = perm[i:i + 8192]
            opt.zero_grad()
            loss = lossf(gate(X[j]).squeeze(1), y[j])
            loss.backward()
            opt.step()
            tot += loss.item() * j.numel()
        if ep % 3 == 0 or ep == args.epochs - 1:
            print(f"gate ep {ep+1}/{args.epochs} loss {tot / X.shape[0]:.4f}")
    gate.eval()

    # ---- evaluate A, B, learned gate, and per-pixel oracle on test ----------
    def eval_split(split):
        loader = DataLoader(split, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, shuffle=False, drop_last=False)
        h = {k: np.zeros((n_classes, n_classes), np.int64) for k in ("A", "B", "gate", "oracle")}
        cond = {k: {c: np.zeros((n_classes, n_classes), np.int64) for c in ("day", "night")}
                for k in h}
        cursor = 0
        with torch.no_grad():
            for sample, label in loader:
                sample = [x.to(device, non_blocking=True) for x in sample]
                label = label.to(device, non_blocking=True)
                pa = model_a([sample[0]]).softmax(1)
                pb = model_b([sample[0], sample[1]]).softmax(1)
                pred_a = pa.argmax(1)
                pred_b = pb.argmax(1)
                f = pixel_features(pa, pb, sample[0], sample[1])
                score = gate(f).squeeze(1)
                pred_gate = torch.where((pred_a.reshape(-1) != pred_b.reshape(-1)) & (score > 0),
                                        pred_b.reshape(-1), pred_a.reshape(-1))
                lab = label.squeeze(1).reshape(-1)
                ya = pred_a.reshape(-1) == lab
                yb = pred_b.reshape(-1) == lab
                pred_oracle = torch.where(yb & ~ya, pred_b.reshape(-1), pred_a.reshape(-1))
                preds = {"A": pred_a.reshape(-1), "B": pred_b.reshape(-1),
                         "gate": pred_gate, "oracle": pred_oracle}
                for b in range(label.shape[0]):
                    c = condition_of(split.files[cursor + b])
                    n = label.shape[-2] * label.shape[-1]
                    sl = slice(b * n, (b + 1) * n)
                    labb = lab[sl].cpu().numpy()
                    for k, p in preds.items():
                        hh = hist_of(p[sl].cpu().numpy(), labb, n_classes)
                        h[k] += hh
                        cond[k][c] += hh
                cursor += label.shape[0]
        return h, cond

    h, cond = eval_split(ds_test)
    out = {
        "meta": {"cfg_a": args.cfg_a, "ckpt_a": args.ckpt_a,
                 "cfg_b": args.cfg_b, "ckpt_b": args.ckpt_b, "root": args.root,
                 "gate_pixels": int(X.shape[0])},
        "test": {k: summarise(h[k]) for k in h},
        "test_by_condition": {k: {c: summarise(cond[k][c]) for c in ("day", "night")} for k in h},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for k in ("A", "B", "gate", "oracle"):
        print(f"{k:6s} official {out['test'][k]['miou_official']:6.3f}  "
              f"fg {out['test'][k]['miou_foreground']:6.3f}  "
              f"night person {out['test_by_condition'][k]['night']['per_class_iou']['person']}")


if __name__ == "__main__":
    main()
