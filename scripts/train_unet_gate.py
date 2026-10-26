#!/usr/bin/env python3
"""U-Net reliability gate with hard-condition augmentation.

The honest gate-val experiment showed that in-domain val is much easier than the
official test split and that a class gate selected there collapses to the thermal
model.  This trains a spatial gate on the train split with hard-condition
augmentation (dark/low-contrast/blur/noise), validates on a held-out hard split,
and touches the official test exactly once.

    python3 scripts/train_unet_gate.py \
        --cfg-a configs/min/ce120.yaml --ckpt-a ... \
        --cfg-b configs/min/lr4_teacher.yaml --ckpt-b ... \
        --train-root data/MFNet_lr4_gateval --test-root data/MFNet_lr4 \
        --out-dir output/gate/unet_lr4_ce120
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


def hard_aug(rgb, therm, p_hard=0.5, train=True):
    """rgb/therm are ImageNet-normalised [B,3,H,W] tensors on the device."""
    if train and torch.rand(1).item() > p_hard:
        return rgb, therm
    b = rgb.shape[0]
    mean = MEAN.to(rgb.device)
    std = STD.to(rgb.device)
    x = rgb * std + mean
    if train:
        scale = 0.35 + 0.65 * torch.rand(b, 1, 1, 1, device=rgb.device)
        noise = 0.06 * torch.randn_like(x)
    else:
        scale = torch.full((b, 1, 1, 1), 0.55, device=rgb.device)
        noise = 0.04 * torch.randn_like(x)
    x = (x * scale + noise).clamp(0, 1)
    if train and torch.rand(1).item() < 0.5:
        x = F.avg_pool2d(x, 5, stride=1, padding=2)
    rgb = (x - mean) / std
    if train:
        therm = therm + 0.03 * torch.randn_like(therm)
    return rgb, therm


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetGate(nn.Module):
    def __init__(self, in_ch=28, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(F.max_pool2d(x1, 2))
        x3 = self.d3(F.max_pool2d(x2, 2))
        x4 = self.d4(F.max_pool2d(x3, 2))
        y = self.c3(torch.cat([self.u3(x4), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.out(y)


def features(pa, pb, rgb, therm):
    eps = 1e-6
    ma = pa.max(1, keepdim=True).values
    mb = pb.max(1, keepdim=True).values
    ea = -(pa * torch.log(pa.clamp_min(eps))).sum(1, keepdim=True)
    eb = -(pb * torch.log(pb.clamp_min(eps))).sum(1, keepdim=True)
    return torch.cat([rgb, therm, pa, pb, ma, mb, ea, eb], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-a", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--cfg-b", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-hard", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_a = yaml.safe_load(Path(args.cfg_a).read_text())
    cfg_b = yaml.safe_load(Path(args.cfg_b).read_text())
    transform = get_val_augmentation(cfg_b["EVAL"]["IMAGE_SIZE"])
    ds_tr = D.MFNet(args.train_root, "train", transform, ["img", "thermal"])
    ds_va = D.MFNet(args.train_root, "val", transform, ["img", "thermal"])
    ds_te = D.MFNet(args.test_root, "val", transform, ["img", "thermal"])
    n_classes = ds_tr.n_classes

    model_a = build_model(cfg_a, n_classes, ["img"]).to(device)
    model_b = build_model(cfg_b, n_classes, ["img", "thermal"]).to(device)
    print("A:", load_state(args.ckpt_a, model_a, strict=True))
    print("B:", load_state(args.ckpt_b, model_b, strict=True))
    model_a.eval()
    model_b.eval()
    for p in model_a.parameters():
        p.requires_grad_(False)
    for p in model_b.parameters():
        p.requires_grad_(False)

    gate = UNetGate().to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * 100)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def predict(sample, hard=False):
        rgb, therm = sample[0], sample[1]
        if hard:
            rgb, therm = hard_aug(rgb, therm, p_hard=1.0, train=False)
        pa = model_a([rgb]).softmax(1)
        pb = model_b([rgb, therm]).softmax(1)
        score = gate(features(pa, pb, rgb, therm)).squeeze(1)
        pa_p = pa.argmax(1)
        pb_p = pb.argmax(1)
        pred = torch.where((pa_p != pb_p) & (score > 0), pb_p, pa_p)
        return pred, pa, pb

    @torch.no_grad()
    def evaluate(ds, hard):
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, shuffle=False, drop_last=False)
        h = np.zeros((n_classes, n_classes), np.int64)
        hn = np.zeros((n_classes, n_classes), np.int64)
        cursor = 0
        for sample, label in loader:
            sample = [x.to(device, non_blocking=True) for x in sample]
            label = label.to(device, non_blocking=True)
            pred, _, _ = predict(sample, hard=hard)
            lab = label.squeeze(1).cpu().numpy()
            pr = pred.cpu().numpy()
            for b in range(lab.shape[0]):
                hh = hist_of(pr[b], lab[b], n_classes)
                h += hh
                if condition_of(ds.files[cursor + b]) == "night":
                    hn += hh
            cursor += lab.shape[0]
        return summarise(h), summarise(hn)

    best = -1e9
    history = []
    for ep in range(args.epochs):
        gate.train()
        loader = DataLoader(ds_tr, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, shuffle=True, drop_last=True)
        total, nb = 0.0, 0
        for sample, label in loader:
            sample = [x.to(device, non_blocking=True) for x in sample]
            label = label.to(device, non_blocking=True)
            rgb, therm = hard_aug(sample[0], sample[1], p_hard=args.p_hard, train=True)
            with torch.no_grad():
                pa = model_a([rgb]).softmax(1)
                pb = model_b([rgb, therm]).softmax(1)
            feats = features(pa, pb, rgb, therm)
            logit = gate(feats).squeeze(1)
            lab = label.squeeze(1)
            with torch.no_grad():
                ya = pa.argmax(1) == lab
                yb = pb.argmax(1) == lab
                valid = (lab != IGNORE) & (ya != yb)
                tgt = yb.float()
                w = 1.0 + 3.0 * (lab == 2).float()  # person-weighted
            loss_map = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
            denom = (valid.float() * w).sum().clamp_min(1)
            loss = (loss_map * valid.float() * w).sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            nb += 1
        va_hard, va_night = evaluate(ds_va, hard=True)
        score = va_hard["miou_official"] + 0.5 * va_night["per_class_iou"]["person"]
        history.append({"epoch": ep + 1, "loss": total / max(nb, 1),
                        "val_hard_official": va_hard["miou_official"],
                        "val_hard_night_person": va_night["per_class_iou"]["person"], "score": score})
        print(f"ep {ep+1}/{args.epochs} loss {total/max(nb,1):.4f} "
              f"val_hard {va_hard['miou_official']:.2f} night_person {va_night['per_class_iou']['person']:.2f}")
        if score > best:
            best = score
            torch.save({"gate": gate.state_dict(), "epoch": ep + 1, "history": history},
                       out / "best_gate.pth")
    torch.save({"gate": gate.state_dict(), "epoch": args.epochs, "history": history},
               out / "last_gate.pth")

    ck = torch.load(out / "best_gate.pth", map_location="cpu", weights_only=False)
    gate.load_state_dict(ck["gate"])
    gate.eval()
    res = {}
    for name, ds, hard in (("val_clean", ds_va, False), ("val_hard", ds_va, True),
                           ("test_clean", ds_te, False)):
        ov, nt = evaluate(ds, hard=hard)
        res[name] = {"overall": ov, "night": nt}
        print(f"{name}: official {ov['miou_official']:.3f} fg {ov['miou_foreground']:.3f} "
              f"night_person {nt['per_class_iou']['person']}")
    (out / "result.json").write_text(json.dumps({
        "meta": {"cfg_a": args.cfg_a, "ckpt_a": args.ckpt_a, "cfg_b": args.cfg_b, "ckpt_b": args.ckpt_b,
                 "train_root": args.train_root, "test_root": args.test_root,
                 "best_epoch": ck["epoch"], "history": history},
        **res,
    }, indent=1))


if __name__ == "__main__":
    main()
