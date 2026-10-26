#!/usr/bin/env python3
"""PST900 transfer experiment: low-resolution thermal reliability gate.

Trains the same U-Net gate used on MFNet on the PST900 fit split, selects on a
scene-disjoint gate-val split, and evaluates the official test split once.
Reports RGB-only, thermal-only and four deployment variants.

    python3 scripts/pst900_gate.py \
        --cfg-a configs/pst900/rgb.yaml --ckpt-a output/pst900/rgb/.../best.pth \
        --cfg-b configs/pst900/lr4.yaml --ckpt-b output/pst900/lr4/.../best.pth \
        --out results/protocol/PST900_GATE_lr4.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

import semseg.datasets as D  # noqa: E402
from semseg.augmentations_mm import get_val_augmentation  # noqa: E402
from train_kd import build_model, load_state  # noqa: E402
from train_unet_gate import UNetGate, features, hard_aug  # noqa: E402

IGNORE = 255


def hist_of(pred, label, n):
    keep = label != IGNORE
    idx = label[keep].astype(np.int64) * n + pred[keep].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def summarise(h, classes):
    inter = np.diag(h).astype(np.float64)
    union = h.sum(1) + h.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan) * 100
    return {
        "miou_official": float(np.nan_to_num(iou, nan=0.0).mean()),
        "miou_foreground": float(np.nan_to_num(iou[1:], nan=0.0).mean()),
        "per_class_iou": {c: (None if np.isnan(v) else round(float(v), 2))
                          for c, v in zip(classes, iou)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-a", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--cfg-b", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--root", default="data/PST900")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dir", default="output/gate/pst900_lr4")
    ap.add_argument("--gate", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-hard", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--person-id", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_a = yaml.safe_load(Path(args.cfg_a).read_text())
    cfg_b = yaml.safe_load(Path(args.cfg_b).read_text())
    transform = get_val_augmentation(cfg_b["EVAL"]["IMAGE_SIZE"])

    ds_tr = D.PST900(args.root, "train", transform, ["img", "thermal"])
    ds_va = D.PST900(args.root, "val", transform, ["img", "thermal"])
    ds_te = D.PST900(args.root, "official", transform, ["img", "thermal"])
    n = ds_tr.n_classes
    classes = ds_tr.CLASSES

    model_a = build_model(cfg_a, n, ["img"]).to(device)
    model_b = build_model(cfg_b, n, ["thermal"]).to(device)
    print("A:", load_state(args.ckpt_a, model_a, strict=True))
    print("B:", load_state(args.ckpt_b, model_b, strict=True))
    model_a.eval()
    model_b.eval()
    for p in model_a.parameters():
        p.requires_grad_(False)
    for p in model_b.parameters():
        p.requires_grad_(False)

    gate = UNetGate(in_ch=2 * n + 10).to(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def forward(sample, hard=False):
        rgb, therm = sample[0], sample[1]
        if hard:
            rgb, therm = hard_aug(rgb, therm, p_hard=1.0, train=False)
        pa = model_a([rgb]).softmax(1)
        pb = model_b([therm]).softmax(1)
        score = gate(features(pa, pb, rgb, therm)).squeeze(1)
        return pa, pb, score

    @torch.no_grad()
    def evaluate(ds, hard):
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, shuffle=False, drop_last=False)
        names = ("A", "B", "full", "person_or", "person_b", "person_prob")
        hs = {k: np.zeros((n, n), np.int64) for k in names}
        for sample, label in loader:
            sample = [x.to(device, non_blocking=True) for x in sample]
            label = label.to(device, non_blocking=True)
            pa, pb, score = forward(sample, hard)
            pa_p, pb_p = pa.argmax(1), pb.argmax(1)
            dis = pa_p != pb_p
            masks = {
                "A": torch.zeros_like(dis),
                "B": torch.ones_like(dis),
                "full": dis & (score > 0),
                "person_or": dis & (score > 0) & ((pa_p == args.person_id) | (pb_p == args.person_id)),
                "person_b": dis & (score > 0) & (pb_p == args.person_id),
                "person_prob": dis & (score > 0) & (pb[:, args.person_id] > 0.5),
            }
            lab = label.squeeze(1).cpu().numpy()
            for b in range(lab.shape[0]):
                for k, m in masks.items():
                    pred = torch.where(m[b], pb_p[b], pa_p[b]).cpu().numpy()
                    hs[k] += hist_of(pred, lab[b], n)
        return {k: summarise(hs[k], classes) for k in names}

    if args.gate:
        ck = torch.load(args.gate, map_location="cpu", weights_only=False)
        gate.load_state_dict(ck["gate"])
        history = ck.get("history", [])
    else:
        opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * 200)
        best, history = -1e9, []
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
                    pb = model_b([therm]).softmax(1)
                logit = gate(features(pa, pb, rgb, therm)).squeeze(1)
                lab = label.squeeze(1)
                with torch.no_grad():
                    ya = pa.argmax(1) == lab
                    yb = pb.argmax(1) == lab
                    valid = (lab != IGNORE) & (ya != yb)
                    tgt = yb.float()
                    w = 1.0 + 3.0 * (lab == args.person_id).float()
                loss_map = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
                loss = (loss_map * valid.float() * w).sum() / (valid.float() * w).sum().clamp_min(1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item()
                nb += 1
            va = evaluate(ds_va, hard=True)
            score = va["A"]["miou_official"] + 0.5 * va["person_or"]["per_class_iou"][classes[args.person_id]]
            history.append({"epoch": ep + 1, "loss": total / max(nb, 1),
                            "val_hard_A": va["A"]["miou_official"],
                            "val_hard_person_or": va["person_or"]["per_class_iou"][classes[args.person_id]],
                            "score": score})
            print(f"ep {ep+1}/{args.epochs} loss {total/max(nb,1):.4f} "
                  f"val_hard_A {va['A']['miou_official']:.2f} "
                  f"person_or {va['person_or']['per_class_iou'][classes[args.person_id]]}")
            if score > best:
                best = score
                torch.save({"gate": gate.state_dict(), "epoch": ep + 1, "history": history},
                           out_dir / "best_gate.pth")
        gate.load_state_dict(torch.load(out_dir / "best_gate.pth", map_location="cpu")["gate"])

    gate.eval()
    official = evaluate(ds_te, hard=False)
    payload = {"root": args.root, "gate": str(out_dir / "best_gate.pth"),
               "seed": args.seed, "person_class": classes[args.person_id],
               "history": history, "official": official}
    Path(args.out).write_text(json.dumps(payload, indent=1))
    for k, v in official.items():
        print(f"{k:12s} mIoU {v['miou_official']:6.2f} fg {v['miou_foreground']:6.2f} "
              f"person {v['per_class_iou'][classes[args.person_id]]}")


if __name__ == "__main__":
    main()
