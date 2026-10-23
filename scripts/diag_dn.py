#!/usr/bin/env python3
"""Day/night stratified evaluation that mirrors the trainer's own eval path.

The trainer builds a model from the config's modality list and calls it with a
list of tensors; diagnostics that deviate from that contract silently load a
mismatched checkpoint or feed the wrong container, so this script reuses the
exact same helpers as tools/train_kd.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import semseg.datasets as _datasets
from semseg.augmentations_mm import get_val_augmentation
from train_kd import build_model, load_state

CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop",
           "guardrail", "color_cone", "bump"]


def unpack(batch):
    """Return (list of modality tensors, label tensor) for any loader layout."""
    if isinstance(batch, dict):
        return None, batch["gt_seg_label_ignore_255"]
    sample, label = batch[0], batch[1]
    while isinstance(sample, (list, tuple)) and sample and not torch.is_tensor(sample[0]):
        sample = sample[0]
    return sample, label


def condition_of(path):
    """MFNet encodes night scenes with an N in the file name."""
    return "night" if "N" in os.path.basename(str(path)).upper()[:12] else "day"


def scores(hist):
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan) * 100.0
    return {
        "mIoU": float(np.nanmean(iou)),
        "per_class": {c: (float(v) if not np.isnan(v) else None)
                      for c, v in zip(CLASSES, iou)},
        "pixels": int(hist.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    kd = cfg.get("KD") or {}
    dcfg = cfg["DATASET"]
    n_classes = cfg.get("MODEL", {}).get("NUM_CLASSES", 9)

    if not kd.get("ENABLE", True):
        modals = list(kd.get("TEACHER_MODALS") or dcfg["MODALS"])
    else:
        modals = ["img"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    valset = getattr(_datasets, dcfg["NAME"])(
        dcfg["ROOT"], "val", get_val_augmentation(cfg["EVAL"]["IMAGE_SIZE"]), modals)
    loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(cfg, n_classes, modals).to(device)
    report = load_state(args.ckpt, model, strict=True)
    print(report)
    model.eval()

    hists = {"day": np.zeros((n_classes, n_classes), np.int64),
             "night": np.zeros((n_classes, n_classes), np.int64)}
    files = valset.files

    with torch.no_grad():
        for i, batch in enumerate(loader):
            sample, label = unpack(batch)
            if sample is None:
                continue
            inputs = [t.to(device, non_blocking=True) for t in sample][:len(modals)]
            logits = model(inputs)
            pred = logits.argmax(1).cpu().numpy()
            lab = label.squeeze(1).numpy()
            base = i * args.batch_size
            for k in range(lab.shape[0]):
                cond = condition_of(files[base + k])
                valid = lab[k] != 255
                hists[cond] += np.bincount(
                    lab[k][valid].astype(np.int64) * n_classes + pred[k][valid],
                    minlength=n_classes * n_classes).reshape(n_classes, n_classes)

    conditions = {c: scores(h) for c, h in hists.items()}
    n_day = sum(1 for f in files if condition_of(f) == "day")
    payload = {
        "cfg": args.cfg,
        "ckpt": args.ckpt,
        "modals": modals,
        "n_images": {"day": n_day, "night": len(files) - n_day},
        "load": report,
        "conditions": conditions,
        "overall": scores(hists["day"] + hists["night"]),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "day_night.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print("%-28s overall %.2f | day %.2f | night %.2f" % (
        os.path.basename(args.ckpt),
        payload["overall"]["mIoU"],
        conditions["day"]["mIoU"],
        conditions["night"]["mIoU"]))
    for c in ("person", "car", "guardrail", "bump"):
        d = conditions["day"]["per_class"][c]
        n = conditions["night"]["per_class"][c]
        print("  %-10s day %6s  night %6s" % (
            c, "n/a" if d is None else "%.1f" % d,
            "n/a" if n is None else "%.1f" % n))


if __name__ == "__main__":
    main()
