#!/usr/bin/env python3
"""Train-selected class gate between an RGB-only model and a low-res RGB-T model.

The gate is deliberately post-hoc and label-free at test time: for each class it
picks the model with the higher *train* per-class IoU, then on test it takes that
model's probability for the class and argmaxes over classes.  This measures
whether the class trade-off seen on test is stable enough to be learned without
test labels, before committing to an end-to-end gated architecture.

    python3 scripts/eval_class_gate.py \
        --cfg-a configs/min/ce120.yaml --ckpt-a output/.../last.pth \
        --cfg-b configs/min/lr8_teacher.yaml --ckpt-b output/.../last.pth \
        --root data/MFNet_lr8 --out results/protocol/GATE_class_lr8_ce120.json
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
        "present": int((~np.isnan(iou)).sum()),
    }


def forward_probs(model, sample, idx):
    return model([sample[i] for i in idx]).softmax(1)


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
    ap.add_argument("--save-per-image", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_a = yaml.safe_load(Path(args.cfg_a).read_text())
    cfg_b = yaml.safe_load(Path(args.cfg_b).read_text())
    modals_a, modals_b = ["img"], ["img", "thermal"]

    transform = get_val_augmentation(cfg_b["EVAL"]["IMAGE_SIZE"])
    ds_train = D.MFNet(args.root, "train", transform, ["img", "thermal"])
    ds_test = D.MFNet(args.root, "val", transform, ["img", "thermal"])
    n_classes = ds_train.n_classes
    assert n_classes == len(CLASSES)

    model_a = build_model(cfg_a, n_classes, modals_a).to(device)
    model_b = build_model(cfg_b, n_classes, modals_b).to(device)
    print("A:", load_state(args.ckpt_a, model_a, strict=True))
    print("B:", load_state(args.ckpt_b, model_b, strict=True))
    model_a.eval()
    model_b.eval()

    def run_split(ds, choices=None, collect_per_image=False):
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, shuffle=False, drop_last=False)
        h_a = np.zeros((n_classes, n_classes), np.int64)
        h_b = np.zeros((n_classes, n_classes), np.int64)
        h_g = np.zeros((n_classes, n_classes), np.int64)
        cond_g = {c: np.zeros((n_classes, n_classes), np.int64) for c in ("day", "night")}
        per_image = []
        cursor = 0
        with torch.no_grad():
            for sample, label in loader:
                sample = [x.to(device, non_blocking=True) for x in sample]
                label = label.to(device, non_blocking=True)
                pa = forward_probs(model_a, sample, [0])
                pb = forward_probs(model_b, sample, [0, 1])
                pred_a = pa.argmax(1).cpu().numpy()
                pred_b = pb.argmax(1).cpu().numpy()
                label_np = label.squeeze(1).cpu().numpy()

                if choices is not None:
                    pc = pa.clone()
                    for c, use_b in enumerate(choices):
                        if use_b:
                            pc[:, c] = pb[:, c]
                    pred_g = pc.argmax(1).cpu().numpy()
                else:
                    pred_g = None

                for b in range(label_np.shape[0]):
                    stem = ds.files[cursor + b]
                    cond = condition_of(stem)
                    h_a += hist_of(pred_a[b], label_np[b], n_classes)
                    h_b += hist_of(pred_b[b], label_np[b], n_classes)
                    if pred_g is not None:
                        hg = hist_of(pred_g[b], label_np[b], n_classes)
                        h_g += hg
                        cond_g[cond] += hg
                        if collect_per_image:
                            per_image.append({"name": Path(stem).name, "condition": cond,
                                              "hist": hg.ravel().tolist()})
                cursor += label_np.shape[0]
        return h_a, h_b, h_g, cond_g, per_image

    print(f"train: {len(ds_train)} images; test: {len(ds_test)} images")
    h_a_tr, h_b_tr, _, _, _ = run_split(ds_train)
    iou_a_tr, iou_b_tr = iou_table(h_a_tr), iou_table(h_b_tr)
    choices = []
    for c in range(n_classes):
        a = 0.0 if np.isnan(iou_a_tr[c]) else float(iou_a_tr[c])
        b = 0.0 if np.isnan(iou_b_tr[c]) else float(iou_b_tr[c])
        choices.append(bool(b > a))
    print("train per-class IoU A/B:", {c: (round(iou_a_tr[i], 2), round(iou_b_tr[i], 2))
                                      for i, c in enumerate(CLASSES)})
    print("choices (True=low-res RGB-T):", {c: choices[i] for i, c in enumerate(CLASSES)})

    h_a_te, h_b_te, h_g_te, cond_g_te, per_image = run_split(
        ds_test, choices=choices, collect_per_image=args.save_per_image)

    iou_a_te, iou_b_te = iou_table(h_a_te), iou_table(h_b_te)
    oracle_te = np.nanmax(np.vstack([iou_a_te, iou_b_te]), axis=0)
    train_gate = summarise(h_g_te)
    out = {
        "meta": {
            "cfg_a": args.cfg_a, "ckpt_a": args.ckpt_a,
            "cfg_b": args.cfg_b, "ckpt_b": args.ckpt_b,
            "root": args.root,
            "n_train": len(ds_train), "n_test": len(ds_test),
            "gate": "per-class model selection by train per-class IoU",
        },
        "train": {
            "A": summarise(h_a_tr), "B": summarise(h_b_tr),
            "choices": {c: bool(choices[i]) for i, c in enumerate(CLASSES)},
            "per_class_iou_A": {c: (None if np.isnan(v) else round(float(v), 3))
                                for c, v in zip(CLASSES, iou_a_tr)},
            "per_class_iou_B": {c: (None if np.isnan(v) else round(float(v), 3))
                                for c, v in zip(CLASSES, iou_b_tr)},
        },
        "test": {
            "A": summarise(h_a_te), "B": summarise(h_b_te),
            "gate": train_gate,
            "gate_by_condition": {c: summarise(h) for c, h in cond_g_te.items()},
            "oracle_per_class_analytic": {
                "miou_official": float(np.nan_to_num(oracle_te, nan=0.0).mean()),
                "miou_foreground": float(np.nan_to_num(oracle_te[FG], nan=0.0).mean()),
                "per_class_iou": {c: (None if np.isnan(v) else round(float(v), 3))
                                  for c, v in zip(CLASSES, oracle_te)},
            },
            "per_class_iou_gate": train_gate["per_class_iou"],
        },
        "per_image": per_image if args.save_per_image else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({
        "A_official": round(out["test"]["A"]["miou_official"], 3),
        "B_official": round(out["test"]["B"]["miou_official"], 3),
        "gate_official": round(out["test"]["gate"]["miou_official"], 3),
        "gate_fg": round(out["test"]["gate"]["miou_foreground"], 3),
        "gate_night_person": out["test"]["gate_by_condition"]["night"]["per_class_iou"]["person"],
        "oracle_official": round(out["test"]["oracle_per_class_analytic"]["miou_official"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
