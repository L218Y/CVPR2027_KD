#!/usr/bin/env python3
"""One evaluation protocol for every run that will be compared.

Every reported number for a checkpoint is produced here: same split, same
transform, same resolution, no test-time augmentation, no model selection.  The
trainer's own validation is only used during training and is never mixed into
these results.

    python3 scripts/eval_protocol.py --cfg configs/rgb_mfnet.yaml \
        --ckpt output/EXP/best.pth --out results/protocol/EXP_best.json

Pass ``--teacher-ckpt`` to add the pixel-level pairing table (teacher right /
student right), which is what the failure analysis needs.
"""

import argparse
import hashlib
import json
import subprocess
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


def sha256(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def condition_of(stem):
    return "night" if "N" in Path(str(stem)).name.upper()[:12] else "day"


def hist_of(pred, label, n_classes):
    keep = label != IGNORE
    idx = label[keep].astype(np.int64) * n_classes + pred[keep].astype(np.int64)
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def iou_table(hist):
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan) * 100.0
    acc = np.where(hist.sum(1) > 0, inter / np.maximum(hist.sum(1), 1), np.nan) * 100.0
    prec = np.where(hist.sum(0) > 0, inter / np.maximum(hist.sum(0), 1), np.nan) * 100.0
    return iou, acc, prec


def summarise(hist):
    iou, acc, prec = iou_table(hist)
    present = ~np.isnan(iou)
    return {
        # trainer semantics: a class with empty union scores 0 and is averaged in
        "miou_official": float(np.nan_to_num(iou, nan=0.0).mean()),
        "miou_present": float(np.nanmean(iou)) if present.any() else float("nan"),
        "miou_foreground": float(np.nan_to_num(iou[FG], nan=0.0).mean()),
        "macc_official": float(np.nan_to_num(acc, nan=0.0).mean()),
        "per_class_iou": {c: (None if np.isnan(v) else round(float(v), 3)) for c, v in zip(CLASSES, iou)},
        "per_class_acc": {c: (None if np.isnan(v) else round(float(v), 3)) for c, v in zip(CLASSES, acc)},
        "person": {
            "iou": None if np.isnan(iou[2]) else round(float(iou[2]), 3),
            "precision": None if np.isnan(prec[2]) else round(float(prec[2]), 3),
            "recall": None if np.isnan(acc[2]) else round(float(acc[2]), 3),
        },
        "pixels": int(hist.sum()),
        "gt_pixels": {c: int(hist[i].sum()) for i, c in enumerate(CLASSES)},
        "pred_pixels": {c: int(hist[:, i].sum()) for i, c in enumerate(CLASSES)},
    }


def pairing_table(t_pred, s_pred, label, n_classes, mask_fg_only=False):
    valid = label != IGNORE
    if mask_fg_only:
        valid = valid & (label != 0)
    t_ok = (t_pred == label) & valid
    s_ok = (s_pred == label) & valid
    out = {
        "valid": int(valid.sum()),
        "both_right": int((t_ok & s_ok).sum()),
        "teacher_only": int((t_ok & ~s_ok).sum()),
        "student_only": int((~t_ok & s_ok).sum()),
        "both_wrong": int((~t_ok & ~s_ok).sum()),
        "by_class": {},
    }
    for c in range(n_classes):
        m = valid & (label == c)
        n = int(m.sum())
        if n == 0:
            continue
        out["by_class"][CLASSES[c]] = {
            "valid": n,
            "both_right": int((t_ok & s_ok & m).sum()),
            "teacher_only": int((t_ok & ~s_ok & m).sum()),
            "student_only": int((~t_ok & s_ok & m).sum()),
            "both_wrong": int((~t_ok & ~s_ok & m).sum()),
        }
    return out


def add_pairing(acc, part):
    for k, v in part.items():
        if k == "by_class":
            dst = acc.setdefault("by_class", {})
            for cls, counts in v.items():
                d = dst.setdefault(cls, {kk: 0 for kk in counts})
                for kk, n in counts.items():
                    d[kk] += n
        else:
            acc[k] = acc.get(k, 0) + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", default=None, help="unused by the metrics; recorded for provenance")
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--teacher-cfg", default="configs/teacher_mfnet.yaml")
    ap.add_argument("--modals", default=None, nargs="+", help="override evaluated model's modalities")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-per-image", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    kd_cfg = cfg.get("KD") or {}
    dataset_cfg = cfg["DATASET"]

    if args.modals:
        modals = list(args.modals)
    elif kd_cfg.get("ENABLE", True):
        # a KD run always distils into an RGB-only student, which is what
        # tools/train_kd.py builds; ENABLE defaults to True when the block exists
        modals = ["img"]
    else:
        modals = list(dataset_cfg["MODALS"])

    transform = get_val_augmentation(cfg["EVAL"]["IMAGE_SIZE"])
    tcfg = yaml.safe_load(Path(args.teacher_cfg).read_text()) if args.teacher_ckpt else None
    t_modals = list((tcfg.get("DATASET") or {}).get("MODALS") or ["img", "thermal"]) if tcfg else []
    load_modals = list(dict.fromkeys(modals + t_modals))
    dataset = getattr(D, dataset_cfg["NAME"])(dataset_cfg["ROOT"], "val", transform, load_modals)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False, drop_last=False)
    n_classes = dataset.n_classes
    s_idx = [load_modals.index(m) for m in modals]
    t_idx = [load_modals.index(m) for m in t_modals]

    model = build_model(cfg, n_classes, modals).to(device)
    msg = load_state(args.ckpt, model, strict=True)
    model.eval()

    teacher = None
    if tcfg is not None:
        teacher = build_model(tcfg, n_classes, t_modals, backbone=tcfg["MODEL"]["BACKBONE"]).to(device)
        tmsg = load_state(args.teacher_ckpt, teacher, strict=True)
        teacher.eval()
        print(f"teacher: {tmsg}")

    total = np.zeros((n_classes, n_classes), np.int64)
    cond = {c: np.zeros((n_classes, n_classes), np.int64) for c in ("day", "night")}
    pair = {"all": {}, "foreground": {}}
    pair_cond = {c: {"all": {}, "foreground": {}} for c in ("day", "night")}
    per_image = []

    cursor = 0
    with torch.no_grad():
        for sample, label in loader:
            sample = [x.to(device, non_blocking=True) for x in sample]
            label = label.to(device, non_blocking=True)
            pred = model([sample[i] for i in s_idx]).argmax(1)
            pred_np = pred.cpu().numpy()
            label_np = label.squeeze(1).cpu().numpy()
            t_np = teacher([sample[i] for i in t_idx]).argmax(1).cpu().numpy() if teacher is not None else None

            for b in range(label_np.shape[0]):
                stem = dataset.files[cursor + b]
                c = condition_of(stem)
                h = hist_of(pred_np[b], label_np[b], n_classes)
                total += h
                cond[c] += h
                if not args.no_per_image:
                    per_image.append({"name": Path(stem).name, "condition": c, "hist": h.ravel().tolist()})
                if t_np is not None:
                    add_pairing(pair["all"], pairing_table(t_np[b], pred_np[b], label_np[b], n_classes))
                    add_pairing(pair["foreground"],
                                pairing_table(t_np[b], pred_np[b], label_np[b], n_classes, True))
                    add_pairing(pair_cond[c]["all"], pairing_table(t_np[b], pred_np[b], label_np[b], n_classes))
                    add_pairing(pair_cond[c]["foreground"],
                                pairing_table(t_np[b], pred_np[b], label_np[b], n_classes, True))
            cursor += label_np.shape[0]

    ckpt_meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    epoch = ckpt_meta.get("epoch") if isinstance(ckpt_meta, dict) else None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, cwd=REPO).stdout.strip()
    except Exception:
        commit = None

    out = {
        "meta": {
            "checkpoint": args.ckpt,
            "checkpoint_sha256": sha256(args.ckpt),
            "checkpoint_epoch": epoch,
            "load": msg,
            "probe": args.probe,
            "probe_sha256": sha256(args.probe) if args.probe else None,
            "cfg": args.cfg,
            "cfg_sha256": sha256(args.cfg),
            "modals": modals,
            "split": "val (MFNet test.txt held out by the official split)",
            "eval_split_file_sha256": sha256(Path(dataset_cfg["ROOT"]) / "test.txt"),
            "train_split_file_sha256": sha256(Path(dataset_cfg["ROOT"]) / "train.txt"),
            "n_images": len(dataset),
            "image_size": cfg["EVAL"]["IMAGE_SIZE"],
            "batch_size": args.batch_size,
            "tta": False,
            "code_commit": commit,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "classes": CLASSES,
            "ignore_label": IGNORE,
            "teacher": args.teacher_ckpt,
            "teacher_sha256": sha256(args.teacher_ckpt) if args.teacher_ckpt else None,
        },
        "overall": summarise(total),
        "conditions": {c: summarise(h) for c, h in cond.items()},
        "confusion": total.tolist(),
        "pairing": pair if teacher is not None else None,
        "pairing_by_condition": pair_cond if teacher is not None else None,
        "per_image": per_image,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({
        "checkpoint": args.ckpt,
        "epoch": epoch,
        "miou_official": round(out["overall"]["miou_official"], 3),
        "miou_foreground": round(out["overall"]["miou_foreground"], 3),
        "night_person_iou": out["conditions"]["night"]["person"]["iou"],
        "day_miou": round(out["conditions"]["day"]["miou_official"], 3),
        "night_miou": round(out["conditions"]["night"]["miou_official"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
