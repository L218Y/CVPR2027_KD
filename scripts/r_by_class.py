#!/usr/bin/env python3
"""Per-class recoverability of the privileged evidence.

The probe is trained inside the KD run; here it is frozen and evaluated over the
validation split, aggregating r by ground-truth class and by lighting
condition. The point is to separate two failure hypotheses that the mean mIoU
cannot distinguish: a class whose evidence the probe already calls recoverable
(method never backs off) from one where it calls it privileged (method backs off
and therefore withholds exactly the useful signal).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import semseg.datasets as _datasets
from semseg.augmentations_mm import get_val_augmentation
from semseg.kd.probe import FeatureProbe
from train_kd import build_model, load_state

CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop",
           "guardrail", "color_cone", "bump"]


def unpack(batch):
    if isinstance(batch, dict):
        return None, batch["gt_seg_label_ignore_255"]
    sample, label = batch[0], batch[1]
    while isinstance(sample, (list, tuple)) and sample and not torch.is_tensor(sample[0]):
        sample = sample[0]
    return sample, label


def condition_of(path):
    return "night" if "N" in os.path.basename(str(path)).upper()[:12] else "day"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output directory of the KD run")
    ap.add_argument("--cfg", default="configs/kd_mfnet_uniform.yaml")
    ap.add_argument("--teacher-cfg", default="configs/teacher_mfnet.yaml")
    ap.add_argument("--student-ckpt", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    tcfg = yaml.safe_load(open(args.teacher_cfg))
    kd = cfg.get("KD") or {}
    dcfg = cfg["DATASET"]
    n_classes = cfg.get("MODEL", {}).get("NUM_CLASSES", 9)

    student_ckpt = args.student_ckpt or os.path.join(args.run, "last.pth")
    probe_path = args.probe or os.path.join(args.run, "probe.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    student = build_model(cfg, n_classes, ["img"]).to(device)
    load_state(student_ckpt, student, strict=True)
    student.eval()

    t_modals = list(kd.get("TEACHER_MODALS") or tcfg["DATASET"]["MODALS"])
    teacher = build_model(tcfg, n_classes, t_modals).to(device)
    t_path = kd.get("TEACHER_PATH")
    if t_path and os.path.isfile(t_path):
        load_state(t_path, teacher, strict=True)
    else:
        raise SystemExit("teacher checkpoint missing: %s" % t_path)
    teacher.eval()

    probe = FeatureProbe(student.backbone.channels[-1], teacher.backbone.channels[-1],
                         hidden_dim=kd.get("PROBE_HIDDEN", 128)).to(device)
    probe.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=False))
    probe.eval()

    valset = getattr(_datasets, dcfg["NAME"])(
        dcfg["ROOT"], "val", get_val_augmentation(cfg["EVAL"]["IMAGE_SIZE"]), t_modals)
    loader = DataLoader(valset, batch_size=4, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    sum_r = {c: np.zeros(n_classes) for c in ("all", "day", "night")}
    cnt = {c: np.zeros(n_classes) for c in ("all", "day", "night")}
    files = valset.files

    with torch.no_grad():
        for i, batch in enumerate(loader):
            sample, label = unpack(batch)
            if sample is None:
                continue
            img = sample[0].to(device, non_blocking=True)
            t_in = [t.to(device, non_blocking=True) for t in sample][:len(t_modals)]
            s_feat = student.backbone([img])[-1]
            t_feat = teacher.backbone(t_in)[-1]
            r, _ = probe(s_feat, t_feat)
            r = F.interpolate(r.unsqueeze(1), size=label.shape[-2:], mode="bilinear",
                              align_corners=False).squeeze(1).cpu().numpy()
            lab = label.squeeze(1).numpy()
            base = i * 4
            for k in range(lab.shape[0]):
                valid = lab[k] != 255
                if not valid.any():
                    continue
                cond = condition_of(files[base + k])
                cls = lab[k][valid].astype(np.int64)
                vals = r[k][valid]
                for key in ("all", cond):
                    sum_r[key] += np.bincount(cls, weights=vals, minlength=n_classes)
                    cnt[key] += np.bincount(cls, minlength=n_classes)

    out = {}
    for key in ("all", "day", "night"):
        out[key] = {c: (float(sum_r[key][j] / cnt[key][j]) if cnt[key][j] > 0 else None)
                    for j, c in enumerate(CLASSES)}
        out[key + "_pixels"] = {c: int(cnt[key][j]) for j, c in enumerate(CLASSES)}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "r_by_class.json"), "w") as fh:
        json.dump({"run": args.run, "cfg": args.cfg, "r": out}, fh, indent=2)

    print("%-10s %8s %8s %8s" % ("class", "all", "day", "night"))
    for c in CLASSES:
        if cnt["all"][CLASSES.index(c)] == 0:
            continue
        print("%-10s %8.3f %8.3f %8.3f" % (
            c, out["all"][c],
            out["day"][c] if out["day"][c] is not None else float("nan"),
            out["night"][c] if out["night"][c] is not None else float("nan")))


if __name__ == "__main__":
    main()
