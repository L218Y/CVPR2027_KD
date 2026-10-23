"""Recompute day/night/pooled per-class IoU for two checkpoints and reconcile
the pooled mIoU with the value the trainer logged."""
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import semseg.datasets as D
from semseg.augmentations_mm import get_val_augmentation
from train_kd import build_model, load_state

C = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "cone", "bump"]


def run(cfg_path, ckpt, modals, tag):
    cfg = yaml.safe_load(open(cfg_path))
    ds = getattr(D, "MFNet")(cfg["DATASET"]["ROOT"], "val",
                             get_val_augmentation(cfg["EVAL"]["IMAGE_SIZE"]), modals)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    model = build_model(cfg, 9, modals).cuda()
    load_state(ckpt, model, strict=True)
    model.eval()
    h = {k: np.zeros((9, 9), np.int64) for k in ("day", "night")}
    with torch.no_grad():
        for i, (sample, label) in enumerate(dl):
            x = [t.cuda() for t in sample][:len(modals)]
            pred = model(x).argmax(1).cpu().numpy()
            lab = label.squeeze(1).numpy()
            for k in range(lab.shape[0]):
                cond = "night" if "N" in os.path.basename(str(ds.files[i * 4 + k])).upper()[:12] else "day"
                v = lab[k] != 255
                h[cond] += np.bincount(lab[k][v].astype(np.int64) * 9 + pred[k][v],
                                       minlength=81).reshape(9, 9)
    pooled = h["day"] + h["night"]
    print("\n==", tag)
    print("%-10s %8s %8s %8s %8s" % ("class", "day", "night", "pooled", "pixels"))
    ious = {}
    for c, j in zip(C, range(9)):
        vals = []
        for hh in (h["day"], h["night"], pooled):
            u = hh[j].sum() + hh[:, j].sum() - hh[j, j]
            vals.append(100.0 * hh[j, j] / u if u > 0 else np.nan)
        ious[c] = vals
        print("%-10s %8.2f %8.2f %8.2f %8d" % (c, vals[0], vals[1], vals[2], pooled[j].sum()))
    for k, j in (("day", 0), ("night", 1), ("pooled", 2)):
        arr = np.array([ious[c][j] for c in C])
        print("%-10s mIoU(all 9) %6.2f | mIoU(present only) %6.2f"
              % (k, np.nanmean(arr), np.nanmean(arr[~np.isnan(arr)])))


run("configs/teacher_mfnet.yaml", "output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth",
    ["img", "thermal"], "teacher s3407 best")
run("configs/rgb_mfnet.yaml", "output/MFNet_CMNeXt-B2_i_rgb_s3407/best.pth",
    ["img"], "rgb s3407 best")
