"""How much of the teacher's night-pedestrian advantage comes from thermal?

Runs the RGB-T teacher twice: normally, and with the thermal branch fed zeros.
If person IoU at night collapses to the RGB student's level in the second case,
the privileged information is genuinely thermal and cannot be expected to
appear from RGB features alone.
"""
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


def evaluate(model, dl, files, drop_thermal):
    h = {k: np.zeros((9, 9), np.int64) for k in ("day", "night")}
    with torch.no_grad():
        for i, (sample, label) in enumerate(dl):
            img, thm = sample[0].cuda(), sample[1].cuda()
            if drop_thermal:
                thm = torch.zeros_like(thm)
            pred = model([img, thm]).argmax(1).cpu().numpy()
            lab = label.squeeze(1).numpy()
            for k in range(lab.shape[0]):
                cond = "night" if "N" in os.path.basename(str(files[i * 4 + k])).upper()[:12] else "day"
                v = lab[k] != 255
                h[cond] += np.bincount(lab[k][v].astype(np.int64) * 9 + pred[k][v],
                                       minlength=81).reshape(9, 9)
    out = {}
    for cond, hh in h.items():
        inter = np.diag(hh)
        union = hh.sum(1) + hh.sum(0) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan) * 100
        out[cond] = {"mIoU": float(np.nanmean(iou)),
                     **{c: float(iou[j]) for j, c in enumerate(C) if union[j] > 0}}
    return out


cfg = yaml.safe_load(open("configs/teacher_mfnet.yaml"))
ds = getattr(D, "MFNet")(cfg["DATASET"]["ROOT"], "val",
                         get_val_augmentation(cfg["EVAL"]["IMAGE_SIZE"]), ["img", "thermal"])
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
model = build_model(cfg, 9, ["img", "thermal"]).cuda()
load_state("output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth", model, strict=True)
model.eval()

for drop in (False, True):
    r = evaluate(model, dl, ds.files, drop)
    tag = "thermal zeroed" if drop else "thermal on"
    print("%-14s day mIoU %5.2f person %5.2f | night mIoU %5.2f person %5.2f"
          % (tag, r["day"]["mIoU"], r["day"]["person"], r["night"]["mIoU"], r["night"]["person"]))
