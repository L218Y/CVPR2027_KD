"""Day/night stratified evaluation and per-class recoverability statistics.

    python3 scripts/diag_split.py --cfg configs/kd_mfnet.yaml \
        --ckpt output/MFNet_CMNeXt-B2_it_recoverability_s3407/best.pth \
        --out results/diag_rapd

MFNet filenames encode the capture condition: ``00001D`` is day, ``00844N`` is
night.  The script reports mIoU per condition, per-class IoU inside each
condition, and - when the teacher checkpoint exists - the distribution of the
recoverability score r over the pixels of each class.  That distribution tells us
whether a class is suppressed because r is high (the probe believes RGB suffices)
or because the conservative target is too strong (r low but the class still fails).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from semseg.augmentations_mm import get_val_augmentation  # noqa: E402
from semseg.kd import FeatureProbe  # noqa: E402
from train_kd import build_model, load_state  # noqa: E402


def hist_add(hist, pred, label, n):
    m = label != 255
    idx = label[m].astype(np.int64) * n + pred[m].astype(np.int64)
    return hist + np.bincount(idx, minlength=n * n).reshape(n, n)


def iou_of(hist):
    inter = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - inter
    return inter / np.maximum(union, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg, mcfg, kd = cfg["DATASET"], cfg["MODEL"], cfg.get("KD", {})
    teacher_modals = kd.get("TEACHER_MODALS", dcfg["MODALS"])
    has_teacher = os.path.isfile(kd.get("TEACHER_PATH", ""))

    ds = eval(dcfg["NAME"])
    modals = teacher_modals if has_teacher else ["img"]
    valset = ds(dcfg["ROOT"], "val", get_val_augmentation(cfg["EVAL"]["IMAGE_SIZE"]), modals)
    loader = DataLoader(valset, batch_size=cfg["EVAL"]["BATCH_SIZE"], num_workers=args.workers, pin_memory=True)
    n = valset.n_classes

    model = build_model(cfg, n, ["img"]).to(device)
    print(load_state(args.ckpt, model, strict=True))
    model.eval()

    probe = teacher = None
    if has_teacher:
        teacher = build_model(cfg, n, teacher_modals,
                              backbone=kd.get("TEACHER_BACKBONE", mcfg["BACKBONE"])).to(device)
        load_state(kd["TEACHER_PATH"], teacher, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        probe = FeatureProbe(model.backbone.channels[-1], teacher.backbone.channels[-1],
                             hidden_dim=kd.get("PROBE_HIDDEN", 128)).to(device)
        pck = os.path.join(os.path.dirname(args.ckpt), "probe.pth")
        if os.path.isfile(pck):
            probe.load_state_dict(torch.load(pck, map_location="cpu"))
            print("probe loaded")
        probe.eval()

    hists = {"day": np.zeros((n, n), dtype=np.int64), "night": np.zeros((n, n), dtype=np.int64)}
    r_sum = {i: 0.0 for i in range(n)}
    r_cnt = {i: 0 for i in range(n)}

    for i, (sample, label) in enumerate(loader):
        if args.max_images and i * loader.batch_size >= args.max_images:
            break
        files = valset.files[i * loader.batch_size:(i + 1) * loader.batch_size]
        sample = [x.to(device) for x in sample]
        label_np = label.squeeze(1).cpu().numpy()

        feats = model.backbone(sample[:1])
        logits = model.decode_head(feats)
        logits = F.interpolate(logits, size=label_np.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy()

        for k in range(label_np.shape[0]):
            cond = "night" if "N" in os.path.basename(str(files[k])).upper()[:12] else "day"
            hists[cond] = hist_add(hists[cond], pred[k], label_np[k], n)

        if probe is not None:
            r = probe.score(feats[-1], teacher.backbone(sample)[-1])
            r = F.interpolate(r.unsqueeze(1), size=label_np.shape[-2:], mode="bilinear",
                              align_corners=False).squeeze(1).cpu().numpy()
            for k in range(label_np.shape[0]):
                lab = label_np[k]
                for c in range(n):
                    sel = lab == c
                    if sel.any():
                        r_sum[c] += float(r[k][sel].sum())
                        r_cnt[c] += int(sel.sum())

    out = {"checkpoint": args.ckpt, "teacher": kd.get("TEACHER_PATH", ""), "conditions": {}}
    for cond, h in hists.items():
        iou = iou_of(h)
        out["conditions"][cond] = {
            "mIoU": float(np.nanmean(iou) * 100),
            "n_pixels": int(h.sum()),
            "per_class": {name: float(iou[j] * 100) for j, name in enumerate(valset.CLASSES)},
        }
    out["r_by_class"] = {
        name: (r_sum[i] / r_cnt[i] if r_cnt[i] else None)
        for i, name in enumerate(valset.CLASSES)
    }
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "day_night.json"), "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
