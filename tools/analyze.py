"""Post-hoc analysis of a checkpoint: mIoU, calibration, hallucinations, r-maps.

    python3 tools/analyze.py --cfg configs/kd_mfnet.yaml --ckpt output/EXP/best.pth \
        --out results/EXP

Reports, on the validation split:
  * per-class IoU / accuracy and the aggregated mIoU / mAcc / aAcc;
  * expected calibration error (ECE, 15 bins) of the max softmax score;
  * hallucination rate: fraction of pixels predicted as a class that is absent
    from that image's ground truth (privileged-only evidence leaking into RGB);
  * a PNG of the recoverability map for a few validation images when the probe
    is available.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semseg.augmentations_mm import get_val_augmentation  # noqa: E402
from semseg.kd import FeatureProbe, RecoverabilityHead  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tools.train_kd import build_model, load_state  # noqa: E402


def confusion_matrix(pred, label, n_classes):
    mask = label != 255
    idx = label[mask].astype(np.int64) * n_classes + pred[mask].astype(np.int64)
    hist = np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return hist


def metrics_from_hist(hist):
    inter = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - inter
    iou = inter / np.maximum(union, 1)
    acc = inter / np.maximum(hist.sum(1), 1)
    return iou, acc


def ece(scores, correct, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, err = len(scores), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (scores > lo) & (scores <= hi)
        if sel.sum() == 0:
            continue
        err += sel.sum() * abs(correct[sel].mean() - scores[sel].mean())
    return float(err / max(total, 1))


def hallucination(pred, label):
    """Pixels predicted as a class that does not occur in this image's GT."""
    valid = label != 255
    present = np.zeros(int(pred.max()) + 2, dtype=bool)
    present[np.unique(label[valid])] = True
    absent = ~present[pred] & valid
    return float(absent.sum()) / float(max(valid.sum(), 1))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-rmaps", type=int, default=0, help="how many r-maps to dump")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg, mcfg, kd_cfg = cfg["DATASET"], cfg["MODEL"], cfg.get("KD", {})

    dataset_cls = eval(dcfg["NAME"])
    valset = dataset_cls(dcfg["ROOT"], "val", get_val_augmentation(cfg["EVAL"]["IMAGE_SIZE"]), ["img"])
    loader = DataLoader(valset, batch_size=cfg["EVAL"]["BATCH_SIZE"], num_workers=args.workers,
                        pin_memory=True, shuffle=False)
    n_classes = valset.n_classes

    model = build_model(cfg, n_classes, ["img"]).to(device)
    load_state(args.ckpt, model, strict=True)
    model.eval()

    probe = None
    kd_path = kd_cfg.get("TEACHER_PATH", "")
    if os.path.isfile(kd_path):
        teacher_modals = kd_cfg.get("TEACHER_MODALS", dcfg["MODALS"])
        teacher = build_model(cfg, n_classes, teacher_modals,
                              backbone=kd_cfg.get("TEACHER_BACKBONE", mcfg["BACKBONE"])).to(device)
        load_state(kd_path, teacher, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        probe = FeatureProbe(model.backbone.channels[-1], teacher.backbone.channels[-1],
                             hidden_dim=kd_cfg.get("PROBE_HIDDEN", 128)).to(device)
        probe_ckpt = os.path.join(os.path.dirname(args.ckpt), "probe.pth")
        if os.path.isfile(probe_ckpt):
            probe.load_state_dict(torch.load(probe_ckpt, map_location="cpu"))
        probe.eval()

    hist = np.zeros((n_classes, n_classes), dtype=np.int64)
    scores, corrects, halls = [], [], []
    rmap_dir = os.path.join(args.out or os.path.dirname(args.ckpt), "rmaps")
    saved = 0

    for i, batch in enumerate(loader):
        if args.max_images and i * loader.batch_size >= args.max_images:
            break
        img = batch["img"].to(device, non_blocking=True)
        label = batch["gt_seg_label_ignore_255"].squeeze(1).cpu().numpy()
        logits = model({"img": img})
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(1)
        pred = pred.cpu().numpy()
        conf = conf.cpu().numpy()
        valid = label != 255

        for k in range(label.shape[0]):
            hist += confusion_matrix(pred[k], label[k], n_classes)
            v = valid[k]
            scores.append(conf[k][v])
            corrects.append((pred[k][v] == label[k][v]).astype(np.float32))
            if v.sum() > 0:
                halls.append(hallucination(pred[k][v][None], label[k]))

        if probe is not None and args.save_rmaps and saved < args.save_rmaps:
            with torch.no_grad():
                s_feat = model.backbone({"img": img})
                t_modals = kd_cfg.get("TEACHER_MODALS", dcfg["MODALS"])
                t_in = {"img": img}
                for m in t_modals[1:]:
                    t_in[m] = batch[m].to(device, non_blocking=True)
                t_feat = teacher.backbone(t_in)
                r = probe(s_feat[-1], t_feat[-1])
            import matplotlib.pyplot as plt
            os.makedirs(rmap_dir, exist_ok=True)
            for k in range(min(label.shape[0], args.save_rmaps - saved)):
                fig, ax = plt.subplots(1, 4, figsize=(12, 2.6))
                rgb = img[k].cpu().numpy().transpose(1, 2, 0)
                rgb = (rgb - rgb.min()) / max(rgb.ptp(), 1e-6)
                ax[0].imshow(rgb); ax[0].set_title("RGB")
                ax[1].imshow(label[k], vmin=0, vmax=n_classes - 1); ax[1].set_title("GT")
                ax[2].imshow(pred[k], vmin=0, vmax=n_classes - 1); ax[2].set_title("Pred")
                im = ax[3].imshow(r[k].cpu().numpy(), vmin=0, vmax=1, cmap="viridis")
                ax[3].set_title("recoverability r")
                for a in ax[:3]:
                    a.axis("off")
                fig.colorbar(im, ax=ax[3], fraction=0.046)
                fig.tight_layout()
                fig.savefig(os.path.join(rmap_dir, f"rmap_{saved:03d}.png"), dpi=140)
                plt.close(fig)
                saved += 1

    iou, acc = metrics_from_hist(hist)
    scores = np.concatenate(scores); corrects = np.concatenate(corrects)
    result = {
        "mIoU": float(np.nanmean(iou) * 100),
        "mAcc": float(np.nanmean(acc) * 100),
        "aAcc": float(corrects.mean() * 100),
        "ECE": ece(scores, corrects),
        "hallucination_rate": float(np.mean(halls)) if halls else 0.0,
        "per_class": {name: {"iou": float(iou[i] * 100), "acc": float(acc[i] * 100)}
                      for i, name in enumerate(valset.CLASSES)},
        "n_images": int(hist.sum() and len(halls)),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "analysis.json"), "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
