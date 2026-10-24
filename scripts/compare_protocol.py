#!/usr/bin/env python3
"""Compare protocol evaluations with an image-level paired bootstrap.

Every eval JSON produced by scripts/eval_protocol.py stores the per-image
confusion histograms, so two runs can be compared on exactly the same images.
The bootstrap resamples images (the independent unit), rebuilds the global
confusion matrix and recomputes mIoU, which keeps the CI on the same statistic
that is reported.

    python3 scripts/compare_protocol.py --ref CE=results/protocol/CE.json \
        --arm KD=results/protocol/KD.json --out results/protocol/compare.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

FG = list(range(1, 9))
CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]


def load(path):
    blob = json.loads(Path(path).read_text())
    hists = np.asarray([im["hist"] for im in blob["per_image"]], dtype=np.int64)
    names = [im["name"] for im in blob["per_image"]]
    conds = np.asarray([im["condition"] for im in blob["per_image"]])
    return blob, hists, names, conds


def metrics_from_hist(hist, classes=None):
    hist = hist.reshape(9, 9)
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0) * 100.0
    if classes is None:
        return float(iou.mean())
    return float(iou[classes].mean())


def paired_bootstrap(a, b, classes=None, n=2000, seed=0):
    """delta = metric(a) - metric(b) over the same resampled image set."""
    rng = np.random.default_rng(seed)
    k = a.shape[0]
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        deltas[i] = metrics_from_hist(a[idx].sum(0), classes) - metrics_from_hist(b[idx].sum(0), classes)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="LABEL=path.json (repeatable)")
    ap.add_argument("--ref", required=True, help="LABEL=path.json used as the reference")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        label, path = spec.split("=", 1)
        blob, hists, names, conds = load(path)
        arms.append((label, blob, hists, names, conds))
    ref_label = args.ref.split("=", 1)[0]
    ref_blob, ref_h, ref_names, ref_conds = load(args.ref.split("=", 1)[1])
    for label, blob, _, names, _ in arms:
        if names != ref_names:
            raise SystemExit(f"{label}: image order differs from {ref_label}; cannot pair")

    lines = ["| arm | official mIoU | foreground mIoU | day mIoU | night mIoU | night person | guardrail | d mIoU vs ref [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    payload = {"reference": ref_label, "arms": {}}
    for label, blob, hists, _, conds in [(ref_label, ref_blob, ref_h, ref_names, ref_conds)] + [
        (a[0], a[1], a[2], a[3], a[4]) for a in arms
    ]:
        o = blob["overall"]
        day, night = blob["conditions"]["day"], blob["conditions"]["night"]
        mean, lo, hi = paired_bootstrap(hists, ref_h, n=args.boot)
        star = "" if label == ref_label else ("*" if (lo > 0 or hi < 0) else "")
        lines.append(
            f"| {label} | {o['miou_official']:.2f} | {o['miou_foreground']:.2f} | "
            f"{day['miou_official']:.2f} | {night['miou_official']:.2f} | "
            f"{night['person']['iou']} | {o['per_class_iou']['guardrail']} | "
            f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]{star} |"
        )
        payload["arms"][label] = {
            "miou_official": o["miou_official"],
            "miou_foreground": o["miou_foreground"],
            "day_miou": day["miou_official"],
            "night_miou": night["miou_official"],
            "night_person_iou": night["person"]["iou"],
            "person_iou": o["person"]["iou"],
            "guardrail_iou": o["per_class_iou"]["guardrail"],
            "per_class_iou": o["per_class_iou"],
            "delta_vs_ref": {"mean": mean, "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)},
            "checkpoint": blob["meta"]["checkpoint"],
            "checkpoint_sha256": blob["meta"]["checkpoint_sha256"],
            "epoch": blob["meta"]["checkpoint_epoch"],
            "pairing": blob.get("pairing"),
        }
        if blob.get("pairing"):
            p = blob["pairing"]["all"]
            payload["arms"][label]["teacher_only_frac"] = p["teacher_only"] / max(p["valid"], 1)

    text = "\n".join(lines) + "\n\n* significant at the 95% level under an image-level paired bootstrap.\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        Path(args.out).with_suffix(".json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
