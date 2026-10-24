#!/usr/bin/env python3
"""Build the P1 report tables from protocol evaluations.

Emits one Markdown document: fixed-checkpoint comparison with bootstrap CIs,
validation-selected numbers, per-class IoU, and the teacher/student pairing
breakdown. Everything is read from results/protocol/*.json, so the report cannot
drift from the measured files.
"""

import argparse
import glob
import json
from pathlib import Path

CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
ARMS = [("CE", "ce"), ("Uniform", "uniform"), ("Correctness", "correctness"),
        ("ClassBalance", "classbalance"), ("Both", "both"), ("BalancedCE", "cbce")]


def load(kind):
    out = {}
    for label, tag in ARMS:
        p = Path(f"results/protocol/P1_{tag}_{kind}.json")
        if p.is_file():
            blob = json.loads(p.read_text())
            blob["_per_image"] = [im["hist"] for im in blob["per_image"]]
            out[label] = blob
    return out


def miou_from_hist(hist, idx=None):
    import numpy as np
    h = np.asarray(hist, dtype=np.int64).reshape(9, 9)
    inter = np.diag(h).astype(float)
    union = h.sum(1) + h.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0) * 100
    return float(iou[idx].mean() if idx is not None else iou.mean())


def boot(a_hists, b_hists, n=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(a_hists, np.int64)
    b = np.asarray(b_hists, np.int64)
    k = a.shape[0]
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        d[i] = miou_from_hist(a[idx].sum(0)) - miou_from_hist(b[idx].sum(0))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/REPORT_P1.md")
    args = ap.parse_args()
    last, best = load("last"), load("best")
    L = ["# P1 report (generated)", "",
         "Every number is read from `results/protocol/P1_*.json`, produced by "
         "`scripts/eval_protocol.py` on the fixed 392-image split with no TTA.", ""]

    for kind, runs in (("last", last), ("best", best)):
        if not runs:
            continue
        title = "Fixed epoch-500 checkpoint (no model selection)" if kind == "last" \
            else "Validation-selected best (secondary, optimistic)"
        L += [f"## {title}", "",
              "| arm | official mIoU | foreground mIoU | day mIoU | night mIoU | person | night person | guardrail | d mIoU vs CE [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|"]
        ref = runs.get("CE")
        for label in [a for a, _ in ARMS if a in runs]:
            b = runs[label]
            o, day, night = b["overall"], b["conditions"]["day"], b["conditions"]["night"]
            if ref is not None:
                m, lo, hi = boot(b["_per_image"], ref["_per_image"])
                ci = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if (lo > 0 or hi < 0) else "")
            else:
                ci = "-"
            L.append(f"| {label} | {o['miou_official']:.2f} | {o['miou_foreground']:.2f} | "
                     f"{day['miou_official']:.2f} | {night['miou_official']:.2f} | "
                     f"{o['person']['iou']} | {night['person']['iou']} | "
                     f"{o['per_class_iou']['guardrail']} | {ci} |")
        L += ["", "*CI excludes 0.", ""]

        L += ["### Per-class IoU", "",
              "| arm | " + " | ".join(CLASSES[1:]) + " |", "|---" * 9 + "|"]
        for label in [a for a, _ in ARMS if a in runs]:
            pc = runs[label]["overall"]["per_class_iou"]
            L.append(f"| {label} | " + " | ".join(str(pc[c]) for c in CLASSES[1:]) + " |")
        L.append("")

    # pairing on the fixed checkpoint
    if "CE" in last:
        L += ["## Teacher/student pairing at epoch 500", "",
              "`teacher only` = teacher right, student wrong (the only pixels distillation can win); "
              "`student only` = the reverse (copying the teacher loses here).", "",
              "| arm | cond | gt | px | both right % | teacher only % | student only % | both wrong % |",
              "|---|---|---|---|---|---|---|---|"]
        for label in [a for a, _ in ARMS if a in last]:
            b = last[label]
            if not b.get("pairing_by_condition"):
                continue
            for cond in ("day", "night"):
                blk = b["pairing_by_condition"][cond]["all"]
                for cls in ("person", "guardrail"):
                    c = blk["by_class"].get(cls)
                    if not c:
                        continue
                    v = c["valid"]
                    L.append(f"| {label} | {cond} | {cls} | {v} | "
                             f"{100*c['both_right']/v:.2f} | {100*c['teacher_only']/v:.2f} | "
                             f"{100*c['student_only']/v:.2f} | {100*c['both_wrong']/v:.2f} |")
                v = blk["valid"]
                L.append(f"| {label} | {cond} | ALL | {v} | {100*blk['both_right']/v:.2f} | "
                         f"{100*blk['teacher_only']/v:.2f} | {100*blk['student_only']/v:.2f} | "
                         f"{100*blk['both_wrong']/v:.2f} |")
        L.append("")

    L += ["## Provenance", "", "| arm | checkpoint | sha256 | epoch |", "|---|---|---|---|"]
    for label in [a for a, _ in ARMS if a in last]:
        m = last[label]["meta"]
        L.append(f"| {label} | `{m['checkpoint']}` | `{m['checkpoint_sha256'][:16]}...` | {m['checkpoint_epoch']} |")
    text = "\n".join(L) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
