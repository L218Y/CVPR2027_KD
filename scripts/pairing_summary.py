#!/usr/bin/env python3
"""Summarise where the teacher is right and the student is wrong, and vice versa.

This bounds what distillation could possibly fix: a pixel the teacher gets wrong
is not a target worth copying, and a pixel the student already gets right cannot
improve.  Reported per ground-truth class and per lighting condition because the
motivating claim is about night pedestrians, not about the 92% background.

    python3 scripts/pairing_summary.py results/protocol/P1_*_last.json
"""

import argparse
import json
from pathlib import Path

CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def row(name, counts):
    v = counts["valid"]
    return (
        f"| {name} | {v} | {pct(counts['both_right'], v):.2f} | {pct(counts['teacher_only'], v):.2f} | "
        f"{pct(counts['student_only'], v):.2f} | {pct(counts['both_wrong'], v):.2f} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--classes", nargs="+", default=["person", "guardrail", "car"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = ["# Teacher/student pixel pairing", "",
           "`teacher_only` is the only set distillation can win on this image set; "
           "`student_only` is where copying the teacher would lose.", ""]
    for path in args.evals:
        blob = json.loads(Path(path).read_text())
        if not blob.get("pairing"):
            out.append(f"## {path}\n\nno teacher pairing recorded\n")
            continue
        ck = blob["meta"]["checkpoint"]
        ck_epoch = blob["meta"]["checkpoint_epoch"]
        o = blob["overall"]
        out.append(f"## {Path(path).stem}")
        out.append("")
        out.append(f"`{ck}` epoch={ck_epoch} official mIoU={o['miou_official']:.2f} "
                   f"night person IoU={blob['conditions']['night']['person']['iou']}")
        out.append("")
        for cond in ("day", "night"):
            out.append(f"**{cond}**")
            out.append("")
            out.append("| gt class | px | both right % | teacher only % | student only % | both wrong % |")
            out.append("|---|---|---|---|---|---|")
            block = blob["pairing_by_condition"][cond]
            for cls in args.classes:
                counts = block["all"]["by_class"].get(cls)
                if counts:
                    out.append(row(cls, counts))
            allc = block["all"]
            out.append(row("ALL", allc))
            out.append("")
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
