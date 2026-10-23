"""Canonical statistics for every run, recomputed from its own log.

The training log can contain epoch lines from an earlier aborted attempt in the
same experiment directory, so duplicates are removed by keeping the last
occurrence of each epoch before the windows are computed.  This is the single
source of truth for the numbers quoted in PROGRESS.md / papers.
"""
import argparse
import json
import re
from pathlib import Path

RUNS = {
    "teacher": ("3407", "output/MFNet_CMNeXt-B2_it_teacher_s3407/train.log", None),
    "rgb": ("3407", "output/MFNet_CMNeXt-B2_i_rgb_s3407/train.log", None),
    "uniform": ("3407", "logs/S1_uniform_kd.log", None),
    "rapd0.5": ("3407", "logs/S2_rapd.log", None),
    "cons": ("3407", "logs/S3_conservative_only.log", None),
    "rapd0.2": ("3407", "logs/S4_rapd_eps_0.2.log", None),
    "teacher": ("3408", "logs/seed3408_teacher.log", None),
}
EXTRA = ["logs/seed{seed}_{name}.log"]


def history(path):
    text = Path(path).read_text(errors="ignore").replace("\r", "\n")
    seen = {}
    for ep, miou in re.findall(r"epoch (\d+) mIoU ([\d.]+)", text):
        seen[int(ep)] = float(miou)
    return sorted(seen.items())


def summary(path):
    h = history(path)
    if not h:
        return None
    vals = [v for _, v in h]
    last10 = vals[-10:]
    return {
        "n_eval": len(h),
        "best": round(max(vals), 3),
        "best_epoch": h[vals.index(max(vals))][0],
        "last10": round(sum(last10) / len(last10), 3),
        "last5": round(sum(vals[-5:]) / 5, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/VERIFIED.md")
    args = ap.parse_args()

    seeds = ["3407", "3408", "3409"]
    names = ["teacher", "rgb", "uniform", "rapd0.5"]
    table = {}
    for seed in seeds:
        for name in names:
            cands = [f"logs/seed{seed}_{name.replace('rapd0.5','rapd')}.log"]
            if seed == "3407":
                cands = [
                    "output/MFNet_CMNeXt-B2_it_teacher_s3407/train.log",
                    "output/MFNet_CMNeXt-B2_i_rgb_s3407/train.log",
                    "logs/S1_uniform_kd.log",
                    "logs/S2_rapd.log",
                ][names.index(name):names.index(name) + 1]
            for c in cands:
                if not Path(c).is_file():
                    continue
                s = summary(c)
                if s:
                    s["source"] = c
                    table[(seed, name)] = s

    lines = ["# Verified statistics", "",
             "Computed from each run's own log after removing duplicated epoch lines.", "",
             "| seed | method | evals | best | best ep | last-10 | last-5 |", "|---|---|---|---|---|---|---|"]
    for (seed, name), s in sorted(table.items()):
        lines.append(f"| {seed} | {name} | {s['n_eval']} | {s['best']:.2f} | {s['best_epoch']} | "
                     f"{s['last10']:.2f} | {s['last5']:.2f} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
