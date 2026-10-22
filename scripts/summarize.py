"""Aggregate every run in results/ into one markdown table.

    python3 scripts/summarize.py [--results results] [--out results/SUMMARY.md]

Privileged Gap Recovery (PGR) is computed against the RGB-only baseline and the
fused teacher of the same dataset:

    PGR = (mIoU_run - mIoU_rgb) / (mIoU_teacher - mIoU_rgb)

A PGR of 0 means the student learned nothing from the teacher; 1 means the
student closed the whole privileged gap.
"""
import argparse
import json
from pathlib import Path

ORDER = ["rgb", "uniform", "hard", "conservative_only", "recoverability"]


def load(results: Path):
    runs = []
    for path in sorted(results.glob("*/metrics.json")):
        try:
            m = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "experiment" not in m:
            m["experiment"] = path.parent.name
        analysis = path.parent / "analysis.json"
        if analysis.is_file():
            m.update(json.loads(analysis.read_text()))
        runs.append(m)
    return runs


ROLES = ("teacher", "rgb", "uniform", "conservative_only", "recoverability")


def role(run):
    name = run["experiment"]
    for r in ROLES:
        if f"_{r}_s" in name or name.endswith(f"_{r}"):
            return r
    return run.get("kd_mode", "other")


def rank(run):
    r = role(run)
    return ORDER.index(r) if r in ORDER else len(ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/SUMMARY.md")
    args = ap.parse_args()

    runs = load(Path(args.results))
    if not runs:
        print("no runs found")
        return

    by_ds = {}
    for r in runs:
        by_ds.setdefault(r.get("dataset", "?"), []).append(r)

    lines = ["# Experiment summary", "", "| dataset | run | role | mIoU | mAcc | ECE | halluc. | PGR | best epoch |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds, group in by_ds.items():
        teacher = next((r for r in group if role(r) == "teacher"), None)
        rgb = next((r for r in group if role(r) == "rgb"), None)
        base = rgb["best_miou"] if rgb else 0.0
        top = teacher["best_miou"] if teacher else None

        for r in sorted(group, key=rank):
            gap = (top - base) if top is not None and top > base else None
            pgr = "—"
            if role(r) == "rgb":
                pgr = "0.000"
            elif gap and role(r) not in ("teacher", "rgb"):
                pgr = f"{(r['best_miou'] - base) / gap:.3f}"
            ece = f"{r['ECE']:.3f}" if "ECE" in r else "—"
            macc = f"{r['mAcc']:.2f}" if r.get("mAcc") == r.get("mAcc") and r.get("mAcc") is not None else "—"
            hall = f"{r['hallucination_rate']:.3f}" if "hallucination_rate" in r else "—"
            lines.append(
                f"| {ds} | {r['experiment']} | {role(r)} | {r['best_miou']:.2f} | "
                f"{macc} | {ece} | {hall} | {pgr} | {r.get('best_epoch', '—')} |"
            )
    lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
