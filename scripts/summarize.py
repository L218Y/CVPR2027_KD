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


def rank(run):
    mode = run.get("kd_mode", "rgb")
    stage = "teacher" if run["experiment"].endswith("teacher") or mode == "teacher" else mode
    return ORDER.index(mode) if mode in ORDER else len(ORDER)


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

    lines = ["# Experiment summary", "", "| dataset | run | mode | mIoU | mAcc | ECE | halluc. | PGR | best epoch |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds, group in by_ds.items():
        teacher = next((r for r in group if r["experiment"].endswith("teacher")), None)
        rgb = next((r for r in group if r.get("kd_mode") in (None, "none", "rgb")
                    and not r["experiment"].endswith("teacher")), None)
        base = rgb["best_miou"] if rgb else 0.0
        top = teacher["best_miou"] if teacher else None

        for r in sorted(group, key=rank):
            gap = (top - base) if top is not None and top > base else None
            pgr = "—"
            if gap and r is not rgb:
                pgr = f"{(r['best_miou'] - base) / gap:.3f}"
            elif r is rgb:
                pgr = "0.000"
            ece = f"{r['ECE']:.3f}" if "ECE" in r else "—"
            hall = f"{r['hallucination_rate']:.3f}" if "hallucination_rate" in r else "—"
            lines.append(
                f"| {ds} | {r['experiment']} | {r.get('kd_mode', '—')} | {r['best_miou']:.2f} | "
                f"{r.get('mAcc', float('nan')):.2f} | {ece} | {hall} | {pgr} | {r.get('best_epoch', '—')} |"
            )
    lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
