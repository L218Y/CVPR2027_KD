#!/usr/bin/env python3
"""Wait for the nine confirmation runs, evaluate them, and publish the table.

Three arms (ce / cbce / cbce_kd) x three seeds at 500 epochs. The decisive number
is `cbce_kd - cbce` computed per seed, reported as mean +- spread: with n=1 the
interval only described evaluation noise, which is why this run exists.

    python3 scripts/finish_confirm.py
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEACHER = "output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth"
ARMS = ["ce", "cbce", "cbce_kd"]
SEEDS = [3407, 3408, 3409]


def run_dir(arm, seed, tag):
    hits = sorted((REPO / "output/min" / arm).glob(f"*_{tag}_{arm}_s{seed}"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--tag", default="confirm")
    ap.add_argument("--timeout-hours", type=float, default=20.0)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    deadline = time.time() + args.timeout_hours * 3600
    while time.time() < deadline:
        missing = [f"{a}_s{s}" for a in ARMS for s in SEEDS
                   if not (run_dir(a, s, args.tag) and (run_dir(a, s, args.tag) / "metrics.json").is_file())]
        if not missing:
            break
        print(f"[{time.strftime('%H:%M')}] waiting for {len(missing)}: {', '.join(missing)}", flush=True)
        time.sleep(180)
    else:
        print("timeout", file=sys.stderr)
        return 1

    out_dir = REPO / "results/protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for arm in ARMS:
        for seed in SEEDS:
            run = run_dir(arm, seed, args.tag)
            out = out_dir / f"CONFIRM_{arm}_s{seed}_e{args.epochs}.json"
            if not out.is_file():
                cmd = [sys.executable, "scripts/eval_protocol.py",
                       "--cfg", f"configs/min/{arm}.yaml",
                       "--ckpt", str(run / "last.pth"),
                       "--teacher-ckpt", TEACHER,
                       "--out", str(out)]
                print(f"eval {arm} s{seed}", flush=True)
                if subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=".:tools")).returncode != 0:
                    print(f"eval failed: {arm} s{seed}", file=sys.stderr)
                    return 1
            d = json.loads(out.read_text())
            o = d["overall"]
            rows[(arm, seed)] = {
                "fg": o["miou_foreground"],
                "official": o["miou_official"],
                "np": d["conditions"]["night"]["person"]["iou"],
            }

    md = [f"# Confirmation: {args.epochs} epochs, three seeds", "",
          "Fixed final epoch, no TTA, no model selection, RGB-only at test time.", ""]
    md += ["| arm | seed | official | foreground | night person |", "|---|---|---|---|---|"]
    for arm in ARMS:
        for seed in SEEDS:
            r = rows[(arm, seed)]
            md.append(f"| {arm} | {seed} | {r['official']:.2f} | {r['fg']:.2f} | {r['np']:.2f} |")

    md += ["", "## Per-arm spread (foreground)", "",
           "| arm | mean | sd | min | max |", "|---|---|---|---|---|"]
    stats = {}
    for arm in ARMS:
        vals = [rows[(arm, s)]["fg"] for s in SEEDS]
        stats[arm] = (statistics.mean(vals), statistics.stdev(vals))
        md.append(f"| {arm} | {stats[arm][0]:.2f} | {stats[arm][1]:.2f} | {min(vals):.2f} | {max(vals):.2f} |")

    deltas = [rows[("cbce_kd", s)]["fg"] - rows[("cbce", s)]["fg"] for s in SEEDS]
    reweight = [rows[("cbce", s)]["fg"] - rows[("ce", s)]["fg"] for s in SEEDS]
    np_deltas = [rows[("cbce_kd", s)]["np"] - rows[("cbce", s)]["np"] for s in SEEDS]

    md += ["", "## Pre-registered contrasts", "",
           f"- `cbce - ce` (re-weighting): " + ", ".join(f"{d:+.2f}" for d in reweight) +
           f"  -> mean {statistics.mean(reweight):+.2f}, sd {statistics.stdev(reweight):.2f}",
           f"- `cbce_kd - cbce` (the teacher): " + ", ".join(f"{d:+.2f}" for d in deltas) +
           f"  -> mean {statistics.mean(deltas):+.2f}, sd {statistics.stdev(deltas):.2f}",
           f"- night person, `cbce_kd - cbce`: " + ", ".join(f"{d:+.2f}" for d in np_deltas) +
           f"  -> mean {statistics.mean(np_deltas):+.2f}",
           ""]
    if abs(statistics.mean(deltas)) < 0.7 and statistics.stdev(deltas) < 0.7:
        md.append("Verdict under the pre-registered rule: **no measurable teacher contribution** "
                  "(mean and spread both inside +-0.7).")
    elif statistics.mean(deltas) > 1.0:
        md.append("Verdict under the pre-registered rule: the teacher shows a contribution worth "
                  "pursuing (> +1.0 mean).")
    else:
        md.append("Verdict under the pre-registered rule: **inconclusive** — the mean sits between "
                  "+-0.7 and +1.0, or the spread is too large.")

    (REPO / "results/RESULTS_CONFIRM.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    if args.no_push:
        return 0
    pat = Path(os.environ.get("GIT_PAT_FILE", "/data/lab/.git_pat"))
    if not pat.is_file():
        return 0
    paths = ["results/RESULTS_CONFIRM.md"] + [f"results/protocol/CONFIRM_{a}_s{s}_e{args.epochs}.json"
                                              for a in ARMS for s in SEEDS]
    subprocess.run(["git", "add", *paths], cwd=REPO, check=False)
    subprocess.run(["git", "commit", "-m",
                    f"results: three-seed confirmation at {args.epochs} epochs\n\n"
                    "ce / cbce / cbce_kd x seeds 3407-3409, fixed final epoch. Reports the "
                    "per-seed decisive contrast so the claim no longer rests on one draw."],
                   cwd=REPO, check=False)
    subprocess.run(["git", "push", f"https://x-access-token:{pat.read_text().strip()}"
                                   "@github.com/L218Y/CVPR2027_KD.git", "HEAD:main"],
                   cwd=REPO, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
