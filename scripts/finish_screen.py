#!/usr/bin/env python3
"""Wait for the five screening arms, evaluate them under one protocol, publish.

Runs after `run_screen.sh`. It does not decide anything: it measures every arm
with `eval_protocol.py` (fixed final epoch, no TTA, no model selection), records
the deltas against `ce`, and writes `results/RESULTS_MIN_SCREEN.md`.

Interpretation rules are pre-registered in docs/MINIMAL_EXPERIMENTS.md; this
script only reports numbers and the mechanical facts about them.

    python3 scripts/finish_screen.py --epochs 120 --seed 3407
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
TEACHER = "output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth"
ARMS = ["ce", "cbce", "uniform_kd", "cbce_kd", "correctness_kd"]


def metrics_path(arm, tag):
    return sorted((REPO / "output/min" / arm).glob(f"*_{tag}/metrics.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--tag-prefix", default="screen")
    ap.add_argument("--timeout-hours", type=float, default=9.0)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    print(f"waiting for 5 arms (timeout {args.timeout_hours}h)", flush=True)
    deadline = time.time() + args.timeout_hours * 3600
    while time.time() < deadline:
        missing = [a for a in ARMS if not metrics_path(a, f"{args.tag_prefix}_{a}")]
        if not missing:
            break
        print(f"[{time.strftime('%H:%M')}] waiting for: {', '.join(missing)}", flush=True)
        time.sleep(120)
    else:
        print("timeout: not all arms finished", file=sys.stderr)
        return 1

    out_dir = REPO / "results/protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    for arm in ARMS:
        run = metrics_path(arm, f"{args.tag_prefix}_{arm}")[0].parent
        out = out_dir / f"MIN_{arm}_{args.seed}_e{args.epochs}.json"
        cmd = [
            sys.executable, "scripts/eval_protocol.py",
            "--cfg", f"configs/min/{arm}.yaml",
            "--ckpt", str(run / "last.pth"),
            "--teacher-ckpt", TEACHER,
            "--out", str(out),
        ]
        print("eval", arm, flush=True)
        rc = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=".:tools")).returncode
        if rc != 0 or not out.is_file():
            print(f"eval failed for {arm} (rc={rc})", file=sys.stderr)
            return 1
        d = json.loads(out.read_text())
        o = d["overall"]
        rows[arm] = {
            "official": o["miou_official"],
            "fg": o["miou_foreground"],
            "day": d["conditions"]["day"]["miou_official"],
            "night": d["conditions"]["night"]["miou_official"],
            "night_person": d["conditions"]["night"]["person"]["iou"],
            "train_last10": json.loads((run / "metrics.json").read_text())["last10_miou"],
            "epoch": d["meta"]["checkpoint_epoch"],
        }

    base = rows["ce"]
    for arm in ARMS:
        rows[arm]["d_fg"] = rows[arm]["fg"] - base["fg"]
        rows[arm]["d_np"] = (rows[arm]["night_person"] or 0) - (base["night_person"] or 0)

    md = [f"# Minimal screen ({args.epochs} epochs, seed {args.seed})", ""]
    md.append("Fixed final epoch, no TTA, no model selection. Student is RGB-only at test time.")
    md.append("")
    md.append("| arm | epoch | official | foreground | day | night | night person | last-10 (train) | Δfg vs ce | Δnight-person vs ce |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        r = rows[arm]
        md.append(
            f"| {arm} | {r['epoch']} | {r['official']:.2f} | {r['fg']:.2f} | {r['day']:.2f} | "
            f"{r['night']:.2f} | {r['night_person']:.2f} | {r['train_last10']:.2f} | "
            f"{r['d_fg']:+.2f} | {r['d_np']:+.2f} |"
        )
    md.append("")
    md.append("The two contrasts that matter are pre-registered in `docs/MINIMAL_EXPERIMENTS.md` §6:")
    md.append("")
    md.append("- `cbce − ce`: how much of the gain is plain class re-weighting.")
    md.append(f"- `cbce_kd − cbce`: how much the teacher adds on top of identical CE treatment "
              f"({rows['cbce_kd']['fg'] - rows['cbce']['fg']:+.2f} foreground).")
    md.append("")
    md.append("Single seed, 120 epochs: these numbers rank candidates, they do not establish "
              "significance, and they must not be compared against the 500-epoch tables.")
    (REPO / "results/RESULTS_MIN_SCREEN.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    subprocess.run(
        [sys.executable, "scripts/compare_protocol.py",
         "--ref", f"ce={out_dir}/MIN_ce_{args.seed}_e{args.epochs}.json",
         "--arm", f"cbce={out_dir}/MIN_cbce_{args.seed}_e{args.epochs}.json",
         "--arm", f"uniform_kd={out_dir}/MIN_uniform_kd_{args.seed}_e{args.epochs}.json",
         "--arm", f"cbce_kd={out_dir}/MIN_cbce_kd_{args.seed}_e{args.epochs}.json",
         "--arm", f"correctness_kd={out_dir}/MIN_correctness_kd_{args.seed}_e{args.epochs}.json",
         "--out", str(out_dir / f"MIN_compare_{args.seed}_e{args.epochs}.md")],
        cwd=REPO, check=False,
    )

    if args.no_push:
        return 0

    pat_file = Path(os.environ.get("GIT_PAT_FILE", "/data/lab/.git_pat"))
    if not pat_file.is_file():
        print("no PAT file; leaving the commit local", file=sys.stderr)
        return 0
    token = pat_file.read_text().strip()
    paths = ["results/RESULTS_MIN_SCREEN.md"]
    paths += [f"results/protocol/MIN_{a}_{args.seed}_e{args.epochs}.json" for a in ARMS]
    paths += [f"results/protocol/MIN_compare_{args.seed}_e{args.epochs}.md"]
    subprocess.run(["git", "add", *paths], cwd=REPO, check=False)
    subprocess.run(
        ["git", "commit", "-m",
         f"results: minimal screen at {args.epochs} epochs, seed {args.seed}\n\n"
         "Five arms with parity-checked configs, evaluated at a fixed final epoch with no\n"
         "TTA and no model selection. Single seed: ranks candidates, does not establish\n"
         "significance."],
        cwd=REPO, check=False,
    )
    subprocess.run(
        ["git", "push", f"https://x-access-token:{token}@github.com/L218Y/CVPR2027_KD.git", "HEAD:main"],
        cwd=REPO, check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
