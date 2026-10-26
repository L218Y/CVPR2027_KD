#!/usr/bin/env python3
"""Collect experiment outputs into a small, git-friendly results directory.

Only metrics, logs and small figures are copied.  Checkpoints stay in
``output/`` and are never pushed.

    python scripts/collect_results.py --status 0 --note "mfnet teacher"
"""

import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path

KEEP_LOG_LINES = 400


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip()
        return out.replace("\n", "; ")
    except Exception:
        return "cpu"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="output")
    args = ap.parse_args()

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    rows = []

    for metrics_path in sorted(Path(args.output).rglob("metrics.json")):
        payload = read_json(metrics_path)
        if not payload:
            continue
        rel = metrics_path.parent.relative_to(Path(args.output))
        exp = str(rel).replace(os.sep, "__")
        dst = results / exp
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_path, dst / "metrics.json")

        log_path = metrics_path.parent / "train.log"
        if log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            (dst / "train.log").write_text("\n".join(lines[-KEEP_LOG_LINES:]) + "\n")

        for pattern in ("*.png", "*.pdf", "*.csv"):
            for fig in metrics_path.parent.glob(pattern):
                if fig.stat().st_size < 5 * 1024 * 1024:
                    shutil.copy2(fig, dst / fig.name)

        rows.append({
            "experiment": exp,
            "dataset": payload.get("dataset", ""),
            "backbone": payload.get("backbone", ""),
            "teacher": payload.get("teacher_backbone", ""),
            "modals": "-".join(payload.get("modals", [])),
            "kd_mode": payload.get("kd_mode", ""),
            "seed": payload.get("seed", ""),
            "epochs": payload.get("epochs", ""),
            "best_miou": payload.get("best_miou", ""),
            "best_epoch": payload.get("best_epoch", ""),
            "train_time": payload.get("train_time", ""),
        })

    if rows:
        keys = list(rows[0].keys())
        with open(results / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    # Use the newest metrics.json mtime, not wall-clock time, so a watchdog
    # that re-collects unchanged runs does not create a commit every cycle.
    latest = max((p.stat().st_mtime for p in Path(args.output).rglob("metrics.json")), default=0)
    now = (dt.datetime.fromtimestamp(latest, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
           if latest else "no runs yet")
    lines = [
        "# Experiment status",
        "",
        f"- updated: {now}",
        f"- host: {socket.gethostname()} ({platform.platform()})",
        f"- gpu: {gpu_info()}",
        f"- commit: {git_rev()}",
        f"- exit status: {args.status}",
    ]
    if args.note:
        lines.append(f"- note: {args.note}")
    lines += ["", "| experiment | dataset | kd | seed | best mIoU | epoch | time |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['dataset']} | {r['kd_mode']} | {r['seed']} | "
            f"{r['best_miou']} | {r['best_epoch']} | {r['train_time']} |"
        )
    if not rows:
        lines.append("| - | - | - | - | - | - | - |")
    (results / "STATUS.md").write_text("\n".join(lines) + "\n")
    print(f"collected {len(rows)} experiment(s) into {results}/")


if __name__ == "__main__":
    main()
