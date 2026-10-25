#!/usr/bin/env python3
"""One-epoch, two-iteration end-to-end check for every arm in a comparison.

This is the gate that must pass *before* any long run: it proves that each arm
builds, loads its teacher, applies its own KD term, writes a checkpoint and
survives validation.  It deliberately does not measure accuracy -- one epoch on
784 images says nothing about quality, and treating it as a result is how the
earlier batch ended up with unattributable numbers.

Usage:
    python3 scripts/smoke_pipeline.py                       # configs/min/*.yaml
    python3 scripts/smoke_pipeline.py --pretrained ''        # no ImageNet weights
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def expected(flat_cfg):
    """What a faithful run of this config must show in final_stats."""
    kd = flat_cfg.get("KD", {})
    enabled = bool(kd.get("ENABLE", True))
    mode = kd.get("MODE", "recoverability") if enabled else "none"
    gate = str(kd.get("GATE", "none"))
    return {
        "kd": enabled,
        "r_is_one": mode == "uniform",
        "select": enabled and mode == "task_aware",
        "gate": gate,
    }


def check(stats, exp, name):
    problems = []
    for key in ("loss", "kd"):
        value = stats.get(key)
        if value is None or not math.isfinite(value):
            problems.append(f"{key} is not finite ({value!r})")
        elif value < 0:
            problems.append(f"{key} is negative ({value!r})")

    if exp["kd"]:
        if not stats.get("kd", 0.0) > 0:
            problems.append(f"KD enabled but kd={stats.get('kd')!r} (term not applied)")
    elif stats.get("kd", 0.0) != 0.0:
        problems.append(f"KD disabled but kd={stats.get('kd')!r}")

    if exp["r_is_one"] and abs(float(stats.get("r", 0.0)) - 1.0) > 1e-6:
        problems.append(f"uniform KD should use r=1, got r={stats.get('r')!r}")

    if exp["select"]:
        frac = float(stats.get("sel", 0.0))
        if not 0.0 < frac <= 1.0:
            problems.append(f"gate {exp['gate']} selected no usable pixels (sel={frac!r})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pretrained", default="", help="override MODEL.PRETRAINED ('' = random init)")
    args = ap.parse_args()

    configs = [Path(p) for p in (args.configs or sorted((REPO / "configs/min").glob("*.yaml")))]
    if not configs:
        print("no configs found", file=sys.stderr)
        return 2

    print(f"smoke: {len(configs)} arms, {args.epochs} epoch x {args.max_iters} iters each\n")
    rows, failed = [], 0

    for cfg_path in configs:
        cfg = yaml.safe_load(cfg_path.read_text())
        flat = {
            "TRAIN.VAL": cfg.get("TRAIN"),
            "KD": cfg.get("KD", {}),
        }
        exp = expected(flat)

        # PRETRAINED is overridden on a temp copy so the checked-in configs stay
        # exactly what a real run uses.
        configured = cfg["MODEL"].get("PRETRAINED") or ""
        if args.pretrained != "":
            override, note = args.pretrained, f"PRETRAINED overridden to {args.pretrained!r}"
        elif configured and Path(configured).is_file():
            override, note = None, "configured pretrained weights"
        else:
            override, note = "", f"random init (missing {configured or '<unset>'})"

        if override is None:
            run_cfg = cfg_path
        else:
            tmp = Path(tempfile.mkdtemp()) / cfg_path.name
            cfg["MODEL"]["PRETRAINED"] = override
            tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
            run_cfg = tmp

        cmd = [
            sys.executable, "tools/train_kd.py",
            "--cfg", str(run_cfg),
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--max-iters", str(args.max_iters),
            "--workers", str(args.workers),
            "--tag", f"smoke_{cfg_path.stem}",
        ]
        env = dict(os.environ, PYTHONPATH=".:tools", CUDA_VISIBLE_DEVICES=str(args.gpu))
        log = REPO / "output/min" / f"{cfg_path.stem}.smoke.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        before = set((REPO / "output/min").glob("**/metrics.json"))
        with log.open("w") as fh:
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        written = sorted(set((REPO / "output/min").glob("**/metrics.json")) - before)

        problems = []
        stats = {}
        if proc.returncode != 0:
            problems.append(f"train_kd exited {proc.returncode} (see {log.relative_to(REPO)})")
        elif len(written) != 1:
            problems.append(f"expected exactly 1 new metrics.json, got {len(written)}")
        else:
            stats = json.loads(written[0].read_text()).get("final_stats", {})
            problems += check(stats, exp, cfg_path.stem)
            if not (written[0].parent / "last.pth").is_file():
                problems.append("last.pth missing")

        failed += bool(problems)
        rows.append((cfg_path.stem, note, stats, problems))
        status = "PASS" if not problems else "FAIL"
        print(f"[{status}] {cfg_path.stem:16s} loss={stats.get('loss', float('nan')):.4f} "
              f"kd={stats.get('kd', float('nan')):.4f} r={stats.get('r', float('nan')):.3f} "
              f"sel={stats.get('sel', float('nan')):.3f}  ({note})")
        for p in problems:
            print(f"        - {p}")

    print()
    if failed:
        print(f"SMOKE FAILED for {failed}/{len(rows)} arms")
        return 1
    print(f"SMOKE PASSED for all {len(rows)} arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
