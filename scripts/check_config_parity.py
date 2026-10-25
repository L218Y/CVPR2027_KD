#!/usr/bin/env python3
"""Fail if two arm configs differ in anything except the intended knob.

A comparison is only an ablation if every setting outside the manipulated
factor is identical.  In this project that was not true across the earlier
P0/P1 arms: ``TRAIN.DETERMINISTIC`` was set on the task-aware arms but absent
on the RGB baseline and the uniform-KD arm, and the uniform arm carried
``LAMBDA_FEAT: 0.1`` while the gated arms used ``0.0``.  Both differences change
the numerics or the loss, so cross-arm deltas were not attributable.

Usage:
    python3 scripts/check_config_parity.py configs/min/*.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

# Keys that are allowed to differ between arms, with the reason.  Everything
# else must match; an absent key is reported rather than treated as a default.
ALLOWED = {
    "SAVE_DIR": "experiment naming only",
    "EVAL.MODEL_PATH": "bookkeeping only; eval_protocol.py takes --ckpt explicitly",
    "DATASET.MODALS": "whether the teacher stream is read (student stays RGB either way)",
    "KD.ENABLE": "the factor under test",
    "KD.MODE": "the factor under test",
    "KD.GATE": "the factor under test",
    "KD.FOREGROUND_ONLY": "the factor under test",
    "LOSS.CLASS_BALANCE.ENABLE": "the factor under test",
}


def flatten(node, prefix=""):
    out = {}
    for key, value in node.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def allowed(key):
    return key in ALLOWED or any(key.startswith(k + ".") for k in ALLOWED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args()

    paths = [Path(p) for p in args.configs]
    if len(paths) < 2:
        print("need at least two configs to compare", file=sys.stderr)
        return 2

    flat = {str(p): flatten(yaml.safe_load(p.read_text())) for p in paths}
    keys = sorted({k for f in flat.values() for k in f})

    violations, intended = [], []
    for key in keys:
        values = [f.get(key, "<absent>") for f in flat.values()]
        if len({repr(v) for v in values}) == 1:
            continue
        row = (key, list(zip([p.name for p in paths], values)))
        (intended if allowed(key) else violations).append(row)

    if intended:
        print("intended differences")
        for key, pairs in intended:
            print(f"  {key:34s} {ALLOWED.get(key, 'factor under test')}")
            for name, value in pairs:
                print(f"      {name:24s} {value!r}")
        print()

    if violations:
        print(f"PARITY FAILED: {len(violations)} unintended difference(s)", file=sys.stderr)
        for key, pairs in violations:
            print(f"  {key}", file=sys.stderr)
            for name, value in pairs:
                print(f"      {name:24s} {value!r}", file=sys.stderr)
        print(
            "\nFix the configs so that only the documented factor varies. "
            "An absent key counts as a difference, not as a default.",
            file=sys.stderr,
        )
        return 1

    print(f"parity ok: {len(paths)} configs, {len(intended)} intended difference(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
