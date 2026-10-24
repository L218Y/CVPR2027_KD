#!/usr/bin/env python3
"""Fold the per-arm protocol evaluations into one small auditable JSON.

The full evaluation files keep 392 per-image confusion matrices (~300 KB each) so
the paired bootstrap can be rerun. This summary keeps every reported number plus
the checkpoint hash, and is small enough to review inside the repository.
"""

import argparse
import glob
import json
from pathlib import Path

KEEP_META = ("checkpoint", "checkpoint_sha256", "checkpoint_epoch", "cfg", "cfg_sha256",
             "modals", "n_images", "image_size", "tta", "code_commit", "teacher_sha256")


def compact(path):
    d = json.loads(Path(path).read_text())
    out = {"meta": {k: d["meta"].get(k) for k in KEEP_META}}
    out["overall"] = d["overall"]
    out["conditions"] = d["conditions"]
    if d.get("pairing"):
        out["pairing"] = {k: v for k, v in d["pairing"].items()}
        out["pairing_by_condition"] = {
            c: {"all": b["all"]} for c, b in d["pairing_by_condition"].items()
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/protocol/P1_*_last.json")
    ap.add_argument("--out", default="results/protocol/summary_p1.json")
    args = ap.parse_args()
    blob = {Path(p).stem: compact(p) for p in sorted(glob.glob(args.pattern))}
    Path(args.out).write_text(json.dumps(blob, indent=1))
    print(args.out, Path(args.out).stat().st_size, "bytes,", len(blob), "runs")


if __name__ == "__main__":
    main()
