#!/usr/bin/env python3
"""Build the PST900 working layout and low-resolution thermal variants.

The raw download (PST900_RGBT_Dataset) is left untouched.  We flatten the
official train/test PNGs into data/PST900/{rgb,thermal,labels}, create a
scene-disjoint gate-validation split from the official train bags, and build
bicubic round-trip thermal variants at 1/2, 1/4 and 1/8 resolution.

    python3 tools/prepare_pst900.py --scales 2 4 8
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/PST900/PST900_RGBT_Dataset"
OUT = ROOT / "data/PST900"

# Scene-disjoint gate-val bags (127 images).  No bag in this list appears in
# the official test set (bags 21-25), and none of these bags is used for gate
# training.
GATE_BAGS = {"bag17", "bag14", "bag19", "bag13", "bag16", "bag11a",
             "bag9a", "bag10a", "bag9", "bag10", "bag15"}


def names(split):
    return sorted(p.stem for p in (SRC / split / "rgb").glob("*.png"))


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src)


def build_layout():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "thermal", "labels"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for n in names(split):
            link(SRC / split / "rgb" / f"{n}.png", OUT / "rgb" / f"{n}.png")
            link(SRC / split / "thermal" / f"{n}.png", OUT / "thermal" / f"{n}.png")
            link(SRC / split / "labels" / f"{n}.png", OUT / "labels" / f"{n}.png")
    tr = names("train")
    gate = [n for n in tr if n.split("_")[1] in GATE_BAGS]
    fit = [n for n in tr if n not in set(gate)]
    test = names("test")
    (OUT / "train.txt").write_text("\n".join(fit) + "\n")
    (OUT / "test.txt").write_text("\n".join(gate) + "\n")
    (OUT / "official.txt").write_text("\n".join(test) + "\n")
    print(f"fit {len(fit)}  gate-val {len(gate)}  official test {len(test)}")
    return fit + gate + test


def make_lr(scale, all_names):
    dst = ROOT / f"data/PST900_lr{scale}"
    (dst / "thermal").mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "labels"):
        link((OUT / sub).resolve(), (dst / sub).resolve())
    for split in ("train.txt", "test.txt", "official.txt"):
        shutil.copyfile(OUT / split, dst / split)
    for i, n in enumerate(all_names, 1):
        im = Image.open(OUT / "thermal" / f"{n}.png").convert("L")
        w, h = im.size
        small = im.resize((max(1, w // scale), max(1, h // scale)), BICUBIC)
        up = small.resize((w, h), BICUBIC)
        up.save(dst / "thermal" / f"{n}.png")
        if i % 200 == 0:
            print(f"  lr{scale}: {i}/{len(all_names)}", flush=True)
    print(f"lr{scale}: {len(all_names)} images -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args()
    all_names = build_layout()
    for s in args.scales:
        make_lr(s, all_names)


if __name__ == "__main__":
    main()
