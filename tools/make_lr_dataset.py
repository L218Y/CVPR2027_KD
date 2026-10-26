"""Build low-resolution thermal variants of MFNet for the sensor-budget gate.

Each variant keeps the original RGB and labels, and replaces the thermal image
with a sensor-realistic round trip: downsample to (W/s, H/s) with bicubic, then
upsample back to the original size with bicubic.  The segmentation loader is
unchanged; only the thermal content is degraded.

    python3 tools/make_lr_dataset.py --scales 4 8
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image

BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/MFNet"


def split_names():
    names = []
    for split in ("train", "test"):
        for line in (SRC / f"{split}.txt").read_text().splitlines():
            name = line.strip().split()[0] if line.strip() else ""
            if name:
                names.append(name)
    return sorted(set(names))


def make_variant(scale: int, names):
    dst = ROOT / f"data/MFNet_lr{scale}"
    (dst / "ther").mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "labels"):
        link = dst / sub
        if link.is_symlink():
            continue
        if not link.exists():
            link.symlink_to(Path("..") / "MFNet" / sub, target_is_directory=True)
    for split in ("train", "test"):
        shutil.copyfile(SRC / f"{split}.txt", dst / f"{split}.txt")

    for i, name in enumerate(names, 1):
        im = Image.open(SRC / "ther" / f"{name}.jpg").convert("RGB")
        w, h = im.size
        small = im.resize((max(1, w // scale), max(1, h // scale)), BICUBIC)
        up = small.resize((w, h), BICUBIC)
        up.save(dst / "ther" / f"{name}.jpg", quality=95)
        if i % 200 == 0:
            print(f"  lr{scale}: {i}/{len(names)}", flush=True)
    print(f"lr{scale}: {len(names)} images -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[4, 8])
    args = ap.parse_args()
    names = split_names()
    print(f"source pairs: {len(names)}")
    for s in args.scales:
        make_variant(s, names)


if __name__ == "__main__":
    main()
