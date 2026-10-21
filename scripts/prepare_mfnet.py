#!/usr/bin/env python3
"""Convert the raw MFNet release into the layout expected by semseg.datasets.MFNet.

Raw layout (either of these works)::

    <src>/MFNet/{RGB,Thermal,Label}/*.png
    <src>/{RGB,Thermal,Label}/*.png
    <src>/{train,test}/{RGB,Thermal,Label}/*.png

Output::

    data/MFNet/rgb/<stem>.jpg
    data/MFNet/ther/<stem>.jpg
    data/MFNet/labels/<stem>.png
    data/MFNet/train.txt, test.txt

If the source ships ``train.txt``/``test.txt`` they are reused; otherwise a
deterministic 80/20 split is written and flagged in ``SPLIT_SOURCE.txt``.
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


def find_dir(root: Path, name: str) -> Path | None:
    for candidate in root.rglob(name):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def stems(folder: Path) -> set[str]:
    return {p.stem for p in folder.glob("*.png")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory containing the extracted MFNet folder")
    ap.add_argument("--dst", default="data/MFNet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    rgb_dir, th_dir, lb_dir = (find_dir(src, n) for n in ("RGB", "Thermal", "Label"))
    if not all((rgb_dir, th_dir, lb_dir)):
        raise SystemExit(f"could not find RGB/Thermal/Label under {src}")

    common = sorted(stems(rgb_dir) & stems(th_dir) & stems(lb_dir))
    if not common:
        raise SystemExit("no common file stems across RGB/Thermal/Label")
    print(f"found {len(common)} aligned pairs in {rgb_dir.parent}")

    for sub in ("rgb", "ther", "labels"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(common):
        rgb = Image.open(rgb_dir / f"{stem}.png").convert("RGB")
        rgb.save(dst / "rgb" / f"{stem}.jpg", quality=95)
        th = Image.open(th_dir / f"{stem}.png")
        th = th.convert("L").convert("RGB")
        th.save(dst / "ther" / f"{stem}.jpg", quality=95)
        lb = Image.open(lb_dir / f"{stem}.png").convert("L")
        lb.save(dst / "labels" / f"{stem}.png")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(common)}")

    train_txt = next((p for p in src.rglob("train.txt")), None)
    test_txt = next((p for p in src.rglob("test.txt")), None)
    if train_txt and test_txt:
        def clean(path):
            names = []
            for line in path.read_text().splitlines():
                name = line.strip().split()[0] if line.strip() else ""
                if name:
                    names.append(Path(name).stem)
            return [n for n in names if n in set(common)]

        train, test = clean(train_txt), clean(test_txt)
        source = f"official ({train_txt}, {test_txt})"
    else:
        names = list(common)
        random.Random(args.seed).shuffle(names)
        cut = int(0.8 * len(names))
        train, test = sorted(names[:cut]), sorted(names[cut:])
        source = f"deterministic 80/20 seed={args.seed} (NO official split found)"

    (dst / "train.txt").write_text("\n".join(train) + "\n")
    (dst / "test.txt").write_text("\n".join(test) + "\n")
    (dst / "SPLIT_SOURCE.txt").write_text(source + "\n")
    print(f"train={len(train)} test={len(test)}  source={source}")


if __name__ == "__main__":
    main()
