"""Materialise the hallucinated-thermal dataset for the baseline.

Builds `data/MFNet_hallu/` where `rgb/` and `labels/` are symlinks to the real
data and `ther/` holds generated thermal images for **both** splits. Training on
that tree with a two-stream model is exactly the standard "hallucinate the
missing modality, then run the multimodal network" pipeline; nothing in the
student code has to change, which is what makes the comparison honest.

    python3 tools/make_hallu_dataset.py            # writes every split
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from train_thermal_gen import UNet  # noqa: E402

SIZE = (480, 640)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="output/gen/thermal_unet.pth")
    ap.add_argument("--out", default="data/MFNet_hallu")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / args.gen, map_location="cpu")
    model = UNet().to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    print(f"generator epoch {ckpt['epoch']} val L1 {ckpt['val_l1']:.4f}")

    out = ROOT / args.out
    (out / "ther").mkdir(parents=True, exist_ok=True)
    for name in ("rgb", "labels"):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(Path("..") / "MFNet" / name, link)

    names = []
    for split in ("train", "test"):
        names += [l.strip() for l in (ROOT / f"data/MFNet/{split}.txt").read_text().split() if l.strip()]
        (out / f"{split}.txt").write_text((ROOT / f"data/MFNet/{split}.txt").read_text())

    done = 0
    with torch.no_grad():
        for i in range(0, len(names), 16):
            batch = []
            for name in names[i:i + 16]:
                im = Image.open(ROOT / f"data/MFNet/rgb/{name}.jpg").convert("RGB").resize(SIZE, Image.BILINEAR)
                batch.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1))
            pred = model(torch.stack(batch).to(device)).cpu()
            for name, p in zip(names[i:i + 16], pred):
                arr = ((p.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
                Image.fromarray(arr).save(out / "ther" / f"{name}.jpg", quality=95)
                done += 1
            if done % 160 < 16:
                print(f"  {done}/{len(names)}", flush=True)
    print(f"wrote {done} hallucinated thermal images to {out}/ther")


if __name__ == "__main__":
    main()
