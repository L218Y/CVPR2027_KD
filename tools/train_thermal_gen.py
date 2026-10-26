"""RGB -> thermal image translation, the standard naive hallucination baseline.

The question this answers is not "can we win", it is "did you test what the field
actually does for train-multimodal / deploy-RGB?"  The usual answer is to
hallucinate the missing modality and run a two-stream model on the fake pair.

This trains a small U-Net on the 784 training pairs with an L1 reconstruction
loss. It is deliberately plain: a faithful baseline should be reproducible in a
few hundred lines, not tuned until it flatters the paper.

    python3 tools/train_thermal_gen.py --epochs 40
"""

import argparse
import fcntl
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


class Pairs(Dataset):
    def __init__(self, split: str, size=(480, 640)):
        self.names = [l.strip() for l in (ROOT / f"data/MFNet/{split}.txt").read_text().split() if l.strip()]
        self.size = size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        rgb = Image.open(ROOT / f"data/MFNet/rgb/{name}.jpg").convert("RGB").resize(self.size, Image.BILINEAR)
        ther = Image.open(ROOT / f"data/MFNet/ther/{name}.jpg").convert("RGB").resize(self.size, Image.BILINEAR)
        to = lambda im: torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        return to(rgb), to(ther), name


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1, self.d2, self.d3, self.d4 = block(3, base), block(base, base * 2), block(base * 2, base * 4), block(base * 4, base * 8)
        self.u3 = block(base * 8 + base * 4, base * 4)
        self.u2 = block(base * 4 + base * 2, base * 2)
        self.u1 = block(base * 2 + base, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        d = self.d4(F.max_pool2d(c, 2))
        u = self.u3(torch.cat([F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False), c], 1))
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2, mode="bilinear", align_corners=False), b], 1))
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2, mode="bilinear", align_corners=False), a], 1))
        return torch.tanh(self.out(u))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="output/gen/thermal_unet.pth")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--force", action="store_true", help="retrain even if --out exists")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # One trainer at a time. If another GPU already produced the checkpoint,
    # reuse it instead of training a duplicate on this GPU.
    lock = open(out_path.with_suffix(out_path.suffix + ".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if out_path.exists() and not args.force:
        print(f"reusing {out_path} (pass --force to retrain)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(Pairs("train"), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    val = DataLoader(Pairs("test"), batch_size=args.batch_size, shuffle=False,
                     num_workers=args.workers, pin_memory=True)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for rgb, ther, _ in train:
            rgb, ther = rgb.to(device), ther.to(device)
            pred = model(rgb)
            # L1 keeps edges; the gradient term stops the blur that L1 alone produces.
            loss = F.l1_loss(pred, ther) + 0.5 * F.l1_loss(
                pred[:, :, :, 1:] - pred[:, :, :, :-1], ther[:, :, :, 1:] - ther[:, :, :, :-1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * rgb.size(0)
        sched.step()

        model.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for rgb, ther, _ in val:
                pred = model(rgb.to(device))
                val_l1 += F.l1_loss(pred, ther.to(device)).item() * rgb.size(0)
        val_l1 /= len(val.dataset)
        print(f"epoch {epoch + 1}/{args.epochs}  train {total / len(train.dataset):.4f}  "
              f"val L1 {val_l1:.4f}  {time.time() - t0:.0f}s", flush=True)
        if val_l1 < best:
            best = val_l1
            Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"generator": model.state_dict(), "val_l1": val_l1, "epoch": epoch + 1}, ROOT / args.out)
    print(f"best val L1 {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
