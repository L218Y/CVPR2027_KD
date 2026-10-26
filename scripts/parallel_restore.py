#!/usr/bin/env python3
"""Parallel restore of the CVPR2027_KD release assets.

Reads the asset listing emitted by the GitHub API (name, size, url) and writes
each asset to the same destination layout as scripts/restore_artifacts.sh.
Resumes partial files and verifies sizes.
"""
import concurrent.futures as cf
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = Path("/tmp/assets.txt")


def dest_for(name: str) -> Path:
    if name == "MFNet_dataset.tar.gz":
        return ROOT / "data" / name
    m = re.match(r"^(.*)_(best|last|probe)\.pth$", name)
    if m:
        run, ext = m.group(1), m.group(2)
        return ROOT / "output" / run / f"{ext}.pth"
    return ROOT / name


def fetch(item):
    name, size, url = item
    dest = dest_for(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(size)
    if dest.exists() and dest.stat().st_size == size:
        return f"have {dest.relative_to(ROOT)} ({size})"
    cmd = [
        "curl", "-L", "--retry", "5", "--retry-delay", "2",
        "--connect-timeout", "30", "-C", "-", "-o", str(dest) + ".part", url,
    ]
    # -C - cannot resume a .part that does not exist; curl handles it.
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return f"FAIL {name}: {p.stderr.decode()[-200:]}"
    part = Path(str(dest) + ".part")
    if part.stat().st_size != size:
        return f"SHORT {name}: {part.stat().st_size}/{size}"
    os.replace(part, dest)
    return f"got {dest.relative_to(ROOT)} ({size})"


def main():
    items = []
    for line in LIST.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            items.append(tuple(parts))
    print(f"{len(items)} assets", flush=True)
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for msg in ex.map(fetch, items):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
