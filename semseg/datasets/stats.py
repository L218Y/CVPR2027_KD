"""Training-split label statistics used to balance supervision.

Reads the raw label PNGs directly instead of iterating the dataset, so that
computing the counts consumes no RNG and cannot perturb the augmentation stream
of the run that uses them.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def train_label_counts(dataset, cache_dir: Optional[Path] = None) -> Tensor:
    root = Path(dataset.root)
    n_classes = dataset.n_classes
    key = f"{root.resolve()}::{len(dataset.files)}::{dataset.files[0]}::{dataset.files[-1]}"
    if cache_dir is not None:
        cache = Path(cache_dir) / "class_counts.json"
        if cache.is_file():
            blob = json.loads(cache.read_text())
            if blob.get("key") == key and len(blob["counts"]) == n_classes:
                return torch.tensor(blob["counts"], dtype=torch.float64)

    counts = np.zeros(n_classes, dtype=np.int64)
    for stem in dataset.files:
        arr = np.asarray(Image.open(root / "labels" / f"{stem}.png"))
        counts += np.bincount(arr.ravel(), minlength=n_classes)[:n_classes]

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "class_counts.json").write_text(
            json.dumps({"key": key, "classes": list(dataset.CLASSES), "counts": counts.tolist()}, indent=2)
        )
    return torch.tensor(counts, dtype=torch.float64)
