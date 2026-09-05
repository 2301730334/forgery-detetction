"""
Turns the raw document images into (ELA-map tensor, label) pairs for the
CNN -- this is the bridge between src/ela.py (a hand-written signal) and
src/model.py (a learned one). The CNN never sees the original document; it
only ever sees the ELA residual, which is the whole point: we're asking the
network to learn the texture-normalized anomaly pattern that a single
global statistic (src/ela.py's score) couldn't separate on its own.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ela import compute_ela, ela_heatmap

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(ROOT_DIR, "data")

IMG_SIZE = 128  # downsized working resolution -- fast on CPU, still large
                # enough for the amount-box-scale anomaly to survive resizing.


def _load_ela_tensor(path: str) -> torch.Tensor:
    img = Image.open(path)
    result = compute_ela(img, quality=90)
    heatmap = ela_heatmap(result.residual)  # (H, W) float32, roughly 0-255

    pil = Image.fromarray(heatmap.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


class ForgeryELADataset(Dataset):
    """label 0 = authentic, 1 = tampered (splice or copy-move, pooled --
    the CNN's job is 'was this touched', copy_move.py already tells us how)."""

    def __init__(self, root: str = DEFAULT_DATA_ROOT, cache: bool = True):
        auth = sorted(glob.glob(os.path.join(root, "authentic", "*.png")))
        tamp = sorted(glob.glob(os.path.join(root, "tampered", "*.png")))
        self.paths = auth + tamp
        self.labels = [0] * len(auth) + [1] * len(tamp)
        # ELA (JPEG re-encode + diff) is the expensive step, and it's
        # deterministic per source image, so compute it once up front rather
        # than redoing it every __getitem__ call across every epoch --
        # otherwise a 25-epoch run redoes thousands of JPEG encodes for
        # nothing. 360 images at 128x128 floats is ~23MB, trivial to hold
        # in memory.
        self._cache: list[torch.Tensor] | None = [None] * len(self.paths) if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if self._cache is not None and self._cache[idx] is not None:
            x = self._cache[idx]
        else:
            x = _load_ela_tensor(self.paths[idx])
            if self._cache is not None:
                self._cache[idx] = x
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def make_splits(root: str = DEFAULT_DATA_ROOT, val_frac: float = 0.2, seed: int = 0):
    """Stratified-ish split: shuffle once with a fixed seed so results are
    reproducible run to run, which matters once you're comparing MLflow
    runs against each other."""
    full = ForgeryELADataset(root)
    n = len(full)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train = torch.utils.data.Subset(full, train_idx)
    val = torch.utils.data.Subset(full, val_idx)
    return train, val


if __name__ == "__main__":
    ds = ForgeryELADataset()
    print(f"total samples: {len(ds)}")
    x, y = ds[0]
    print("tensor shape:", x.shape, "label:", y.item())
    train, val = make_splits()
    print(f"train: {len(train)}  val: {len(val)}")
