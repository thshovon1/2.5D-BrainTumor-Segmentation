"""Preprocessed-data access, 2.5D input construction and balanced slice sampling.

Preprocessed layout (written by scripts/preprocess.py), one folder per subject:

    <data_dir>/<subject_id>/image.npy          float16 (D, 4, S, S)  z-scored T1, T1ce, T2, FLAIR
    <data_dir>/<subject_id>/label.npy          uint8   (D, S, S)     BraTS labels on the model grid
    <data_dir>/<subject_id>/label_native.npz   uint8   (D, H, W)     original labels (native grid)
    <data_dir>/<subject_id>/meta.json          geometry mode, native shape, voxel spacing

Images are stored slice-major so that the three slices needed for one 2.5D
sample are a single contiguous read from the memory-mapped file.

The 2.5D input for axial slice i is the concatenation of slices i-1, i, i+1
across the four sequences -> 12 channels ordered
[prev(T1, T1ce, T2, FLAIR), curr(T1, T1ce, T2, FLAIR), next(T1, T1ce, T2, FLAIR)];
out-of-range neighbours at the first and last slice are all-zero
(paper Section 3.2).
"""

import csv
import json
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import regions_from_label


def read_split(splits_dir: str, name: str):
    path = os.path.join(splits_dir, f"{name}.csv")
    with open(path, newline="") as f:
        return [r["subject_id"] for r in csv.DictReader(f)]


def load_meta(data_dir, sid):
    with open(os.path.join(data_dir, sid, "meta.json")) as f:
        return json.load(f)


def load_image(data_dir, sid, mmap=True):
    """(D, 4, S, S) float16."""
    return np.load(os.path.join(data_dir, sid, "image.npy"), mmap_mode="r" if mmap else None)


def load_label(data_dir, sid, mmap=True):
    """(D, S, S) uint8 BraTS labels on the model grid."""
    return np.load(os.path.join(data_dir, sid, "label.npy"), mmap_mode="r" if mmap else None)


def load_native_label(data_dir, sid):
    """(D, H, W) uint8 BraTS labels on the original grid."""
    with np.load(os.path.join(data_dir, sid, "label_native.npz")) as z:
        return z["label"]


def stack_25d(image: np.ndarray, i: int) -> np.ndarray:
    """image: (D, 4, S, S) -> (12, S, S) float32 for axial slice i."""
    d = image.shape[0]
    lo, hi = max(i - 1, 0), min(i + 2, d)
    out = np.zeros((3,) + tuple(image.shape[1:]), dtype=np.float32)
    out[lo - (i - 1):hi - (i - 1)] = image[lo:hi]
    return out.reshape(-1, *image.shape[2:])


def volume_to_25d(image: np.ndarray) -> np.ndarray:
    """(D, 4, S, S) -> (D, 12, S, S) float32 for whole-volume inference."""
    img = np.asarray(image, dtype=np.float32)
    d = img.shape[0]
    pad = np.zeros((d + 2,) + img.shape[1:], dtype=np.float32)
    pad[1:-1] = img
    return np.concatenate([pad[0:d], pad[1:d + 1], pad[2:d + 2]], axis=1)


def n_empty_slices(n_tumour: int, rho: float) -> int:
    """n_e = floor(|D_t| * rho / (1 - rho)), so that tumour-free slices make up
    (at most) a share rho of the balanced corpus. With the paper's 81,191
    tumour-present slices and rho = 0.25 this gives 27,063."""
    return int(math.floor(n_tumour * rho / (1 - rho) + 1e-9))


def balanced_index(data_dir, subject_ids, rho: float, seed: int, log=print):
    """Balanced slice sampling (paper Eq. 2): keep every tumour-present slice
    and draw n_e tumour-free slices at random (fixed seed, drawn once before
    training). Returns a list of (subject_id, slice_index)."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    tumour, empty = [], []
    for sid in subject_ids:
        lab = load_label(data_dir, sid)
        has = np.asarray(lab).reshape(lab.shape[0], -1).max(1) > 0
        for i, h in enumerate(has):
            (tumour if h else empty).append((sid, i))
    n_e = min(len(empty), n_empty_slices(len(tumour), rho))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(empty), size=n_e, replace=False) if n_e else np.array([], dtype=int)
    sampled = [empty[k] for k in sorted(pick.tolist())]
    index = tumour + sampled
    log(f"balanced sampling: {len(tumour)} tumour-present + {len(sampled)} of {len(empty)} tumour-free slices "
        f"-> {len(index)} slices (tumour-free share {len(sampled) / max(len(index), 1):.3f}, rho={rho})")
    return index


class SliceDataset(Dataset):
    """Returns x (12, S, S) float32 and y (n_classes, S, S) float32 with
    channels ordered (WT, TC, ET); n_classes=1 gives the single-task WT target."""

    def __init__(self, data_dir, index, n_classes=3, augment=None):
        self.data_dir, self.index, self.n_classes, self.augment = data_dir, list(index), n_classes, augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        sid, i = self.index[k]
        # memory-mapped and opened per item: only the three needed slices are
        # read, and no file handles are kept open (safe with many subjects)
        img, lab = load_image(self.data_dir, sid), load_label(self.data_dir, sid)
        x = torch.from_numpy(stack_25d(img, i))
        y = torch.from_numpy(regions_from_label(np.asarray(lab[i]))[:self.n_classes].astype(np.float32))
        del img, lab
        if self.augment is not None:
            x, y = self.augment(x, y)
        return x, y
