"""NIfTI loading and intensity normalisation shared by preprocessing and prediction."""

import os

import nibabel as nib
import numpy as np

from . import MODALITIES
from .geometry import image_to_model_grid


def zscore_in_brain(vol: np.ndarray) -> np.ndarray:
    """Z-score a sequence within the brain mask (non-zero voxels); background stays 0."""
    brain = vol > 0
    out = np.zeros_like(vol, dtype=np.float32)
    if brain.any():
        v = vol[brain]
        out[brain] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def find_sequence(subj_dir: str, sid: str, name: str):
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(subj_dir, f"{sid}_{name}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_subject(subj_dir: str):
    """Load the four sequences of a BraTS subject folder.

    Returns (stack (4, D, H, W) float32 z-scored, reference nibabel image,
    zooms (x, y, z)). Raises FileNotFoundError if a sequence is missing."""
    sid = os.path.basename(os.path.normpath(subj_dir))
    vols, ref = [], None
    for m in MODALITIES:
        p = find_sequence(subj_dir, sid, m)
        if p is None:
            raise FileNotFoundError(f"{sid}: missing {m}")
        img = nib.load(p)
        ref = ref or img
        vols.append(zscore_in_brain(np.asarray(img.dataobj, dtype=np.float32)))
    zooms = tuple(float(z) for z in ref.header.get_zooms()[:3])
    # NIfTI (H, W, D) -> (C, D, H, W)
    return np.stack(vols).transpose(0, 3, 1, 2), ref, zooms


def to_model_input(stack: np.ndarray, size: int, mode: str) -> np.ndarray:
    """(4, D, H, W) -> slice-major (D, 4, size, size) float16, as stored on disk."""
    return np.ascontiguousarray(image_to_model_grid(stack, size, mode).transpose(1, 0, 2, 3), dtype=np.float16)
