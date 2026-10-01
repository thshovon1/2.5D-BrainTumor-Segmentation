"""Synthetic BraTS-like subjects for tests (no real data needed).

Each subject has four co-registered sequences and a label map with a nested
tumour: edema (2) around a core (1) with an enhancing rim/centre (4). Some
subjects are multi-focal (two separate lesions); one has no tumour.
"""

import os

import nibabel as nib
import numpy as np


def _ellipsoid(shape, centre, radii):
    z = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), 0).astype(np.float32)
    d = sum(((z[k] - centre[k]) / radii[k]) ** 2 for k in range(3))
    return d <= 1.0


def make_subject(root, sid, shape=(48, 48, 24), seed=0, n_lesions=1):
    rng = np.random.default_rng(seed)
    h, w, d = shape
    brain = _ellipsoid(shape, (h / 2, w / 2, d / 2), (h * 0.42, w * 0.38, d * 0.45))
    seg = np.zeros(shape, np.uint8)
    for _ in range(n_lesions):
        c = (rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w, rng.uniform(0.35, 0.65) * d)
        r = rng.uniform(4, 7)
        wt = _ellipsoid(shape, c, (r, r, r * 0.7)) & brain
        tc = _ellipsoid(shape, c, (r * 0.6, r * 0.6, r * 0.45)) & brain
        et = _ellipsoid(shape, c, (r * 0.35, r * 0.35, r * 0.3)) & brain
        seg[wt & (seg == 0)] = 2
        seg[tc] = 1
        seg[et] = 4
    base = {"t1": 1.0, "t1ce": 1.1, "t2": 1.3, "flair": 0.9}
    boost = {"t1": {1: -0.4, 2: -0.2, 4: -0.3}, "t1ce": {1: -0.3, 2: 0.0, 4: 1.2},
             "t2": {1: 0.5, 2: 0.9, 4: 0.3}, "flair": {1: 0.4, 2: 1.2, 4: 0.5}}
    dst = os.path.join(root, sid)
    os.makedirs(dst, exist_ok=True)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    for m in ("t1", "t1ce", "t2", "flair"):
        v = np.where(brain, base[m] * 400, 0).astype(np.float32)
        for lab, b in boost[m].items():
            v[seg == lab] += b * 400
        v = np.where(brain, v + rng.normal(0, 30, shape), 0).clip(0).astype(np.float32)
        nib.save(nib.Nifti1Image(v, affine), os.path.join(dst, f"{sid}_{m}.nii.gz"))
    nib.save(nib.Nifti1Image(seg, affine), os.path.join(dst, f"{sid}_seg.nii.gz"))
    return seg


def make_dataset(root, n=12, shape=(48, 48, 24)):
    for k in range(n):
        n_lesions = 0 if k == 0 else (2 if k % 4 == 1 else 1)
        make_subject(root, f"BraTS2021_{k:05d}", shape, seed=k, n_lesions=n_lesions)
    return root
