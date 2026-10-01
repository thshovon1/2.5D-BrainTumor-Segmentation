"""In-plane geometry: native grid (e.g. 240 x 240) <-> model grid (e.g. 160 x 160).

Two modes are supported and recorded in each subject's ``meta.json``:

* ``resize`` - bilinear (images) / nearest (labels) in-plane resampling.
  With BraTS 240 x 240 slices and a 160 x 160 model grid, one model pixel
  covers 1.5 mm in-plane.
* ``crop``   - centre crop (or zero pad) to the model grid; pixel size is
  unchanged (1 mm), but anatomy outside the central window is not seen.

By default, evaluation runs on the model grid with the model-grid voxel
spacing (1 x 1.5 x 1.5 mm for ``resize``), as reported in the paper.
``scripts/evaluate.py --grid native`` instead maps probabilities back to the
original grid with ``to_native`` and compares them with the original labels.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _crop_pad_offsets(n: int, size: int):
    """Offsets used by centre crop / pad along one axis.

    Returns (src_start, dst_start, length): native[src_start:src_start+length]
    maps to model[dst_start:dst_start+length].
    """
    if n >= size:
        return (n - size) // 2, 0, size
    return 0, (size - n) // 2, n


def image_to_model_grid(vol: np.ndarray, size: int, mode: str) -> np.ndarray:
    """vol: (C, D, H, W) float -> (C, D, size, size) float32."""
    c, d, h, w = vol.shape
    if mode == "resize":
        t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).reshape(c * d, 1, h, w)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False,
                          antialias=(h > size or w > size))
        return t.reshape(c, d, size, size).numpy()
    if mode == "crop":
        out = np.zeros((c, d, size, size), dtype=np.float32)
        sh, dh, lh = _crop_pad_offsets(h, size)
        sw, dw, lw = _crop_pad_offsets(w, size)
        out[:, :, dh:dh + lh, dw:dw + lw] = vol[:, :, sh:sh + lh, sw:sw + lw]
        return out
    raise ValueError(f"unknown mode {mode!r}")


def label_to_model_grid(lab: np.ndarray, size: int, mode: str) -> np.ndarray:
    """lab: (D, H, W) integer labels -> (D, size, size) uint8 (nearest neighbour)."""
    d, h, w = lab.shape
    if mode == "resize":
        t = torch.from_numpy(np.ascontiguousarray(lab, dtype=np.float32)).reshape(d, 1, h, w)
        t = F.interpolate(t, size=(size, size), mode="nearest-exact")
        return t.reshape(d, size, size).round().to(torch.uint8).numpy()
    if mode == "crop":
        out = np.zeros((d, size, size), dtype=np.uint8)
        sh, dh, lh = _crop_pad_offsets(h, size)
        sw, dw, lw = _crop_pad_offsets(w, size)
        out[:, dh:dh + lh, dw:dw + lw] = lab[:, sh:sh + lh, sw:sw + lw]
        return out
    raise ValueError(f"unknown mode {mode!r}")


def to_native(probs: np.ndarray, native_hw, mode: str) -> np.ndarray:
    """Map model-grid probabilities (C, D, S, S) back to the native grid (C, D, H, W).

    ``resize``: bilinear upsampling of probabilities (thresholding happens after).
    ``crop``  : the prediction is placed back into the centre; outside the
                model field of view the prediction is background (0).
    """
    c, d, s, _ = probs.shape
    h, w = int(native_hw[0]), int(native_hw[1])
    if mode == "resize":
        t = torch.from_numpy(np.ascontiguousarray(probs, dtype=np.float32)).reshape(c * d, 1, s, s)
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        return t.reshape(c, d, h, w).numpy()
    if mode == "crop":
        out = np.zeros((c, d, h, w), dtype=np.float32)
        sh, dh, lh = _crop_pad_offsets(h, s)
        sw, dw, lw = _crop_pad_offsets(w, s)
        out[:, :, sh:sh + lh, sw:sw + lw] = probs[:, :, dh:dh + lh, dw:dw + lw]
        return out
    raise ValueError(f"unknown mode {mode!r}")
