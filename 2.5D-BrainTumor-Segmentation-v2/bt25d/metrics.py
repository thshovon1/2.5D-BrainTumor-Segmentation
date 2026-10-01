"""Segmentation metrics (paper Eqs. 12-15).

Conventions (also stated in the README):
* Dice: 1 if prediction and reference are both empty, 0 if exactly one is.
* Sensitivity is undefined (NaN) when the reference is empty; precision is
  undefined when the prediction is empty. NaNs are excluded from per-case
  means and counted in the summary.
* HD95 (Eq. 14): max of the two directed 95th-percentile surface distances,
  in millimetres using the voxel spacing. Both empty -> 0. Exactly one empty
  -> NaN (excluded and counted) or, with ``empty_penalty``, a fixed value
  (the BraTS challenges used 373.13 mm).
* Pooled metrics sum TP/FP/FN/TN over all cases before computing the ratio.
"""

import math
import numpy as np
from scipy import ndimage as ndi

_STRUCT = ndi.generate_binary_structure(3, 1)


def confusion(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def dice_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def _ratio(num, den):
    return num / den if den > 0 else math.nan


def _surface(mask):
    return mask & ~ndi.binary_erosion(mask, structure=_STRUCT, border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), empty_penalty=None, margin: int = 2):
    pred = pred.astype(bool); gt = gt.astype(bool)
    pe, ge = not pred.any(), not gt.any()
    if pe and ge:
        return 0.0
    if pe or ge:
        return math.nan if empty_penalty is None else float(empty_penalty)
    # restrict to the joint bounding box (+margin) for speed; distances inside
    # the box are exact because both surfaces lie inside it
    idx = np.argwhere(pred | gt)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, pred.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = pred[sl], gt[sl]
    sp, sg = _surface(p), _surface(g)
    dt_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    d_pg = dt_g[sp]
    d_gp = dt_p[sg]
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def case_metrics(pred, gt, spacing=(1.0, 1.0, 1.0), empty_penalty=None):
    tp, fp, fn, tn = confusion(pred, gt)
    return dict(dice=dice_from_counts(tp, fp, fn),
                hd95=hd95(pred, gt, spacing, empty_penalty),
                sensitivity=_ratio(tp, tp + fn),
                specificity=_ratio(tn, tn + fp),
                precision=_ratio(tp, tp + fp),
                tp=tp, fp=fp, fn=fn, tn=tn,
                gt_voxels=tp + fn, pred_voxels=tp + fp)


def pooled(rows):
    """rows: iterable of dicts with tp, fp, fn, tn -> pooled metrics."""
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows); tn = sum(r["tn"] for r in rows)
    return dict(dice=dice_from_counts(tp, fp, fn), sensitivity=_ratio(tp, tp + fn),
                specificity=_ratio(tn, tn + fp), precision=_ratio(tp, tp + fp))


def mean_sd(values):
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return math.nan, math.nan, 0
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return float(v.mean()), sd, int(v.size)
