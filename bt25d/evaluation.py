"""Whole-volume validation and evaluation helpers shared by the scripts.

All metrics are computed on complete, reassembled volumes (paper Section 4.1).
By default the evaluation grid is the model grid (160 x 160 in-plane, 1.5 mm
pixels after resizing, 1 mm slice spacing), which is how the paper reports
HD95. ``grid="native"`` maps probabilities back to the original 240 x 240
grid first (bilinear) and compares with the original labels.
"""

import math
import numpy as np
import torch

from . import REGIONS
from .data import load_image, load_label, load_meta, load_native_label
from .geometry import to_native
from .inference import predict_volume
from .labels import regions_from_label
from .metrics import case_metrics, dice_from_counts, mean_sd, pooled


def reference_masks(data_dir, sid, n_classes, grid="model"):
    lab = np.asarray(load_label(data_dir, sid)) if grid == "model" else load_native_label(data_dir, sid)
    return regions_from_label(lab)[:n_classes].astype(bool)


def grid_spacing(meta, grid="model"):
    if grid == "model":
        return tuple(meta["model_spacing_dhw"])
    return tuple(meta["spacing_dhw"])


def probabilities(model, data_dir, sid, device, grid="model", batch_slices=64, amp=True):
    """Sigmoid probabilities of the main head on the requested grid, (C, D, H, W)."""
    probs = predict_volume(model, load_image(data_dir, sid), device, batch_slices, amp)
    if grid == "native":
        meta = load_meta(data_dir, sid)
        probs = to_native(probs, meta["native_shape_hwd"][:2], meta["mode"])
    return probs


def binarize(probs, threshold):
    t = np.asarray(threshold, dtype=np.float32).reshape(-1, 1, 1, 1)
    return probs > t


@torch.no_grad()
def validate(model, data_dir, subject_ids, thresholds, device, n_classes=3, batch_slices=64, amp=True,
             per_region_threshold=False):
    """Pooled (all voxels of the split) Dice per region for each candidate
    threshold. Returns a dict with the Dice table, the Macro-Dice per
    threshold and the selected threshold(s)."""
    was_training = model.training
    counts = np.zeros((len(thresholds), n_classes, 3), dtype=np.int64)  # TP, FP, FN
    for sid in subject_ids:
        probs = predict_volume(model, load_image(data_dir, sid), device, batch_slices, amp)
        gt = reference_masks(data_dir, sid, n_classes)
        for k, t in enumerate(thresholds):
            pred = probs > t
            counts[k, :, 0] += (pred & gt).sum(axis=(1, 2, 3))
            counts[k, :, 1] += (pred & ~gt).sum(axis=(1, 2, 3))
            counts[k, :, 2] += (~pred & gt).sum(axis=(1, 2, 3))
    model.train(was_training)
    dice = np.array([[dice_from_counts(*counts[k, c]) for c in range(n_classes)] for k in range(len(thresholds))])
    macro = dice.mean(1)
    if per_region_threshold:
        idx = dice.argmax(0)
        thr = [float(thresholds[i]) for i in idx]
        best = [float(dice[i, c]) for c, i in enumerate(idx)]
    else:
        i = int(macro.argmax())
        thr = [float(thresholds[i])] * n_classes
        best = [float(x) for x in dice[i]]
    return {"thresholds": [float(t) for t in thresholds], "dice": dice.tolist(), "macro": macro.tolist(),
            "threshold": thr, "region_dice": dict(zip(REGIONS, best)), "macro_dice": float(np.mean(best))}


def summarize_cases(rows, regions):
    """rows: list of {region: case_metrics dict}. Returns the summary dict
    with pooled metrics and per-case mean / SD for every region."""
    out = {}
    for r in regions:
        rr = [row[r] for row in rows]
        pooled_m = pooled(rr)
        s = {"pooled": pooled_m, "per_case": {}}
        for m in ("dice", "hd95", "sensitivity", "specificity", "precision"):
            vals = [x[m] for x in rr]
            mu, sd, n = mean_sd(vals)
            s["per_case"][m] = {"mean": mu, "sd": sd, "n": n,
                                "n_undefined": int(sum(1 for v in vals if isinstance(v, float) and math.isnan(v)))}
        s["n_cases"] = len(rr)
        s["n_empty_reference"] = int(sum(1 for x in rr if x["gt_voxels"] == 0))
        out[r] = s
    macro = {
        "pooled_dice": float(np.mean([out[r]["pooled"]["dice"] for r in regions])),
        "per_case_dice": float(np.mean([out[r]["per_case"]["dice"]["mean"] for r in regions])),
        "per_case_hd95": float(np.mean([out[r]["per_case"]["hd95"]["mean"] for r in regions])),
    }
    return {"regions": out, "macro": macro}


def evaluate_case(pred, gt, regions, spacing, empty_penalty=None):
    return {r: case_metrics(pred[c], gt[c], spacing, empty_penalty) for c, r in enumerate(regions)}
