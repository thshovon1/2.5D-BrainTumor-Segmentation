#!/usr/bin/env python
"""Evaluate a trained model on complete volumes of a split (default: the held-out test set).

Per case and region (WT, TC, ET): Dice, HD95 (mm), sensitivity, specificity,
precision and the voxel counts. The summary reports, for every region, both
the pooled value (TP/FP/FN/TN summed over all voxels of the split, as used for
the validation table) and the per-case mean +/- SD.

The decision threshold is the one stored in the checkpoint (selected on the
validation split by scripts/train.py); it is never tuned on the test set.

Conventions: see bt25d/metrics.py and the README (empty masks, HD95).

Outputs in --out-dir: per_case.csv, summary.json, summary.csv and, with
--save-preds, one compressed binary prediction per subject in preds/
(used by scripts/lcc_analysis.py).

Examples:
    python scripts/evaluate.py --checkpoint runs/final/seed2026/best.pth --split test --out-dir results/test/seed2026
    python scripts/evaluate.py --checkpoint runs/proposed/best.pth --split val --save-preds --out-dir results/val
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d import REGIONS  # noqa: E402
from bt25d.config import load_config  # noqa: E402
from bt25d.data import load_meta, read_split  # noqa: E402
from bt25d.evaluation import binarize, evaluate_case, grid_spacing, probabilities, reference_masks, summarize_cases  # noqa: E402
from bt25d.model import build_model, infer_model_config, read_checkpoint  # noqa: E402
from bt25d.utils import get_device, save_json  # noqa: E402

METRICS = ("dice", "hd95", "sensitivity", "specificity", "precision")


def fmt(v, nd=6):
    return "nan" if isinstance(v, float) and math.isnan(v) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", help="split name in --splits-dir (test, val, ...)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None, help="only needed for bare state_dict checkpoints")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--threshold", type=float, nargs="+", default=None,
                    help="override the stored threshold (one value, or one per region)")
    ap.add_argument("--grid", choices=["model", "native"], default="model",
                    help="model: 160x160 grid at 1.5 mm in-plane (as reported in the paper); "
                         "native: predictions mapped back to the original 240x240 grid")
    ap.add_argument("--hd95-empty-penalty", type=float, default=None,
                    help="HD95 when exactly one of prediction/reference is empty "
                         "(default: undefined, excluded and counted; BraTS used 373.13)")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-slices", type=int, default=64)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-subjects", type=int, default=None, help="debugging only")
    a = ap.parse_args()

    state, meta = read_checkpoint(a.checkpoint)
    cfg = meta.get("config") or load_config(a.config)
    if a.data_dir: cfg["data"]["data_dir"] = a.data_dir
    if a.splits_dir: cfg["data"]["splits_dir"] = a.splits_dir
    mcfg = dict(cfg["model"]); mcfg.update(infer_model_config(state))
    model = build_model(mcfg)
    model.load_state_dict(state, strict=True)
    device = get_device(a.device)
    model.to(device).eval()
    n_classes = mcfg["n_classes"]
    regions = REGIONS[:n_classes]

    thr = a.threshold if a.threshold is not None else meta.get("threshold")
    if thr is None:
        sys.exit("no threshold stored in the checkpoint - pass --threshold (chosen on the validation split)")
    thr = [float(thr)] * n_classes if np.isscalar(thr) else [float(t) for t in thr]
    if len(thr) == 1:
        thr = thr * n_classes
    data_dir = cfg["data"]["data_dir"]
    ids = read_split(cfg["data"]["splits_dir"], a.split)
    if a.max_subjects:
        ids = ids[:a.max_subjects]
    print(f"{a.checkpoint}: {mcfg['arch']} ({n_classes} output channel(s)), threshold {thr}, "
          f"{len(ids)} subjects in '{a.split}', grid {a.grid}")

    os.makedirs(a.out_dir, exist_ok=True)
    if a.save_preds:
        os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
    rows, t_inf = [], 0.0
    for k, sid in enumerate(ids, 1):
        t0 = time.time()
        probs = probabilities(model, data_dir, sid, device, a.grid, a.batch_slices, not a.no_amp)
        t_inf += time.time() - t0
        pred = binarize(probs, thr)
        gt = reference_masks(data_dir, sid, n_classes, a.grid)
        spacing = grid_spacing(load_meta(data_dir, sid), a.grid)
        row = evaluate_case(pred, gt, regions, spacing, a.hd95_empty_penalty)
        row["subject_id"] = sid
        rows.append(row)
        if a.save_preds:
            np.savez_compressed(os.path.join(a.out_dir, "preds", f"{sid}.npz"), pred=pred.astype(np.uint8),
                                regions=np.array(regions), grid=a.grid, spacing=np.array(spacing))
        if k % 25 == 0 or k == len(ids):
            print(f"  {k}/{len(ids)}")

    with open(os.path.join(a.out_dir, "per_case.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = [f"{r}_{m}" for r in regions for m in METRICS + ("gt_voxels", "pred_voxels", "tp", "fp", "fn", "tn")]
        w.writerow(["subject_id"] + cols)
        for row in rows:
            w.writerow([row["subject_id"]] + [fmt(row[c.split("_", 1)[0]][c.split("_", 1)[1]]) for c in cols])

    summary = summarize_cases(rows, regions)
    summary.update({"checkpoint": os.path.abspath(a.checkpoint), "split": a.split, "n_subjects": len(ids),
                    "grid": a.grid, "threshold": thr, "hd95_empty_penalty": a.hd95_empty_penalty,
                    "arch": mcfg["arch"], "epoch": meta.get("epoch"),
                    "inference_s_per_volume": t_inf / max(len(ids), 1)})
    save_json(summary, os.path.join(a.out_dir, "summary.json"))
    with open(os.path.join(a.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "metric", "pooled", "per_case_mean", "per_case_sd", "n", "n_undefined"])
        for r in regions:
            s = summary["regions"][r]
            for m in METRICS:
                pc = s["per_case"][m]
                w.writerow([r, m, fmt(s["pooled"][m]) if m in s["pooled"] else "", fmt(pc["mean"]), fmt(pc["sd"]),
                            pc["n"], pc["n_undefined"]])

    print(f"\n{'region':<7}{'Dice(pooled)':>13}{'Dice(case)':>12}{'HD95 mm':>9}{'Sens':>8}{'Spec':>8}{'Prec':>8}")
    for r in regions:
        s = summary["regions"][r]
        pc = s["per_case"]
        print(f"{r:<7}{s['pooled']['dice']:>13.4f}{pc['dice']['mean']:>12.4f}{pc['hd95']['mean']:>9.2f}"
              f"{s['pooled']['sensitivity']:>8.4f}{s['pooled']['specificity']:>8.4f}{s['pooled']['precision']:>8.4f}")
        if pc["hd95"]["n_undefined"]:
            print(f"       HD95 undefined (one mask empty) in {pc['hd95']['n_undefined']} case(s), excluded from the mean")
    mc = summary["macro"]
    print(f"Macro-Dice: pooled {mc['pooled_dice']:.4f} | per-case {mc['per_case_dice']:.4f} | "
          f"mean HD95 {mc['per_case_hd95']:.2f} mm")
    print(f"written to {a.out_dir}/")


if __name__ == "__main__":
    main()
