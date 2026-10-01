#!/usr/bin/env python
"""Cohort-level effect of largest-connected-component (LCC) filtering (paper Section 4.5).

For every subject, the raw binary prediction of a region (default WT) is
compared with its LCC-filtered version (only the largest 3D connected
component kept). Per-case Dice is computed for both, and the paired
differences (LCC - raw) are tested with a Wilcoxon signed-rank test. Cases
are stratified by the number of connected components in the reference mask
(uni-focal: 1; multi-focal: >= 2).

Input: the prediction folder written by
    python scripts/evaluate.py --checkpoint runs/proposed/best.pth --split val --save-preds --out-dir results/val

Outputs in --out-dir: lcc_per_case.csv and lcc_summary.json (Tables 12 and 13).

Example:
    python scripts/lcc_analysis.py --pred-dir results/val/preds --out-dir results/val_lcc
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d.config import load_config  # noqa: E402
from bt25d.evaluation import reference_masks  # noqa: E402
from bt25d.metrics import confusion, dice_from_counts  # noqa: E402
from bt25d.utils import save_json  # noqa: E402


def structure(connectivity):
    return ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])


def largest_component(mask, struct):
    lab, n = ndi.label(mask, structure=struct)
    if n <= 1:
        return mask.astype(bool), n
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (int(sizes.argmax()) + 1), n


def dice(p, g):
    tp, fp, fn, _ = confusion(p, g)
    return dice_from_counts(tp, fp, fn)


def mean_sd(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, help="preds/ folder written by evaluate.py --save-preds")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=None, help="default: data.data_dir of --config")
    ap.add_argument("--config", default="configs/proposed.yaml")
    ap.add_argument("--region", default="WT", choices=["WT", "TC", "ET"])
    ap.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26],
                    help="3D connectivity for LCC and for counting reference components")
    ap.add_argument("--zero-method", default="wilcox", choices=["wilcox", "pratt", "zsplit"],
                    help="treatment of zero differences in the Wilcoxon test (scipy)")
    a = ap.parse_args()

    data_dir = a.data_dir or load_config(a.config)["data"]["data_dir"]
    files = sorted(glob.glob(os.path.join(a.pred_dir, "*.npz")))
    if not files:
        sys.exit(f"no predictions in {a.pred_dir} (run evaluate.py with --save-preds)")
    struct = structure(a.connectivity)
    rows = []
    for k, fpath in enumerate(files, 1):
        sid = os.path.splitext(os.path.basename(fpath))[0]
        with np.load(fpath) as z:
            regions = [str(r) for r in z["regions"]]
            grid = str(z["grid"])
            if a.region not in regions:
                sys.exit(f"{a.region} not among predicted regions {regions}")
            pred = z["pred"][regions.index(a.region)].astype(bool)
        gt = reference_masks(data_dir, sid, len(regions), grid)[regions.index(a.region)]
        lcc, n_pred = largest_component(pred, struct)
        _, n_gt = ndi.label(gt, structure=struct)
        d_raw, d_lcc = dice(pred, gt), dice(lcc, gt)
        rows.append({"subject_id": sid, "gt_components": int(n_gt), "pred_components": int(n_pred),
                     "dice_raw": d_raw, "dice_lcc": d_lcc, "delta": d_lcc - d_raw})
        if k % 50 == 0 or k == len(files):
            print(f"  {k}/{len(files)}")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "lcc_per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in r.items()})

    raw = np.array([r["dice_raw"] for r in rows]); lcc = np.array([r["dice_lcc"] for r in rows])
    delta = lcc - raw
    n = len(rows)
    if np.any(delta != 0):
        test = stats.wilcoxon(lcc, raw, zero_method=a.zero_method)
        p = float(test.pvalue); stat = float(test.statistic)
    else:
        p, stat = 1.0, float("nan")
    uni = [r for r in rows if r["gt_components"] <= 1]
    multi = [r for r in rows if r["gt_components"] >= 2]

    def group(g):
        if not g:
            return {"n": 0, "share": 0.0, "mean_delta": float("nan")}
        d = np.array([r["delta"] for r in g])
        return {"n": len(g), "share": len(g) / n, "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                "n_changed": int(np.count_nonzero(d))}

    summary = {
        "region": a.region, "n": n, "connectivity": a.connectivity, "zero_method": a.zero_method,
        "raw_dice_mean_sd": mean_sd(raw), "lcc_dice_mean_sd": mean_sd(lcc),
        "delta_mean_sd": mean_sd(delta), "delta_median": float(np.median(delta)),
        "n_changed": int(np.count_nonzero(delta)), "wilcoxon_statistic": stat, "wilcoxon_p": p,
        "uni_focal": group(uni), "multi_focal": group(multi),
        "note": "reference components counted with the same connectivity; uni-focal = 1 component "
                "(cases with an empty reference are counted as uni-focal)",
    }
    save_json(summary, os.path.join(a.out_dir, "lcc_summary.json"))

    m, s = summary["raw_dice_mean_sd"]; ml, sl = summary["lcc_dice_mean_sd"]; md, sd = summary["delta_mean_sd"]
    print(f"\nLCC effect on {a.region} Dice (N = {n}, {a.connectivity}-connectivity)")
    print(f"  raw prediction      {m:.4f} +/- {s:.4f}")
    print(f"  LCC-filtered        {ml:.4f} +/- {sl:.4f}")
    print(f"  delta (LCC - raw)   {md:+.4f} +/- {sd:.4f}   median {summary['delta_median']:+.4f}   "
          f"changed in {summary['n_changed']} case(s)")
    print(f"  Wilcoxon signed-rank p = {p:.3g}")
    for name, g in (("uni-focal (1 component)", summary["uni_focal"]), ("multi-focal (>=2)", summary["multi_focal"])):
        print(f"  {name:<24} N = {g['n']:>4} ({100 * g['share']:.1f}%)  mean delta {g['mean_delta']:+.4f}")
    print(f"written to {a.out_dir}/")


if __name__ == "__main__":
    main()
