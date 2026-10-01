#!/usr/bin/env python
"""Combine the held-out test evaluations of several training seeds (paper Table 5).

Reads summary.json from each evaluation folder (scripts/evaluate.py) and
writes a table with one column per seed and the mean +/- SD across seeds
(sample SD, ddof = 1). Summary statistics are computed from the unrounded
per-seed values and rounded only for display.

--aggregate chooses how each seed's Dice / sensitivity / specificity /
precision is formed: 'pooled' (all voxels of the split, the convention of the
validation table) or 'per_case' (mean over subjects). HD95 is always the
per-case mean in millimetres.

Example:
    python scripts/summarize_seeds.py results/test/seed2026 results/test/seed2027 results/test/seed2028 \
        --out results/test/table5
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

METRICS = [("dice", "Dice", 4), ("hd95", "HD95 (mm)", 2), ("sensitivity", "Sensitivity", 4),
           ("specificity", "Specificity", 4), ("precision", "Precision", 4)]


def value(summary, region, metric, aggregate):
    s = summary["regions"][region]
    if metric == "hd95" or aggregate == "per_case":
        return s["per_case"][metric]["mean"]
    return s["pooled"][metric]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dirs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None, help="column labels (default: folder names)")
    ap.add_argument("--aggregate", choices=["pooled", "per_case"], default="pooled")
    ap.add_argument("--out", default="results/seed_summary", help="output prefix (.md, .csv, .json)")
    a = ap.parse_args()

    sums = []
    for d in a.eval_dirs:
        with open(os.path.join(d, "summary.json")) as f:
            sums.append(json.load(f))
    labels = a.labels or [os.path.basename(os.path.normpath(d)) for d in a.eval_dirs]
    if len(labels) != len(sums):
        sys.exit("--labels must match the number of folders")
    regions = list(sums[0]["regions"])
    splits = {s["split"] for s in sums}; ns = {s["n_subjects"] for s in sums}
    if len(splits) > 1 or len(ns) > 1:
        print(f"[warn] evaluations differ in split/size: {splits} {ns}")

    rows = []
    for r in regions:
        for m, name, nd in METRICS:
            vals = [value(s, r, m, a.aggregate) for s in sums]
            rows.append((f"{r} {name}", vals, nd))
    macro = [float(np.mean([value(s, r, "dice", a.aggregate) for r in regions])) for s in sums]
    rows.append(("Macro-Dice", macro, 4))

    table = []
    for name, vals, nd in rows:
        v = np.asarray(vals, dtype=float)
        mu = float(np.nanmean(v)); sd = float(np.nanstd(v, ddof=1)) if np.sum(~np.isnan(v)) > 1 else 0.0
        table.append({"metric": name, "values": [float(x) for x in v], "mean": mu, "sd": sd, "decimals": nd})

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump({"labels": labels, "aggregate": a.aggregate, "split": sorted(splits), "n_subjects": sorted(ns),
                   "rows": table}, f, indent=2)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + labels + ["mean", "sd"])
        for t in table:
            w.writerow([t["metric"]] + [f"{x:.6f}" for x in t["values"]] + [f"{t['mean']:.6f}", f"{t['sd']:.6f}"])
    md = [f"Split: {', '.join(sorted(splits))} (N = {', '.join(map(str, sorted(ns)))}); "
          f"Dice/Sens/Spec/Prec aggregation: {a.aggregate}; HD95: per-case mean", "",
          "| Metric | " + " | ".join(labels) + " | Mean ± SD |", "|---" * (len(labels) + 2) + "|"]
    for t in table:
        nd = t["decimals"]
        md.append(f"| {t['metric']} | " + " | ".join(f"{x:.{nd}f}" for x in t["values"])
                  + f" | {t['mean']:.{nd}f} ± {t['sd']:.{nd}f} |")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten to {a.out}.md/.csv/.json")


if __name__ == "__main__":
    main()
