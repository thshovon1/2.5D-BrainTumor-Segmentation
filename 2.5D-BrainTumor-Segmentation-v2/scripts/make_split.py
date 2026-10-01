#!/usr/bin/env python
"""Subject-level train / validation / held-out test split (paper Section 3.3).

1. The sorted subject list is shuffled with a fixed seed; ``--n-val`` subjects
   form the development validation split and the rest the development
   training split (1,001 / 250 for BraTS 2021's 1,251 subjects).
2. ``--n-test`` subjects are then drawn from the development training split
   (second fixed seed) as the held-out test set, leaving the final training
   split (851).

Ablations train on ``dev_train`` and select on ``val``; final models train on
``train`` and are evaluated once on ``test``. Pairwise overlaps are checked
and the script stops if any subject appears in two final partitions.

Writes <splits-dir>/{dev_train,train,val,test}.csv and subject_splits.csv.
"""

import argparse
import csv
import os
import sys

import numpy as np


def write(path, ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id"]); w.writerows([[s] for s in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed", help="preprocessed data (one folder per subject)")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2021, help="seed for the train/validation split")
    ap.add_argument("--test-seed", type=int, default=2022, help="seed for drawing the held-out test set")
    ap.add_argument("--expect", type=int, default=1251, help="expected number of subjects (BraTS 2021 training set)")
    a = ap.parse_args()

    ids = sorted(d for d in os.listdir(a.data_dir) if os.path.exists(os.path.join(a.data_dir, d, "meta.json")))
    if len(ids) < a.n_val + a.n_test + 1:
        sys.exit(f"only {len(ids)} subjects in {a.data_dir}")
    if len(ids) != a.expect:
        print(f"[warn] {len(ids)} subjects found, expected {a.expect}: the split depends on the full subject "
              f"list, so it will differ from a split made on all {a.expect} subjects")
    perm = np.random.default_rng(a.seed).permutation(len(ids))
    val = sorted(ids[k] for k in perm[:a.n_val])
    dev_train = sorted(ids[k] for k in perm[a.n_val:])
    pick = np.random.default_rng(a.test_seed).choice(len(dev_train), size=a.n_test, replace=False)
    test = sorted(dev_train[k] for k in pick)
    train = sorted(set(dev_train) - set(test))

    S = {"train": set(train), "val": set(val), "test": set(test)}
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(S[x] & S[y])
        print(f"overlap {x} & {y}: {n}")
        if n:
            sys.exit("overlap detected - aborting")
    assert len(train) + len(val) + len(test) == len(ids)

    os.makedirs(a.splits_dir, exist_ok=True)
    for name, lst in (("dev_train", dev_train), ("train", train), ("val", val), ("test", test)):
        write(os.path.join(a.splits_dir, f"{name}.csv"), lst)
    with open(os.path.join(a.splits_dir, "subject_splits.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject_id", "split"])
        for name in ("train", "val", "test"):
            w.writerows([[s, name] for s in sorted(S[name])])
    print(f"subjects: {len(ids)} | dev_train {len(dev_train)} | train {len(train)} | val {len(val)} | test {len(test)}")
    print(f"written to {a.splits_dir}/")


if __name__ == "__main__":
    main()
