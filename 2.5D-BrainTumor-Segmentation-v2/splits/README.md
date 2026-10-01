# Subject splits

`scripts/make_split.py` writes the subject lists here:

| File | Subjects | Used for |
|---|---|---|
| `dev_train.csv` | 1,001 | development training (ablations, Table 4) |
| `val.csv` | 250 | checkpoint and threshold selection, ablations, LCC study |
| `test.csv` | 150 | held-out test set (drawn from `dev_train`; evaluated once per final run) |
| `train.csv` | 851 | final training (`dev_train` without `test`) |
| `subject_splits.csv` | 1,251 | every subject with its partition (`train`, `val` or `test`) |

The split is made at the subject level, so every slice of a subject belongs to one partition. The script checks that the training, validation and test sets do not overlap and stops if they do.

```bash
python scripts/make_split.py --data-dir data/processed --splits-dir splits           # seeds 2021 / 2022
```

The lists depend only on the sorted subject IDs and the two seeds, so the same command gives the same split on any machine with the full BraTS 2021 training set.
