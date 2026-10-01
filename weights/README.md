# Trained weights

The final model was trained three times on the 851 training subjects, with seeds 2026, 2027 and 2028 (`configs/final.yaml`). The paper's held-out test results are the mean ± SD over these three runs.

| File | Content |
|---|---|
| `decoder_free_seed2026.pth` | final run, seed 2026 |
| `decoder_free_seed2027.pth` | final run, seed 2027 |
| `decoder_free_seed2028.pth` | final run, seed 2028 |

Each file is a checkpoint dictionary with these keys:

* `model`: the state_dict
* `threshold`: the decision threshold per region (WT, TC, ET), selected on the validation split
* `config`: the training configuration
* `epoch`, `seed`: the selected epoch and the training seed

Evaluate the weights on the subjects in `splits/test.csv`. These are the held-out subjects the weights were never trained on. A split regenerated elsewhere may differ.

Check a file against the model code, with parameter and FLOP counts:

```bash
python scripts/check_checkpoint.py weights/*.pth
```

Evaluate the three runs on the held-out test split and summarise them as in Table 5:

```bash
for s in 2026 2027 2028; do
  python scripts/evaluate.py --checkpoint weights/decoder_free_seed$s.pth --split test --out-dir results/test/seed$s
done
python scripts/summarize_seeds.py results/test/seed2026 results/test/seed2027 results/test/seed2028 \
    --labels "Seed 2026" "Seed 2027" "Seed 2028" --out results/test/table5
```

Weights saved as a bare state_dict can be converted to this format. Replace `T` with the threshold that was selected on the validation split for that run:

```bash
python scripts/package_weights.py --weights my_seed2026.pth --threshold T --seed 2026 \
    --out weights/decoder_free_seed2026.pth
```

The weights of the single-task prototype are in `legacy/`.
