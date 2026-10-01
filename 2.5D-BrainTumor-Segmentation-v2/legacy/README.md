# Legacy: single-task WT prototype (v1)

These files are the first version of this repository, kept unchanged for reference. The single-task WT analysis in Section 4.3 of the paper (Table A1, Figures 4-6 and 8) comes from this prototype. Everything else in the paper uses the joint WT/TC/ET model. The prototype is superseded by the package in `bt25d/` and the scripts in `scripts/`.

| File | Content |
|---|---|
| `single_task_v1.ipynb` | training and evaluation notebook of the prototype (formerly `2.5D Brain Tumor.ipynb`) |
| `preprocess_v1.py` | early preprocessing script (formerly `data/preprocess.py`) |
| `best_v1_single_task_wt.pth` | weights of the single-task WT model (formerly `best.pth`), decision threshold 0.6 |

The prototype differs from the current code in several ways:

* It has one output channel (whole tumour).
* It uses its own data files and data split.
* It computes validation Dice over the sampled validation slices, not over complete volumes.

The network has the same layer names as `bt25d/model.py`, so the weights load directly:

```bash
python scripts/check_checkpoint.py legacy/best_v1_single_task_wt.pth
```

```python
from bt25d.model import DecoderFree25D, load_checkpoint
model = DecoderFree25D(in_channels=12, n_classes=1)
load_checkpoint(model, "legacy/best_v1_single_task_wt.pth")
```

These weights were trained on a different subject split from the one in `splits/`. The subjects in `splits/val.csv` and `splits/test.csv` may therefore include subjects this model was trained on, so do not use them to evaluate it.
