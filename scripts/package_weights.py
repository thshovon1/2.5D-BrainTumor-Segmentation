#!/usr/bin/env python
"""Package trained weights for release in the checkpoint format used by this repository.

Wraps a bare state_dict (for example weights saved from a notebook with
torch.save(model.state_dict(), ...)) together with the metadata that
evaluation and prediction need: the decision threshold selected on the
validation split, the training seed and the configuration. The weights are
checked against bt25d/model.py with strict=True before anything is written.

Example (T = the threshold selected on the validation split for this run):
    python scripts/package_weights.py --weights my_run_seed2026.pth --threshold T --seed 2026 \
        --config configs/final.yaml --out weights/decoder_free_seed2026.pth
"""

import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d import __version__  # noqa: E402
from bt25d.config import load_config  # noqa: E402
from bt25d.model import build_model, count_parameters, infer_model_config, read_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="state_dict or checkpoint file")
    ap.add_argument("--threshold", type=float, nargs="+", required=True,
                    help="threshold selected on the validation split (one value or one per region)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    state, meta = read_checkpoint(a.weights)
    mcfg = infer_model_config(state)
    model = build_model(mcfg)
    model.load_state_dict(state, strict=True)          # fails loudly if the architecture differs
    cfg = load_config(a.config)
    cfg["model"].update(mcfg)
    if a.seed is not None:
        cfg["training"]["seed"] = a.seed
    thr = a.threshold * mcfg["n_classes"] if len(a.threshold) == 1 else a.threshold
    if len(thr) != mcfg["n_classes"]:
        sys.exit(f"expected 1 or {mcfg['n_classes']} thresholds")
    ckpt = {"model": model.state_dict(), "config": cfg, "epoch": a.epoch if a.epoch is not None else meta.get("epoch"),
            "threshold": [float(t) for t in thr], "seed": a.seed, "note": a.note, "version": __version__}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(ckpt, a.out)
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"{a.out}: {mcfg['arch']}, {count_parameters(model):,} parameters, threshold {ckpt['threshold']}, "
          f"seed {a.seed}\nsha256 {sha}")


if __name__ == "__main__":
    main()
