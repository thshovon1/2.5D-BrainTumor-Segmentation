"""YAML configuration with single inheritance (``base:``) and dotted overrides.

Example:
    cfg = load_config("configs/proposed.yaml", ["training.seed=2027", "training.rho=0.4"])
"""

import copy
import os
import re
import yaml

# PyYAML (YAML 1.1) reads "3e-4" as a string; accept it as a number.
_FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def _coerce(v):
    if isinstance(v, str) and _FLOAT.match(v.strip()):
        return float(v)
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    return v


DEFAULTS = {
    "data": {
        "data_dir": "data/processed",
        "splits_dir": "splits",
        "train_split": "dev_train",   # 'dev_train' (1,001) for development runs, 'train' (851) for final runs
        "val_split": "val",           # 250 subjects: checkpoint and threshold selection
    },
    "model": {"arch": "decoder_free", "in_channels": 12, "n_classes": 3,
              "widths": [32, 64, 128], "aspp_rates": [6, 12]},
    "training": {
        "seed": 2026,
        "epochs": 10,
        "batch_size": 32,
        "lr": 3.0e-4,
        "eta_min": 0.0,               # cosine annealing floor
        "scheduler_step": "epoch",    # cosine annealing stepped once per epoch ('iteration' also possible)
        "amp": True,                  # FP16 mixed precision (CUDA only)
        "num_workers": 4,
        "rho": 0.25,                  # tumour-free share of the balanced training corpus
        "sampling_seed": 2021,        # the balanced slice index is drawn once with this seed
        "augment": False,
        "deterministic": False,
        "log_every": 200,
        "max_steps_per_epoch": None,  # debugging only
    },
    "loss": {"w_dice": 0.5, "w_bce": 0.3, "w_focal": 0.2, "pos_weight": 2.0,
             "focal_alpha": 0.8, "focal_gamma": 2.0, "head_weights": [1.0, 0.3, 0.2]},
    "validation": {
        "thresholds": [0.3, 0.4, 0.5, 0.6],
        "per_region_threshold": False,  # one global threshold chosen by validation Macro-Dice
        "batch_slices": 64,
        "max_subjects": None,           # debugging only
    },
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        chain, p, seen = [], path, set()
        while p:
            p = os.path.normpath(p)
            if p in seen:
                raise ValueError(f"circular 'base:' reference at {p}")
            seen.add(p)
            with open(p) as f:
                c = yaml.safe_load(f) or {}
            chain.append(c)
            base = c.pop("base", None)
            p = os.path.join(os.path.dirname(p), base) if base else None
        for c in reversed(chain):
            cfg = _merge(cfg, c)
    for ov in overrides or ():
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {ov!r}")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return _coerce(cfg)


def dump_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
