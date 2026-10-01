#!/usr/bin/env python
"""Train the decoder-free 2.5D model (or a variant) from a YAML config.

Balanced slice sampling on the training subjects, hybrid loss with deep
supervision, Adam + cosine annealing, FP16 mixed precision. After every epoch
the model is validated on the complete validation volumes: pooled Dice per
region is computed for each candidate threshold, and the checkpoint with the
highest validation Macro-Dice (and its threshold) is kept as best.pth.

Outputs in --out-dir:
    config.yaml        resolved configuration
    train.log          log
    history.csv        per-epoch loss, learning rate and validation Dice per threshold
    best.pth           {'model', 'config', 'epoch', 'threshold', 'val'}  (selected on validation)
    last.pth           final-epoch weights
    train_summary.json best epoch, validation Macro-Dice / region Dice, threshold, parameters, time

Examples:
    python scripts/train.py --config configs/proposed.yaml --out-dir runs/proposed
    python scripts/train.py --config configs/final.yaml --seed 2027 --out-dir runs/final/seed2027
    python scripts/train.py --config configs/proposed.yaml --set training.rho=0.4 --out-dir runs/rho_040
"""

import argparse
import csv
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d import REGIONS, __version__  # noqa: E402
from bt25d.augment import Augment  # noqa: E402
from bt25d.config import dump_config, load_config  # noqa: E402
from bt25d.data import SliceDataset, balanced_index, read_split  # noqa: E402
from bt25d.evaluation import validate  # noqa: E402
from bt25d.losses import HybridLoss  # noqa: E402
from bt25d.model import build_model, count_parameters  # noqa: E402
from bt25d.utils import Logger, get_device, save_json, seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/proposed.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=None, help="overrides training.seed")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. training.epochs=5 loss.w_focal=0")
    ap.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if available)")
    a = ap.parse_args()

    overrides = list(a.set) + ([f"training.seed={a.seed}"] if a.seed is not None else [])
    cfg = load_config(a.config, overrides)
    D, M, T, L, V = cfg["data"], cfg["model"], cfg["training"], cfg["loss"], cfg["validation"]
    os.makedirs(a.out_dir, exist_ok=True)
    dump_config(cfg, os.path.join(a.out_dir, "config.yaml"))
    log = Logger(os.path.join(a.out_dir, "train.log"))
    seed_everything(T["seed"], T["deterministic"])
    device = get_device(a.device)
    n_classes = M["n_classes"]
    regions = REGIONS[:n_classes]
    log(f"bt25d {__version__} | config {a.config} | seed {T['seed']} | device {device}")

    # ---- data ------------------------------------------------------------
    train_ids = read_split(D["splits_dir"], D["train_split"])
    val_ids = read_split(D["splits_dir"], D["val_split"])
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        sys.exit(f"{len(overlap)} subjects appear in both {D['train_split']} and {D['val_split']} - aborting")
    if V.get("max_subjects"):
        val_ids = val_ids[:V["max_subjects"]]
    log(f"subjects: {D['train_split']} {len(train_ids)} | {D['val_split']} {len(val_ids)}")
    index = balanced_index(D["data_dir"], train_ids, T["rho"], T["sampling_seed"], log)
    ds = SliceDataset(D["data_dir"], index, n_classes, Augment() if T["augment"] else None)
    g = torch.Generator(); g.manual_seed(T["seed"])
    loader = DataLoader(ds, batch_size=T["batch_size"], shuffle=True, num_workers=T["num_workers"],
                        pin_memory=device.type == "cuda", drop_last=False, generator=g,
                        persistent_workers=T["num_workers"] > 0)

    # ---- model / optimisation --------------------------------------------
    model = build_model(M).to(device)
    n_params = count_parameters(model)
    log(f"model {M['arch']} | parameters {n_params:,}")
    criterion = HybridLoss(**L)
    opt = torch.optim.Adam(model.parameters(), lr=T["lr"], betas=(0.9, 0.999))
    steps_per_epoch = len(loader) if not T["max_steps_per_epoch"] else min(len(loader), T["max_steps_per_epoch"])
    per_iter = T["scheduler_step"] == "iteration"
    if T["scheduler_step"] not in ("epoch", "iteration"):
        sys.exit("training.scheduler_step must be 'epoch' or 'iteration'")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=T["epochs"] * (steps_per_epoch if per_iter else 1), eta_min=T["eta_min"])
    use_amp = bool(T["amp"]) and device.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_amp))

    thresholds = V["thresholds"]
    hist_path = os.path.join(a.out_dir, "history.csv")
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "lr", "epoch_time_s", "val_time_s"]
                               + [f"val_macro_t{t}" for t in thresholds]
                               + [f"val_{r}_t{t}" for t in thresholds for r in regions])

    best, best_epoch, t_start = -math.inf, 0, time.time()
    for epoch in range(1, T["epochs"] + 1):
        model.train()
        t0, run_loss, n_seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(loader, 1):
            if step > steps_per_epoch:
                break
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(x)
            loss = criterion(outputs, y)
            if not torch.isfinite(loss):
                sys.exit(f"non-finite loss at epoch {epoch} step {step} - aborting")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if per_iter:
                sched.step()
            run_loss += loss.item() * x.shape[0]; n_seen += x.shape[0]
            if T["log_every"] and step % T["log_every"] == 0:
                log(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {run_loss / n_seen:.4f}")
        train_loss, t_train = run_loss / max(n_seen, 1), time.time() - t0
        lr_used = opt.param_groups[0]["lr"]
        if not per_iter:
            sched.step()

        t1 = time.time()
        val = validate(model, D["data_dir"], val_ids, thresholds, device, n_classes,
                       V["batch_slices"], T["amp"], V["per_region_threshold"])
        t_val = time.time() - t1
        region_txt = " ".join(f"{r} {d:.4f}" for r, d in val["region_dice"].items())
        log(f"epoch {epoch}/{T['epochs']} | loss {train_loss:.4f} | val Macro-Dice {val['macro_dice']:.4f} "
            f"({region_txt}) at t={val['threshold']} | {t_train:.0f}s + val {t_val:.0f}s")
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{lr_used:.3e}", f"{t_train:.1f}",
                                    f"{t_val:.1f}"] + [f"{m:.6f}" for m in val["macro"]]
                                   + [f"{val['dice'][k][c]:.6f}" for k in range(len(thresholds))
                                      for c in range(n_classes)])
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "threshold": val["threshold"],
                "val": val, "seed": T["seed"], "version": __version__}
        torch.save(ckpt, os.path.join(a.out_dir, "last.pth"))
        if val["macro_dice"] > best:
            best, best_epoch = val["macro_dice"], epoch
            torch.save(ckpt, os.path.join(a.out_dir, "best.pth"))
            best_val = val

    summary = {"config": a.config, "seed": T["seed"], "arch": M["arch"], "parameters": n_params,
               "train_split": D["train_split"], "n_train_subjects": len(train_ids), "n_train_slices": len(index),
               "val_split": D["val_split"], "n_val_subjects": len(val_ids),
               "best_epoch": best_epoch, "val_macro_dice": best_val["macro_dice"],
               "val_region_dice": best_val["region_dice"], "threshold": best_val["threshold"],
               "train_time_min": (time.time() - t_start) / 60}
    save_json(summary, os.path.join(a.out_dir, "train_summary.json"))
    log(f"best epoch {best_epoch}: validation Macro-Dice {best_val['macro_dice']:.4f} at t={best_val['threshold']} "
        f"-> {os.path.join(a.out_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
