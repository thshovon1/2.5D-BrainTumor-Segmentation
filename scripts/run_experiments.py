#!/usr/bin/env python
"""Run (or collect) the experiment groups of the paper.

Groups
    proposed     proposed configuration on the development split           (Table 4 reference run)
    loss         loss-component ablation                                    (Table 7)
    heads        deep-supervision head weights                              (Table 8)
    rho          sampling ratio                                             (Table 9)
    weights      loss-term weights                                          (Table 10)
    augment      data augmentation                                          (Table 11)
    decoder      decoder-free vs encoder-decoder                            (Table 14)
    single_task  single-task WT model, per-epoch threshold sweep            (Section 4.3, Table A1)
    final        3 seeds on the 851 training subjects -> held-out test set  (Table 5)

Ablation groups include the proposed configuration as their reference and
reuse runs/dev/proposed when it already exists. Each ablation is a single run;
the paper notes that differences below about 0.5 pp are within run-to-run
variation.

Run from the repository root (config paths such as data/processed are
relative to the current directory).

Examples:
    python scripts/run_experiments.py --groups loss rho --dry-run
    python scripts/run_experiments.py --groups proposed loss heads rho weights augment decoder
    python scripts/run_experiments.py --groups final
    python scripts/run_experiments.py --groups loss rho --collect-only
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
S = os.path.join(ROOT, "scripts")
C = os.path.join(ROOT, "configs")
A = os.path.join(C, "ablations")

GROUPS = {
    "proposed": [("proposed", "proposed.yaml", "Dice + BCE + Focal, rho 0.25, heads 1.0/0.3/0.2")],
    "loss": [("loss_dice_only", "ablations/loss_dice_only.yaml", "Dice only"),
             ("loss_bce_only", "ablations/loss_bce_only.yaml", "BCE only"),
             ("loss_focal_only", "ablations/loss_focal_only.yaml", "Focal only"),
             ("loss_dice_bce", "ablations/loss_dice_bce.yaml", "Dice + BCE"),
             ("loss_dice_focal", "ablations/loss_dice_focal.yaml", "Dice + Focal"),
             ("loss_bce_focal", "ablations/loss_bce_focal.yaml", "BCE + Focal"),
             ("proposed", "proposed.yaml", "Dice + BCE + Focal (proposed)")],
    "heads": [("proposed", "proposed.yaml", "1.0 / 0.3 / 0.2 (proposed)"),
              ("heads_1_05_05", "ablations/heads_1_05_05.yaml", "1.0 / 0.5 / 0.5"),
              ("heads_1_01_01", "ablations/heads_1_01_01.yaml", "1.0 / 0.1 / 0.1"),
              ("heads_p3_only", "ablations/heads_p3_only.yaml", "1.0 / 0.0 / 0.0 (single P3 head)")],
    "rho": [("rho_000", "ablations/rho_000.yaml", "0.00 (tumour-only)"),
            ("rho_010", "ablations/rho_010.yaml", "0.10"),
            ("proposed", "proposed.yaml", "0.25 (proposed)"),
            ("rho_040", "ablations/rho_040.yaml", "0.40"),
            ("rho_050", "ablations/rho_050.yaml", "0.50")],
    "weights": [("proposed", "proposed.yaml", "0.5 / 0.3 / 0.2 (proposed)"),
                ("weights_04_04_02", "ablations/weights_04_04_02.yaml", "0.4 / 0.4 / 0.2"),
                ("weights_06_02_02", "ablations/weights_06_02_02.yaml", "0.6 / 0.2 / 0.2"),
                ("weights_05_02_03", "ablations/weights_05_02_03.yaml", "0.5 / 0.2 / 0.3"),
                ("weights_equal", "ablations/weights_equal.yaml", "0.33 / 0.33 / 0.34 (equal)")],
    "augment": [("proposed", "proposed.yaml", "No augmentation (proposed)"),
                ("augment", "ablations/augment.yaml", "+ geometric + intensity")],
    "decoder": [("proposed", "proposed.yaml", "Decoder-free (proposed)"),
                ("encoder_decoder", "encoder_decoder.yaml", "Encoder-decoder variant")],
    "single_task": [("single_task_wt", "single_task_wt.yaml", "Single-task WT")],
}
FINAL_SEEDS = (2026, 2027, 2028)


def run(cmd, dry):
    print("$ " + shlex.join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def train_cmd(cfg, out, extra):
    return [PY, os.path.join(S, "train.py"), "--config", os.path.join(C, cfg), "--out-dir", out] + extra


def collect(group, runs_dir, results_dir):
    rows = []
    for name, _, label in GROUPS[group]:
        p = os.path.join(runs_dir, name, "train_summary.json")
        if not os.path.exists(p):
            rows.append((label, None)); continue
        with open(p) as f:
            rows.append((label, json.load(f)))
    lines = ["| Configuration | Validation Macro-Dice | WT | TC | ET | threshold | best epoch |", "|---|---|---|---|---|---|---|"]
    for label, s in rows:
        if s is None:
            lines.append(f"| {label} | (not run) | | | | | |"); continue
        rd = s["val_region_dice"]
        lines.append(f"| {label} | {s['val_macro_dice']:.4f} | " + " | ".join(
            f"{rd[r]:.4f}" if r in rd else "-" for r in ("WT", "TC", "ET"))
            + f" | {s['threshold'][0]} | {s['best_epoch']} |")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"{group}.md")
    with open(out, "w") as f:
        f.write(f"# {group}\n\n" + "\n".join(lines) + "\n")
    print(f"\n[{group}]\n" + "\n".join(lines) + f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", required=True, choices=list(GROUPS) + ["final"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--collect-only", action="store_true", help="only build the result tables")
    ap.add_argument("--rerun", action="store_true", help="retrain even if best.pth exists")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides passed to every run")
    a = ap.parse_args()
    extra = (["--set"] + a.set) if a.set else []
    dev_dir = os.path.join(a.runs_dir, "dev")
    scheduled = set()

    for group in a.groups:
        if group == "final":
            evals = []
            for seed in FINAL_SEEDS:
                out = os.path.join(a.runs_dir, "final", f"seed{seed}")
                ev = os.path.join(a.results_dir, "test", f"seed{seed}")
                if not a.collect_only:
                    if a.rerun or not os.path.exists(os.path.join(out, "best.pth")):
                        run(train_cmd("final.yaml", out, ["--seed", str(seed)] + extra), a.dry_run)
                    run([PY, os.path.join(S, "evaluate.py"), "--checkpoint", os.path.join(out, "best.pth"),
                         "--split", "test", "--out-dir", ev], a.dry_run)
                evals.append(ev)
            run([PY, os.path.join(S, "summarize_seeds.py")] + evals
                + ["--labels"] + [f"Seed {s}" for s in FINAL_SEEDS] + ["--out", os.path.join(a.results_dir, "test", "table5")],
                a.dry_run)
            continue
        if not a.collect_only:
            for name, cfg, _ in GROUPS[group]:
                out = os.path.join(dev_dir, name)
                if out in scheduled:
                    continue
                scheduled.add(out)
                if a.rerun or not os.path.exists(os.path.join(out, "best.pth")):
                    run(train_cmd(cfg, out, extra), a.dry_run)
                else:
                    print(f"[skip] {out} already trained")
        if not a.dry_run:
            collect(group, dev_dir, a.results_dir)


if __name__ == "__main__":
    main()
