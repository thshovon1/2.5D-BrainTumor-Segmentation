#!/usr/bin/env python
"""Check that a checkpoint matches the architecture implemented in bt25d/model.py.

Reports the tensors stored in the checkpoint, the architecture inferred from
them, whether they load with strict=True, and the parameter / MAC / FLOP
counts of the matching model. If the checkpoint does not load, the missing,
unexpected and shape-mismatched tensors are listed so the model code can be
aligned with the weights that produced the reported results.

Accepts bare state_dicts (original notebook format) and checkpoint dicts
({'model': ...}, {'state_dict': ...}); DataParallel / torch.compile prefixes
are removed automatically.

Example:
    python scripts/check_checkpoint.py weights/*.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d.complexity import summarize  # noqa: E402
from bt25d.model import build_model, infer_model_config, read_checkpoint  # noqa: E402


def check(path, verbose):
    print(f"\n=== {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    state, meta = read_checkpoint(path)
    tensors = {k: v for k, v in state.items() if hasattr(v, "shape")}
    n_stored = sum(v.numel() for k, v in tensors.items()
                   if not k.endswith(("running_mean", "running_var", "num_batches_tracked")))
    print(f"tensors: {len(tensors)} | stored parameter values (excluding BatchNorm buffers): {n_stored:,}")
    if meta:
        print("metadata: " + ", ".join(f"{k}={meta[k]}" for k in ("epoch", "threshold", "version") if k in meta))
    try:
        mcfg = infer_model_config(state)
    except KeyError as e:
        print(f"could not infer the architecture (missing {e}); first tensors:")
        for k in list(tensors)[:40]:
            print(f"  {k:<45} {tuple(tensors[k].shape)}")
        return False
    print(f"inferred: arch={mcfg['arch']} in_channels={mcfg['in_channels']} n_classes={mcfg['n_classes']} "
          f"widths={mcfg['widths']}")
    model = build_model(mcfg)
    ref = model.state_dict()
    missing = [k for k in ref if k not in state]
    unexpected = [k for k in state if k not in ref]
    shape = [(k, tuple(state[k].shape), tuple(ref[k].shape)) for k in ref
             if k in state and hasattr(state[k], "shape") and tuple(state[k].shape) != tuple(ref[k].shape)]
    ok = not (missing or unexpected or shape)
    if ok:
        model.load_state_dict(state, strict=True)
        s = summarize(model, (1, mcfg["in_channels"], 160, 160))
        print("strict load: OK - the checkpoint matches bt25d/model.py exactly")
        print(f"parameters: {s['parameters']:,} ({s['parameters_M']:.3f} M)")
        print(f"per 12 x 160 x 160 slice: {s['GMACs']:.3f} GMACs = {s['GFLOPs']:.3f} GFLOPs (FLOPs = 2 x MACs)")
    else:
        print("strict load: FAILED - the checkpoint does not match bt25d/model.py")
        for k in missing[:50]:
            print(f"  missing in checkpoint: {k} {tuple(ref[k].shape)}")
        for k in unexpected[:50]:
            print(f"  unexpected in checkpoint: {k} {tuple(state[k].shape) if hasattr(state[k], 'shape') else ''}")
        for k, a, b in shape[:50]:
            print(f"  shape mismatch: {k} checkpoint {a} vs model {b}")
        if verbose:
            print("all checkpoint tensors:")
            for k, v in tensors.items():
                print(f"  {k:<45} {tuple(v.shape)}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every tensor on failure")
    a = ap.parse_args()
    results = [check(p, a.verbose) for p in a.checkpoints]
    print(f"\n{sum(results)}/{len(results)} checkpoint(s) match the model code")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
