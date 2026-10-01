#!/usr/bin/env python
"""Segment new BraTS-format subjects with a trained model and write NIfTI label maps.

Input: one or more subject folders containing <id>_{t1,t1ce,t2,flair}.nii.gz
(co-registered and skull-stripped, as distributed by BraTS). The output label
map uses the BraTS convention on the original grid and header:
    2 = WT only (edema), 1 = TC only (necrosis / non-enhancing core), 4 = ET.
Regions are written in the order WT -> TC -> ET, so a voxel predicted as ET
is labelled 4 and a voxel predicted as TC or ET always counts as tumour; the
WT region read back from the file is therefore the union of the three
predicted channels. No post-processing is applied.

--size and --mode must match the preprocessing used for training.

Example:
    python scripts/predict.py --checkpoint weights/decoder_free_seed2026.pth \
        --inputs /data/BraTS2021_00000 --out-dir predictions
"""

import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d.config import load_config  # noqa: E402
from bt25d.evaluation import binarize  # noqa: E402
from bt25d.geometry import to_native  # noqa: E402
from bt25d.inference import predict_volume  # noqa: E402
from bt25d.model import build_model, infer_model_config, read_checkpoint  # noqa: E402
from bt25d.preprocessing import load_subject, to_model_input  # noqa: E402
from bt25d.utils import get_device  # noqa: E402


def regions_to_label(pred: np.ndarray) -> np.ndarray:
    """(C, D, H, W) binary regions (WT, TC, ET) -> (D, H, W) BraTS labels."""
    lab = np.zeros(pred.shape[1:], dtype=np.uint8)
    lab[pred[0]] = 2
    if pred.shape[0] > 1:
        lab[pred[1]] = 1
    if pred.shape[0] > 2:
        lab[pred[2]] = 4
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="subject folder(s)")
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--threshold", type=float, nargs="+", default=None, help="default: value stored in the checkpoint")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--mode", choices=["resize", "crop"], default="resize")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-probabilities", action="store_true", help="also write <id>_prob.nii.gz (WT, TC, ET probabilities as 4D float32)")
    a = ap.parse_args()

    state, meta = read_checkpoint(a.checkpoint)
    mcfg = dict((meta.get("config") or load_config())["model"]); mcfg.update(infer_model_config(state))
    model = build_model(mcfg); model.load_state_dict(state, strict=True)
    device = get_device(a.device); model.to(device).eval()
    thr = a.threshold if a.threshold is not None else meta.get("threshold")
    if thr is None:
        sys.exit("no threshold stored in the checkpoint - pass --threshold")
    thr = [float(t) for t in (thr if isinstance(thr, (list, tuple)) else [thr])]
    thr = thr * mcfg["n_classes"] if len(thr) == 1 else thr
    os.makedirs(a.out_dir, exist_ok=True)

    for subj in a.inputs:
        sid = os.path.basename(os.path.normpath(subj))
        t0 = time.time()
        stack, ref, _ = load_subject(subj)
        probs = predict_volume(model, to_model_input(stack, a.size, a.mode), device)
        probs = to_native(probs, stack.shape[2:], a.mode)          # (C, D, H, W)
        lab = regions_to_label(binarize(probs, thr)).transpose(1, 2, 0)  # -> NIfTI (H, W, D)
        hdr = ref.header.copy(); hdr.set_data_dtype(np.uint8)
        out = os.path.join(a.out_dir, f"{sid}_pred.nii.gz")
        nib.save(nib.Nifti1Image(lab, ref.affine, hdr), out)
        if a.save_probabilities:
            p = probs.transpose(2, 3, 1, 0).astype(np.float32)             # (H, W, D, C)
            nib.save(nib.Nifti1Image(p, ref.affine), os.path.join(a.out_dir, f"{sid}_prob.nii.gz"))
        vox = {k: int((lab == v).sum()) for k, v in (("edema(2)", 2), ("core(1)", 1), ("enhancing(4)", 4))}
        print(f"{sid}: {out} ({time.time() - t0:.1f}s) voxels {vox}")


if __name__ == "__main__":
    main()
