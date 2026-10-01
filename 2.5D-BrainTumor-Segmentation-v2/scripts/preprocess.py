#!/usr/bin/env python
"""Convert BraTS 2021 NIfTI subjects into the preprocessed layout used for training.

Input: a directory with one folder per subject, e.g.
    BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_{t1,t1ce,t2,flair,seg}.nii.gz
(extract the official archive first).

Per subject: z-score normalisation of each sequence within the brain mask
(non-zero voxels), in-plane mapping to the model grid (``--mode resize``, as in
the paper: 240 x 240 -> 160 x 160, i.e. 1.5 mm pixels; or ``--mode crop``:
centre crop at 1 mm), and storage as memory-mappable arrays. Original labels
are also kept at native resolution so that evaluation can optionally be run
on the original grid (scripts/evaluate.py --grid native).

Disk use for all 1,251 subjects at 160 x 160: about 40 GB (float16).

Example:
    python scripts/preprocess.py --raw-dir /data/BraTS2021_Training_Data --out-dir data/processed \
        --size 160 --mode resize --workers 8
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d.geometry import label_to_model_grid  # noqa: E402
from bt25d.preprocessing import find_sequence, load_subject, to_model_input  # noqa: E402


def process_subject(args):
    subj_dir, out_dir, size, mode, overwrite = args
    sid = os.path.basename(os.path.normpath(subj_dir))
    dst = os.path.join(out_dir, sid)
    if not overwrite and os.path.exists(os.path.join(dst, "meta.json")):
        return sid, "skipped"
    seg_p = find_sequence(subj_dir, sid, "seg")
    if seg_p is None:
        return sid, "missing seg"
    try:
        stack, ref, zooms = load_subject(subj_dir)
    except FileNotFoundError as e:
        return sid, str(e)
    seg = np.asarray(nib.load(seg_p).dataobj).astype(np.uint8)
    if seg.shape != ref.shape[:3]:
        return sid, f"seg shape {seg.shape} != image shape {ref.shape[:3]}"
    shape = seg.shape                                   # (H, W, D) as stored in the NIfTI
    seg_dhw = np.ascontiguousarray(seg.transpose(2, 0, 1))
    image = to_model_input(stack, size, mode)          # (D, 4, S, S) float16
    label = label_to_model_grid(seg_dhw, size, mode)   # (D, S, S) uint8

    os.makedirs(dst, exist_ok=True)
    np.save(os.path.join(dst, "image.npy"), image)
    np.save(os.path.join(dst, "label.npy"), label)
    np.savez_compressed(os.path.join(dst, "label_native.npz"), label=seg_dhw)
    pix = ([zooms[0] * shape[0] / size, zooms[1] * shape[1] / size] if mode == "resize"
           else [zooms[0], zooms[1]])
    meta = {"subject_id": sid, "mode": mode, "size": size,
            "native_shape_hwd": [int(s) for s in shape],
            # voxel spacing in array order (D, H, W), in mm
            "spacing_dhw": [zooms[2], zooms[0], zooms[1]],          # native grid
            "model_spacing_dhw": [zooms[2], pix[0], pix[1]],        # model grid (1 x 1.5 x 1.5 for resize)
            "n_slices": int(label.shape[0]),
            "tumour_slices": int((label.reshape(label.shape[0], -1) > 0).any(1).sum())}
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return sid, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--mode", choices=["resize", "crop"], default="resize")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    subjects = sorted(d for d in os.listdir(a.raw_dir)
                      if os.path.isdir(os.path.join(a.raw_dir, d)) and d.startswith("BraTS"))
    if not subjects:
        sys.exit(f"no subject folders found in {a.raw_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(os.path.join(a.raw_dir, s), a.out_dir, a.size, a.mode, a.overwrite) for s in subjects]
    status = {}
    with Pool(a.workers) as pool:
        for k, (sid, st) in enumerate(pool.imap_unordered(process_subject, jobs), 1):
            status[sid] = st
            if st not in ("ok", "skipped"):
                print(f"[warn] {sid}: {st}")
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} subjects", flush=True)
    bad = {s: v for s, v in status.items() if v not in ("ok", "skipped")}
    tumour = total = 0
    for s in status:
        if s not in bad:
            with open(os.path.join(a.out_dir, s, "meta.json")) as f:
                m = json.load(f)
            tumour += m["tumour_slices"]; total += m["n_slices"]
    print(f"done: {len(status) - len(bad)} subjects in {a.out_dir}; {len(bad)} failed")
    print(f"slices: {total} total, {tumour} tumour-present, {total - tumour} tumour-free")


if __name__ == "__main__":
    main()
