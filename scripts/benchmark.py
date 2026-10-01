#!/usr/bin/env python
"""Model complexity and standardised inference benchmark (paper Section 4.7, Tables 14-16).

Complexity (any device): parameters, multiply-accumulates (MACs) and FLOPs
(= 2 x MACs) for one 12 x 160 x 160 input slice, for the proposed model, the
encoder-decoder variant and the 2D U-Net / Attention U-Net baselines (32 base
filters).

Timing (GPU recommended): a full volume is processed slice by slice at batch
size 1 after a warm-up pass, repeated --repeats times (medians are reported).
    preprocessing time = loading the preprocessed volume + building the 2.5D inputs
                         (the offline NIfTI preprocessing is not included; time it
                         separately with --raw-subject-dir)
    model time         = host-to-device copy + forward passes + sigmoid for every
                         slice + copying the probability volume back, synchronised
Peak GPU memory is torch.cuda.max_memory_allocated during model inference.

Examples:
    python scripts/benchmark.py                                  # complexity + timing on a synthetic volume
    python scripts/benchmark.py --data-dir data/processed --subject BraTS2021_00000 --repeats 20
    python scripts/benchmark.py --checkpoint runs/proposed/best.pth --archs decoder_free
    python scripts/benchmark.py --archs decoder_free --raw-subject-dir /data/BraTS2021_Training_Data/BraTS2021_00000
"""

import argparse
import os
import statistics
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bt25d.baselines import attention_unet2d, unet2d  # noqa: E402
from bt25d.complexity import summarize  # noqa: E402
from bt25d.data import volume_to_25d  # noqa: E402
from bt25d.model import DecoderFree25D, EncoderDecoder25D, load_checkpoint  # noqa: E402
from bt25d.utils import get_device, save_json  # noqa: E402

BUILDERS = {
    "decoder_free": lambda: DecoderFree25D(12, 3),
    "encoder_decoder": lambda: EncoderDecoder25D(12, 3),
    "unet2d": lambda: unet2d(base=32, in_channels=12, n_classes=3),
    "attention_unet2d": lambda: attention_unet2d(base=32, in_channels=12, n_classes=3),
}
NAMES = {"decoder_free": "Proposed (decoder-free)", "encoder_decoder": "Encoder-decoder variant",
         "unet2d": "2D U-Net (32 base filters)", "attention_unet2d": "Attention U-Net (32 base filters)"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_volume(model, path, device, amp):
    t0 = time.perf_counter()
    image = np.load(path)
    x_all = volume_to_25d(image)
    t_pre = time.perf_counter() - t0
    sync(device)
    t1 = time.perf_counter()
    use_amp = amp and device.type == "cuda"
    outs = []
    for i in range(x_all.shape[0]):
        x = torch.from_numpy(x_all[i:i + 1]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outs.append(torch.sigmoid(model(x)[0].float()))
    torch.cat(outs).cpu()
    sync(device)
    return t_pre, time.perf_counter() - t1, x_all.shape[0]


def time_raw_preprocessing(subj_dir, size, repeats):
    from bt25d.preprocessing import load_subject, to_model_input
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        stack, _, _ = load_subject(subj_dir)
        to_model_input(stack, size, "resize")
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--checkpoint", default=None, help="optional weights for decoder_free (timing is weight-independent)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--subject", default=None, help="subject folder in --data-dir to time on (default: synthetic volume)")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--depth", type=int, default=155)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--raw-subject-dir", default=None,
                    help="BraTS NIfTI subject folder: also time the offline preprocessing (load, z-score, resize)")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()
    device = get_device(a.device)

    tmp = None
    if a.subject:
        vol_path = os.path.join(a.data_dir or "data/processed", a.subject, "image.npy")
    else:
        tmp = tempfile.TemporaryDirectory()
        vol_path = os.path.join(tmp.name, "image.npy")
        rng = np.random.default_rng(0)
        np.save(vol_path, rng.standard_normal((a.depth, 4, a.size, a.size)).astype(np.float16))

    results = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "torch": torch.__version__, "input": [12, a.size, a.size], "batch_size": 1,
               "amp": (not a.no_amp) and device.type == "cuda", "volume": a.subject or f"synthetic {a.depth} slices",
               "models": {}}
    for arch in a.archs:
        model = BUILDERS[arch]()
        if arch == "decoder_free" and a.checkpoint:
            load_checkpoint(model, a.checkpoint, strict=False)
        r = summarize(model, (1, 12, a.size, a.size))
        r["name"] = NAMES[arch]
        if not a.no_timing:
            model.to(device).eval()
            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            time_volume(model, vol_path, device, not a.no_amp)  # warm-up
            pre, inf = [], []
            for _ in range(a.repeats):
                tp, ti, n = time_volume(model, vol_path, device, not a.no_amp)
                pre.append(tp); inf.append(ti)
            r.update({"preprocess_s": statistics.median(pre), "inference_s": statistics.median(inf),
                      "inference_s_sd": statistics.stdev(inf) if len(inf) > 1 else 0.0,
                      "total_s": statistics.median(pre) + statistics.median(inf),
                      "slices_per_s": n / statistics.median(inf), "repeats": a.repeats})
            if device.type == "cuda":
                r["peak_gpu_memory_GB"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            model.cpu()
        results["models"][arch] = r

    if a.raw_subject_dir:
        results["raw_preprocess_s"] = time_raw_preprocessing(a.raw_subject_dir, a.size, max(1, min(a.repeats, 5)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_json(results, a.out)
    if tmp:
        tmp.cleanup()

    print(f"device: {results['gpu'] or device} | input 12 x {a.size} x {a.size} | batch size 1")
    head = f"{'model':<36}{'params (M)':>11}{'GMACs':>8}{'GFLOPs':>8}"
    if not a.no_timing:
        head += f"{'pre (s)':>9}{'model (s)':>10}{'slices/s':>10}" + (f"{'peak GB':>9}" if device.type == "cuda" else "")
    print(head)
    for arch, r in results["models"].items():
        line = f"{r['name']:<36}{r['parameters_M']:>11.3f}{r['GMACs']:>8.2f}{r['GFLOPs']:>8.2f}"
        if not a.no_timing:
            line += f"{r['preprocess_s']:>9.3f}{r['inference_s']:>10.3f}{r['slices_per_s']:>10.1f}"
            if device.type == "cuda":
                line += f"{r['peak_gpu_memory_GB']:>9.2f}"
        print(line)
    if "raw_preprocess_s" in results:
        print(f"offline NIfTI preprocessing (load, z-score, resize): {results['raw_preprocess_s']:.3f} s per subject")
    print("FLOPs = 2 x MACs (tools such as thop/ptflops report MACs under the name FLOPs).")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
