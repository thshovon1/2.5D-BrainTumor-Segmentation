"""Whole-volume slice-wise inference (paper Section 3.8)."""

import numpy as np
import torch

from .data import volume_to_25d


@torch.no_grad()
def predict_volume(model, image: np.ndarray, device, batch_slices: int = 64, amp: bool = True) -> np.ndarray:
    """image: (D, 4, S, S) preprocessed volume -> sigmoid probabilities of the
    main head (P3), shape (n_classes, D, S, S), float32.

    Every axial slice is processed with its two neighbours as a 12-channel
    input; the slice predictions are stacked back into the full volume.
    ``batch_slices`` only groups slices for speed and does not change the
    result (BatchNorm runs in eval mode)."""
    model.eval()
    x_all = volume_to_25d(image)
    out = []
    use_amp = amp and device.type == "cuda"
    for k in range(0, x_all.shape[0], batch_slices):
        x = torch.from_numpy(x_all[k:k + batch_slices]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)[0]
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out, axis=0).transpose(1, 0, 2, 3)
