"""Decoder-free 2.5D multi-scale feed-forward network (paper Sections 3.4-3.5).

Module and attribute names (e1, e2, e3, att2, att3, aspp, ds1, ds2, out and
the sub-module names inside each block) are kept identical to the original
training notebook, so checkpoints saved from that code load with
``strict=True``. ``scripts/check_checkpoint.py`` reports any mismatch.

Architecture (Eqs. 3-5):
    F1 = ResBlock(12 -> 32)            at 160 x 160
    F2 = ResBlock(32 -> 64)(pool F1)   at  80 x  80
    F3 = ResBlock(64 -> 128)(pool F2)  at  40 x  40
    attention gates (Eq. 4, F * sigmoid(Conv1x1 F)) recalibrate F2 and F3;
    the main path pools the *ungated* F2, as in Eq. 3.
    ASPP (1x1 branch + 3x3 branches with dilation 6 and 12, fused by 1x1)
    is applied to the gated F3.
    Heads: P1 = 1x1 conv on F1, P2 = 1x1 conv on gated F2,
           P3 = 1x1 conv on ASPP(gated F3); all bilinearly upsampled to the
           input size. P3 is the inference output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    """Self-gating: F * sigmoid(Conv1x1(F))  (paper Eq. 4)."""

    def __init__(self, c: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.attn(x)


class ASPP(nn.Module):
    def __init__(self, in_c: int, out_c: int, rates=(6, 12)):
        super().__init__()
        r1, r2 = rates
        self.conv1 = nn.Conv2d(in_c, out_c, 1)
        self.conv2 = nn.Conv2d(in_c, out_c, 3, padding=r1, dilation=r1)
        self.conv3 = nn.Conv2d(in_c, out_c, 3, padding=r2, dilation=r2)
        self.out = nn.Conv2d(out_c * 3, out_c, 1)

    def forward(self, x):
        return self.out(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))


class DecoderFree25D(nn.Module):
    """Returns (P3, P2, P1) logits, each of shape (B, n_classes, H, W)."""

    def __init__(self, in_channels: int = 12, n_classes: int = 3, widths=(32, 64, 128), aspp_rates=(6, 12)):
        super().__init__()
        w1, w2, w3 = widths
        self.e1 = ResBlock(in_channels, w1)
        self.e2 = ResBlock(w1, w2)
        self.e3 = ResBlock(w2, w3)
        self.pool = nn.MaxPool2d(2)
        self.att2 = AttentionGate(w2)
        self.att3 = AttentionGate(w3)
        self.aspp = ASPP(w3, w3, aspp_rates)
        self.ds1 = nn.Conv2d(w1, n_classes, 1)
        self.ds2 = nn.Conv2d(w2, n_classes, 1)
        self.out = nn.Conv2d(w3, n_classes, 1)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x2 = self.att2(x2)
        x3 = self.aspp(self.att3(x3))
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return up(self.out(x3)), up(self.ds2(x2)), up(self.ds1(x1))


class EncoderDecoder25D(nn.Module):
    """Controlled comparison (paper Section 4.6): identical encoder, attention
    gates and ASPP; the three direct heads are replaced by a decoder with two
    learned upsampling stages (2 x 2 transposed convolutions, stride 2). The
    first fuses the bottleneck with the gated F2 and the second with F1
    (concatenation followed by a residual block); one 1 x 1 head then predicts
    at full resolution. Returns a 1-tuple (logits,)."""

    def __init__(self, in_channels: int = 12, n_classes: int = 3, widths=(32, 64, 128), aspp_rates=(6, 12)):
        super().__init__()
        w1, w2, w3 = widths
        self.e1 = ResBlock(in_channels, w1)
        self.e2 = ResBlock(w1, w2)
        self.e3 = ResBlock(w2, w3)
        self.pool = nn.MaxPool2d(2)
        self.att2 = AttentionGate(w2)
        self.att3 = AttentionGate(w3)
        self.aspp = ASPP(w3, w3, aspp_rates)
        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = ResBlock(w2 + w2, w2)
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = ResBlock(w1 + w1, w1)
        self.out = nn.Conv2d(w1, n_classes, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x2 = self.att2(x2)
        x3 = self.aspp(self.att3(x3))
        d2 = self.dec2(torch.cat([self.up2(x3), x2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], 1))
        return (self.out(d1),)


# Backwards-compatible name used in the original notebook.
Model = DecoderFree25D

ARCHS = {"decoder_free": DecoderFree25D, "encoder_decoder": EncoderDecoder25D}


def build_model(cfg_model: dict) -> nn.Module:
    arch = cfg_model.get("arch", "decoder_free")
    kwargs = dict(in_channels=cfg_model.get("in_channels", 12), n_classes=cfg_model.get("n_classes", 3),
                  widths=tuple(cfg_model.get("widths", (32, 64, 128))),
                  aspp_rates=tuple(cfg_model.get("aspp_rates", (6, 12))))
    return ARCHS[arch](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


_WRAPPER_KEYS = ("model", "state_dict", "model_state_dict", "net")
_PREFIXES = ("module.", "_orig_mod.", "model.")


def read_checkpoint(path: str, map_location="cpu"):
    """Return (state_dict, metadata) from either a bare state_dict (original
    notebook format) or a checkpoint dict (scripts/train.py writes
    {'model', 'config', 'epoch', 'threshold', 'val'}). Prefixes added by
    DataParallel ('module.') or torch.compile ('_orig_mod.') are removed."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    meta = {}
    if isinstance(obj, dict):
        for k in _WRAPPER_KEYS:
            if k in obj and isinstance(obj[k], dict):
                meta = {kk: vv for kk, vv in obj.items() if kk != k}
                obj = obj[k]
                break
    state = dict(obj)
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if state and all(k.startswith(p) for k in state):
                state = {k[len(p):]: v for k, v in state.items()}
                changed = True
    return state, meta


def infer_model_config(state: dict) -> dict:
    """Infer arch / in_channels / n_classes / widths from a state_dict
    (dilation rates are not stored in weights and keep their defaults)."""
    arch = "encoder_decoder" if any(k.startswith("dec2.") for k in state) else "decoder_free"
    w1, in_c = state["e1.conv.0.weight"].shape[:2]
    w2 = state["e2.conv.0.weight"].shape[0]
    w3 = state["e3.conv.0.weight"].shape[0]
    n_classes = state["out.weight"].shape[0]
    return {"arch": arch, "in_channels": int(in_c), "n_classes": int(n_classes), "widths": [int(w1), int(w2), int(w3)]}


def load_checkpoint(model: nn.Module, path: str, strict: bool = True, map_location="cpu"):
    """Load weights into ``model``. Returns (metadata dict, load result)."""
    state, meta = read_checkpoint(path, map_location)
    result = model.load_state_dict(state, strict=strict)
    return meta, result
