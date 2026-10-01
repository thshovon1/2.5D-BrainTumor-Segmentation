"""2D reference networks used only for the complexity comparison (paper Table 16).

Standard 2D U-Net (Ronneberger et al.) with four down-sampling stages,
transposed-convolution up-sampling and a configurable base width, and an
Attention U-Net (Oktay et al.) that adds additive attention gates on the skip
connections. Both take the same 12 x 160 x 160 input as the proposed model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class AdditiveGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.wx = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.wg(g) + self.wx(x)))


class UNet2D(nn.Module):
    def __init__(self, in_channels=12, n_classes=3, base=32, attention=False):
        super().__init__()
        c = [base * 2 ** k for k in range(5)]
        self.inc = DoubleConv(in_channels, c[0])
        self.downs = nn.ModuleList([DoubleConv(c[k], c[k + 1]) for k in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in reversed(range(4))])
        self.decs = nn.ModuleList([DoubleConv(c[k] * 2, c[k]) for k in reversed(range(4))])
        self.gates = nn.ModuleList([AdditiveGate(c[k], c[k], c[k] // 2) for k in reversed(range(4))]) if attention else None
        self.outc = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for j, (up, dec) in enumerate(zip(self.ups, self.decs)):
            h = up(h)
            s = skips.pop()
            if self.gates is not None:
                s = self.gates[j](h, s)
            h = dec(torch.cat([s, h], 1))
        return (self.outc(h),)


def unet2d(base=32, **kw):
    return UNet2D(base=base, attention=False, **kw)


def attention_unet2d(base=32, **kw):
    return UNet2D(base=base, attention=True, **kw)
