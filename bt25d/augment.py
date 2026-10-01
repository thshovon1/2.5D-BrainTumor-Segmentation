"""On-the-fly augmentation for the augmentation ablation (paper Section 4.4.5).

Geometric: random horizontal/vertical flips and an in-plane rotation (images
bilinear, masks nearest). Intensity: gamma and additive Gaussian noise applied
to brain voxels only (non-zero after z-scoring), with the same random draw for
a sequence across the three stacked slices. Disabled in the final
configuration.
"""

import math
import torch
import torch.nn.functional as F


class Augment:
    def __init__(self, p_flip=0.5, p_rot=0.5, max_rot_deg=15.0, p_gamma=0.3, gamma_range=(0.7, 1.5),
                 p_noise=0.3, noise_sd=0.1, n_modalities=4):
        self.p_flip, self.p_rot, self.max_rot = p_flip, p_rot, max_rot_deg
        self.p_gamma, self.gamma_range = p_gamma, gamma_range
        self.p_noise, self.noise_sd, self.nm = p_noise, noise_sd, n_modalities

    @staticmethod
    def _u():
        return float(torch.rand(()))

    def __call__(self, x, y):
        if self._u() < self.p_flip:
            x, y = torch.flip(x, [2]), torch.flip(y, [2])
        if self._u() < self.p_flip:
            x, y = torch.flip(x, [1]), torch.flip(y, [1])
        if self._u() < self.p_rot:
            a = math.radians((self._u() * 2 - 1) * self.max_rot)
            theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]]).unsqueeze(0)
            grid = F.affine_grid(theta, (1, x.shape[0], x.shape[1], x.shape[2]), align_corners=False)
            x = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
            y = F.grid_sample(y.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
        brain = x != 0
        if self._u() < self.p_gamma:
            x = x.clone()
            for m in range(self.nm):
                g = self.gamma_range[0] + self._u() * (self.gamma_range[1] - self.gamma_range[0])
                chans = list(range(m, x.shape[0], self.nm))
                v = x[chans]; b = brain[chans]
                if b.any():
                    lo, hi = v[b].min(), v[b].max()
                    scaled = ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1) ** g * (hi - lo) + lo
                    x[chans] = torch.where(b, scaled, v)
        if self._u() < self.p_noise:
            x = torch.where(brain, x + torch.randn_like(x) * self.noise_sd, x)
        return x, y
