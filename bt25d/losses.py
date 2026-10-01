"""Hybrid loss with deep supervision (paper Eqs. 6-11).

Per head:   L_head = w_dice * Dice + w_bce * weighted BCE + w_focal * Focal   (Eq. 10)
For the three-channel model each term is averaged over the WT, TC and ET
channels (Eq. 11); averaging per-element terms over (B, C, H, W) and the Dice
term over (B, C) is exactly that channel average.
Across heads: L = sum_k head_weight_k * L_head_k, default (P3, P2, P1) =
(1.0, 0.3, 0.2) (Eq. 9).
Setting a weight to 0 removes that term, which is how the loss-component and
head-weight ablations are run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, eps: float = 1e-5):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    union = p.sum((2, 3)) + target.sum((2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def weighted_bce(logits, target, pos_weight: float = 2.0):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def focal_loss(logits, target, alpha: float = 0.8, gamma: float = 2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


class HybridLoss(nn.Module):
    def __init__(self, w_dice=0.5, w_bce=0.3, w_focal=0.2, pos_weight=2.0,
                 focal_alpha=0.8, focal_gamma=2.0, head_weights=(1.0, 0.3, 0.2)):
        super().__init__()
        self.w = (float(w_dice), float(w_bce), float(w_focal))
        if sum(self.w) <= 0:
            raise ValueError("at least one loss weight must be positive")
        self.pos_weight, self.alpha, self.gamma = pos_weight, focal_alpha, focal_gamma
        self.head_weights = tuple(float(h) for h in head_weights)

    def head_loss(self, logits, target):
        logits = logits.float()
        wd, wb, wf = self.w
        total = logits.new_zeros(())
        if wd: total = total + wd * dice_loss(logits, target)
        if wb: total = total + wb * weighted_bce(logits, target, self.pos_weight)
        if wf: total = total + wf * focal_loss(logits, target, self.alpha, self.gamma)
        return total

    def forward(self, outputs, target):
        if torch.is_tensor(outputs):
            outputs = (outputs,)
        weights = self.head_weights[:len(outputs)]
        total = outputs[0].new_zeros((), dtype=torch.float32)
        for w, o in zip(weights, outputs):
            if w:
                total = total + w * self.head_loss(o, target)
        return total
