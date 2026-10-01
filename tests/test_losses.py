import pytest
import torch

from bt25d.losses import HybridLoss, dice_loss, focal_loss, weighted_bce


def test_perfect_prediction_small_loss():
    y = torch.zeros(2, 3, 16, 16); y[:, :, 4:10, 4:10] = 1
    logits = (y * 2 - 1) * 20
    assert dice_loss(logits, y) < 1e-3
    assert weighted_bce(logits, y) < 1e-6 and focal_loss(logits, y) < 1e-6


def test_zero_weights_disable_terms():
    torch.manual_seed(0)
    y = (torch.rand(2, 3, 8, 8) > 0.7).float(); o = torch.randn(2, 3, 8, 8)
    only_dice = HybridLoss(w_dice=1, w_bce=0, w_focal=0, head_weights=(1.0,))
    assert only_dice((o,), y).item() == pytest.approx(dice_loss(o, y).item(), rel=1e-6)
    full = HybridLoss()
    expected = 0.5 * dice_loss(o, y) + 0.3 * weighted_bce(o, y, 2.0) + 0.2 * focal_loss(o, y, 0.8, 2.0)
    assert full.head_loss(o, y).item() == pytest.approx(expected.item(), rel=1e-6)


def test_deep_supervision_weights():
    torch.manual_seed(1)
    y = (torch.rand(1, 3, 8, 8) > 0.5).float(); outs = tuple(torch.randn(1, 3, 8, 8) for _ in range(3))
    L = HybridLoss(head_weights=(1.0, 0.3, 0.2))
    exp = sum(w * L.head_loss(o, y) for w, o in zip((1.0, 0.3, 0.2), outs))
    assert L(outs, y).item() == pytest.approx(exp.item(), rel=1e-6)
    p3_only = HybridLoss(head_weights=(1.0, 0.0, 0.0))
    assert p3_only(outs, y).item() == pytest.approx(L.head_loss(outs[0], y).item(), rel=1e-6)


def test_pos_weight_increases_positive_penalty():
    y = torch.ones(1, 1, 4, 4); o = torch.zeros(1, 1, 4, 4)
    assert weighted_bce(o, y, 2.0).item() == pytest.approx(2 * weighted_bce(o, y, 1.0).item(), rel=1e-6)
