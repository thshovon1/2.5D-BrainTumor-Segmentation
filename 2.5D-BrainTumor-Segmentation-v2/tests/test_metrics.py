import math

import numpy as np
import pytest
from scipy import ndimage as ndi

from bt25d.metrics import _surface, case_metrics, dice_from_counts, hd95, mean_sd, pooled


def brute_hd95(pred, gt, spacing):
    sp = np.argwhere(_surface(pred)) * np.asarray(spacing)
    sg = np.argwhere(_surface(gt)) * np.asarray(spacing)
    d = np.sqrt(((sp[:, None, :] - sg[None, :, :]) ** 2).sum(-1))
    return max(np.percentile(d.min(1), 95), np.percentile(d.min(0), 95))


def test_dice_values():
    assert dice_from_counts(0, 0, 0) == 1.0
    assert dice_from_counts(0, 5, 0) == 0.0
    assert dice_from_counts(3, 1, 1) == pytest.approx(0.75)


def test_hd95_single_voxels_uses_spacing():
    a = np.zeros((8, 8, 8), bool); b = np.zeros_like(a)
    a[1, 2, 2] = True; b[4, 2, 2] = True
    assert hd95(a, b, spacing=(2.0, 1.0, 1.0)) == pytest.approx(6.0)
    assert hd95(a, b, spacing=(1.0, 1.5, 1.5)) == pytest.approx(3.0)


@pytest.mark.parametrize("seed", range(5))
def test_hd95_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    a = ndi.binary_dilation(rng.random((20, 22, 18)) > 0.985, iterations=2)
    b = ndi.binary_dilation(rng.random((20, 22, 18)) > 0.985, iterations=2)
    spacing = (1.0, 1.5, 1.5)
    assert hd95(a, b, spacing) == pytest.approx(brute_hd95(a, b, spacing), abs=1e-6)


def test_hd95_empty_conventions():
    e = np.zeros((5, 5, 5), bool); f = e.copy(); f[2, 2, 2] = True
    assert hd95(e, e) == 0.0
    assert math.isnan(hd95(e, f)) and math.isnan(hd95(f, e))
    assert hd95(e, f, empty_penalty=373.13) == pytest.approx(373.13)


def test_case_metrics_and_pooled():
    gt = np.zeros((4, 10, 10), bool); gt[1:3, 2:6, 2:6] = True    # 32 voxels
    pr = np.zeros_like(gt); pr[1:3, 2:6, 4:8] = True               # 16 overlap
    m = case_metrics(pr, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (16, 16, 16)
    assert m["dice"] == pytest.approx(0.5)
    assert m["sensitivity"] == pytest.approx(0.5) and m["precision"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1 - 16 / (400 - 32))
    e = case_metrics(np.zeros_like(gt), np.zeros_like(gt))
    assert e["dice"] == 1.0 and math.isnan(e["sensitivity"]) and math.isnan(e["precision"])
    p = pooled([m, e])
    assert p["dice"] == pytest.approx(0.5)


def test_mean_sd_excludes_nan():
    mu, sd, n = mean_sd([1.0, 3.0, float("nan")])
    assert (mu, n) == (2.0, 2) and sd == pytest.approx(math.sqrt(2))
