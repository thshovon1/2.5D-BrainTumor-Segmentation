import numpy as np
import pytest

from bt25d.data import SliceDataset, balanced_index, n_empty_slices, stack_25d, volume_to_25d
from bt25d.labels import regions_from_label


def test_regions_nested():
    lab = np.array([0, 1, 2, 3, 4])
    r = regions_from_label(lab)
    assert r.tolist() == [[0, 1, 1, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1]]
    assert np.all(r[2] <= r[1]) and np.all(r[1] <= r[0])


def test_25d_stacking_order_and_padding():
    d, s = 5, 4
    img = np.zeros((d, 4, s, s), np.float16)
    for i in range(d):
        for m in range(4):
            img[i, m] = 10 * (i + 1) + m
    vol = volume_to_25d(img)
    assert vol.shape == (d, 12, s, s)
    for i in range(d):
        x = stack_25d(img, i)
        np.testing.assert_array_equal(x, vol[i])
        np.testing.assert_array_equal(x[4:8, 0, 0], [10 * (i + 1) + m for m in range(4)])  # current slice
        prev = x[0:4, 0, 0]; nxt = x[8:12, 0, 0]
        assert np.all(prev == 0) if i == 0 else np.array_equal(prev, [10 * i + m for m in range(4)])
        assert np.all(nxt == 0) if i == d - 1 else np.array_equal(nxt, [10 * (i + 2) + m for m in range(4)])


def _write_subject(root, sid, tumour_slices, d=20, s=8):
    import os
    os.makedirs(root / sid)
    lab = np.zeros((d, s, s), np.uint8)
    lab[list(tumour_slices), 2:4, 2:4] = 4
    np.save(root / sid / "label.npy", lab)
    np.save(root / sid / "image.npy", np.random.default_rng(0).standard_normal((d, 4, s, s)).astype(np.float16))


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.25, 0.5])
def test_balanced_sampling_count(tmp_path, rho):
    _write_subject(tmp_path, "a", range(3, 9))     # 6 tumour, 14 empty
    _write_subject(tmp_path, "b", range(0, 6))     # 6 tumour, 14 empty
    idx = balanced_index(str(tmp_path), ["a", "b"], rho, seed=1, log=lambda *_: None)
    n_t = 12
    assert len(idx) == n_t + {0.0: 0, 0.1: 1, 0.25: 4, 0.5: 12}[rho]
    assert len(set(idx)) == len(idx)
    assert idx == balanced_index(str(tmp_path), ["a", "b"], rho, seed=1, log=lambda *_: None)


def test_slice_dataset(tmp_path):
    _write_subject(tmp_path, "a", [5])
    ds = SliceDataset(str(tmp_path), [("a", 5), ("a", 0)], n_classes=3)
    x, y = ds[0]
    assert x.shape == (12, 8, 8) and y.shape == (3, 8, 8)
    assert y.sum() == 12 and ds[1][1].sum() == 0
    assert SliceDataset(str(tmp_path), [("a", 5)], n_classes=1)[0][1].shape == (1, 8, 8)


def test_augment_keeps_shapes_and_binary_masks():
    import torch
    from bt25d.augment import Augment
    torch.manual_seed(0)
    aug = Augment(p_flip=1.0, p_rot=1.0, p_gamma=1.0, p_noise=1.0)
    x = torch.randn(12, 32, 32); x[:, :4] = 0
    y = torch.zeros(3, 32, 32); y[:, 10:20, 10:20] = 1
    xa, ya = aug(x, y)
    assert xa.shape == x.shape and ya.shape == y.shape
    assert set(torch.unique(ya).tolist()) <= {0.0, 1.0}


def test_paper_slice_counts():
    # Section 3.3: 81,191 tumour-present slices, rho = 0.25 -> 27,063 empty slices, 108,254 in total
    assert n_empty_slices(81191, 0.25) == 27063
    assert 81191 + n_empty_slices(81191, 0.25) == 108254
