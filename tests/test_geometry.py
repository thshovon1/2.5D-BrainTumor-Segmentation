import numpy as np
import pytest

from bt25d.geometry import image_to_model_grid, label_to_model_grid, to_native


@pytest.mark.parametrize("hw", [(240, 240), (150, 170)])
def test_crop_roundtrip(hw):
    rng = np.random.default_rng(0)
    p = rng.random((3, 2, 160, 160)).astype(np.float32)
    nat = to_native(p, hw, "crop")
    assert nat.shape == (3, 2) + hw
    back = image_to_model_grid(nat, 160, "crop")
    # the central window survives the round trip
    h, w = hw
    ch, cw = min(h, 160), min(w, 160)
    oh, ow = (160 - ch) // 2, (160 - cw) // 2
    np.testing.assert_allclose(back[:, :, oh:oh + ch, ow:ow + cw], p[:, :, oh:oh + ch, ow:ow + cw])


def test_resize_shapes_and_constants():
    vol = np.full((4, 3, 240, 240), 2.5, np.float32)
    out = image_to_model_grid(vol, 160, "resize")
    assert out.shape == (4, 3, 160, 160)
    np.testing.assert_allclose(out, 2.5, atol=1e-5)
    back = to_native(out[:3], (240, 240), "resize")
    assert back.shape == (3, 3, 240, 240)
    np.testing.assert_allclose(back, 2.5, atol=1e-5)


def test_label_resize_keeps_label_values():
    lab = np.zeros((2, 240, 240), np.uint8)
    lab[:, 100:140, 90:150] = 2; lab[:, 110:130, 100:120] = 1; lab[:, 115:125, 105:115] = 4
    out = label_to_model_grid(lab, 160, "resize")
    assert out.shape == (2, 160, 160)
    assert set(np.unique(out)) <= {0, 1, 2, 4}
    assert abs((out > 0).sum() / (lab > 0).sum() - (160 / 240) ** 2) < 0.02
