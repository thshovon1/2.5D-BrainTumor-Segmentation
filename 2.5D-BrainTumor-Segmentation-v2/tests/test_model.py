import torch

from bt25d.baselines import attention_unet2d, unet2d
from bt25d.complexity import summarize
from bt25d.model import (DecoderFree25D, EncoderDecoder25D, build_model, count_parameters, infer_model_config,
                         read_checkpoint)


def test_decoder_free_outputs_and_size():
    m = DecoderFree25D(12, 3).eval()
    with torch.no_grad():
        p3, p2, p1 = m(torch.randn(2, 12, 160, 160))
    assert p3.shape == p2.shape == p1.shape == (2, 3, 160, 160)
    assert count_parameters(m) == 683_657


def test_state_dict_names_match_original_notebook():
    keys = set(DecoderFree25D(12, 1).state_dict())
    for k in ("e1.conv.0.weight", "e1.conv.1.running_mean", "e1.skip.weight", "e3.conv.3.weight",
              "att2.attn.0.weight", "att3.attn.0.bias", "aspp.conv1.weight", "aspp.conv2.weight",
              "aspp.conv3.weight", "aspp.out.weight", "ds1.weight", "ds2.weight", "out.weight"):
        assert k in keys, k


def test_encoder_decoder_single_output():
    m = EncoderDecoder25D(12, 3).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 12, 160, 160))
    assert len(out) == 1 and out[0].shape == (1, 3, 160, 160)


def test_complexity_counts():
    s = summarize(DecoderFree25D(12, 3))
    assert s["parameters"] == 683_657
    assert abs(s["GFLOPs"] - 2 * s["GMACs"]) < 1e-9
    assert 1.6 < s["GMACs"] < 1.8
    assert abs(summarize(unet2d())["parameters_M"] - 7.766) < 0.001
    assert abs(summarize(attention_unet2d())["parameters_M"] - 7.854) < 0.001


def test_checkpoint_roundtrip_with_prefix(tmp_path):
    m = DecoderFree25D(12, 3)
    torch.save({"model": {"module." + k: v for k, v in m.state_dict().items()}, "epoch": 3}, tmp_path / "c.pth")
    state, meta = read_checkpoint(str(tmp_path / "c.pth"))
    assert meta["epoch"] == 3
    cfg = infer_model_config(state)
    assert cfg == {"arch": "decoder_free", "in_channels": 12, "n_classes": 3, "widths": [32, 64, 128]}
    build_model(cfg).load_state_dict(state, strict=True)
    cfg_ed = infer_model_config(EncoderDecoder25D(12, 3).state_dict())
    assert cfg_ed["arch"] == "encoder_decoder"


def test_mac_counter_agrees_with_pytorch_flop_counter():
    from torch.utils.flop_counter import FlopCounterMode
    for m in (DecoderFree25D(12, 3), EncoderDecoder25D(12, 3), unet2d(), attention_unet2d()):
        m.eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(torch.zeros(1, 12, 160, 160))
        assert fc.get_total_flops() == 2 * summarize(m)["GMACs"] * 1e9
