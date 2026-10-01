"""Parameter and multiply-accumulate (MAC) counting with forward hooks.

MACs are counted for Conv2d, ConvTranspose2d and Linear layers (the layers
that dominate cost). FLOPs are reported as 2 x MACs; note that some tools
(e.g. thop) report MACs under the name "FLOPs", so always state which
convention a number uses.
"""

import torch
import torch.nn as nn


def count_macs(model: nn.Module, input_shape=(1, 12, 160, 160), device="cpu"):
    macs = [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        cin = m.in_channels // m.groups
        macs[0] += out.numel() * cin * k

    def convT_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        macs[0] += inp[0].numel() * (m.out_channels // m.groups) * k

    def lin_hook(m, inp, out):
        macs[0] += out.numel() * m.in_features

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.ConvTranspose2d):
            hooks.append(mod.register_forward_hook(convT_hook))
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs[0] // input_shape[0]


def summarize(model: nn.Module, input_shape=(1, 12, 160, 160), device="cpu"):
    params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, input_shape, device)
    return {"parameters": params, "parameters_M": params / 1e6, "GMACs": macs / 1e9, "GFLOPs": 2 * macs / 1e9}
