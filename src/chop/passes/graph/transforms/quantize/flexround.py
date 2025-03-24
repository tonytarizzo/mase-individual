import torch
import math
import torch.nn as nn


###############################################################################
# 1. STE-based scale+round for activations
###############################################################################
class _FlexRoundActSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, min_val, max_val):
        # clamp to representable range
        x_clamped = torch.clamp(x, min_val, max_val)
        # scale and round
        x_scaled = x_clamped * scale
        x_rounded = torch.floor(x_scaled + 0.5)
        return x_rounded / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through pass
        return grad_output, None, None, None


def _flexround_act_quantize(x: torch.Tensor, bit_width=8, frac_width=4, symmetric=True):
    """
    Activation quantization: scale, round, clamp.
    With a straight-through estimator in backward pass.
    """
    # compute scale
    scale = 2.0 ** frac_width

    if symmetric:
        # e.g. -128..127 for 8 bits
        min_val = -2**(bit_width - 1) / scale
        max_val = (2**(bit_width - 1) - 1) / scale
    else:
        min_val = 0
        max_val = (2**bit_width - 1) / scale

    return _FlexRoundActSTEFunction.apply(x, scale, min_val, max_val)


###############################################################################
# 2. FlexRoundQuantizer: used for activation quantization in modules
###############################################################################
class FlexRoundQuantizer(nn.Module):
    """
    FlexRound quantization module for activation, using an STE for rounding.
    """
    def __init__(self, bit_width=8, frac_width=4, symmetric=True):
        super().__init__()
        self.bit_width = bit_width
        self.frac_width = frac_width
        self.symmetric = symmetric

    def forward(self, x):
        # If x is tiny or has very few elements, skip
        if x.numel() < 2:
            return x
        # Just call the STE-based function
        return _flexround_act_quantize(x, self.bit_width, self.frac_width, self.symmetric)

    def get_config(self):
        return {
            "bit_width": self.bit_width,
            "frac_width": self.frac_width,
            "symmetric": self.symmetric,
        }


###############################################################################
# 3. (Optional) transform pass for MASE
###############################################################################
def apply_flexround_transform(graph, config):
    """
    Called by MASE to insert forward hooks or to wrap certain modules.
    The details can remain similar to your original code. This function
    is an example if you want to do node-by-node hooking instead of
    module-level replacement. 
    """
    weight_bit_width = config.get("weight_bit_width", 8)
    weight_frac_width = config.get("weight_frac_width", 4)

    for node in graph.fx_graph.nodes:
        if node.op == 'call_module':
            module = graph.modules[node.target]
            module_name = node.target
            if hasattr(module, 'weight'):
                # Example: attach a forward_pre_hook that STE-rounds the weight
                pass  # see your original logic or keep it minimal

    return graph, {}
