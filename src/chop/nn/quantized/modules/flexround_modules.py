import torch
import torch.nn as nn
import torch.nn.functional as F
from chop.passes.graph.transforms.quantize.flexround import FlexRoundQuantizer

###############################################################################
# 1. Straight-Through Estimator for FlexRound
###############################################################################
class FlexRoundSTEFunction(torch.autograd.Function):
    """
    Straight-through estimator for the floor(...) operation used by FlexRound.
    In forward: quant_out = s1 * floor(x / (s1 * S2 * s3) + 0.5)
    In backward: pass grad_input = grad_output (STE).
    """

    @staticmethod
    def forward(ctx, weight, s1, s2, s3):
        # Avoid division by zero
        div = s1 * s2 * s3 + 1e-12
        w_scaled = weight / div
        w_rounded = torch.floor(w_scaled + 0.5)  # standard "round to nearest"
        # Multiply back by s1 (like the equation in your code)
        wq = s1 * w_rounded
        return wq

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through => pass gradient to inputs, none to s1, s2, s3
        return grad_output, None, None, None


def flexround_ste(weight, s1, s2, s3):
    """
    Helper function calling the custom autograd function.
    """
    return FlexRoundSTEFunction.apply(weight, s1, s2, s3)


###############################################################################
# 2. Modules
###############################################################################
class LinearFlexRound(nn.Linear):
    """
    A linear layer wrapped with FlexRound quantization using STE.
    Weight quantization:
        Wq = s1 * floor( W / (s1 * S2 * s3 ) + 0.5 )
    where s1, S2, s3 are learnable parameters.
    Activation quantization is optional, using a simpler scale+round STE approach.
    """
    def __init__(self, in_features, out_features, bias=True, config=None):
        super().__init__(in_features, out_features, bias=bias)
        config = config or {}
        s1_init = config.get("s1_init", 1.0)

        # Learnable scale parameters for weight
        self.s1 = nn.Parameter(torch.tensor(s1_init, dtype=self.weight.dtype))
        self.S2 = nn.Parameter(torch.ones_like(self.weight))
        # s3 => shape is [out_features, 1] to scale each output channel
        self.s3 = nn.Parameter(torch.ones(self.out_features, 1, dtype=self.weight.dtype))

        self.weight_only = config.get("weight_only", False)
        
        # Activation quantizer (simple scale-round STE)
        if not self.weight_only:
            act_width = config.get("data_in_width", 8)
            act_frac = config.get("data_in_frac_width", 4)
            self.act_quant = FlexRoundQuantizer(bit_width=act_width, frac_width=act_frac)
        else:
            self.act_quant = None

    def forward(self, input):
        # Quantize weights using the STE function
        quant_w = flexround_ste(self.weight, self.s1, self.S2, self.s3)

        # F.linear with quantized weight
        out = F.linear(input, quant_w, self.bias)

        # Optionally quantize activations
        if self.act_quant is not None:
            out = self.act_quant(out)
        return out


class Conv2dFlexRound(nn.Conv2d):
    """
    A Conv2d layer wrapped with FlexRound quantization using STE.
    Weight quantization:
       Wq = s1 * floor( W / (s1 * S2 * s3 * s4) + 0.5 )
    s3 is per-out-channel scale, s4 is per-in-channel scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros",
                 config=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        config = config or {}
        s1_init = config.get("s1_init", 1.0)

        self.s1 = nn.Parameter(torch.tensor(s1_init, dtype=self.weight.dtype))
        self.S2 = nn.Parameter(torch.ones_like(self.weight))
        # s3 => shape: [out_channels, 1, 1, 1]
        self.s3 = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1, dtype=self.weight.dtype))
        # s4 => shape: [1, in_channels, 1, 1]
        self.s4 = nn.Parameter(torch.ones(1, self.in_channels, 1, 1, dtype=self.weight.dtype))

        self.weight_only = config.get("weight_only", False)
        if not self.weight_only:
            act_width = config.get("data_in_width", 8)
            act_frac = config.get("data_in_frac_width", 4)
            self.act_quant = FlexRoundQuantizer(bit_width=act_width, frac_width=act_frac)
        else:
            self.act_quant = None

    def forward(self, x):
        div_factor = self.s1 * self.S2 * self.s3 * self.s4 + 1e-12
        # Use the STE approach for 2D conv as well
        # Flatten the shape in a single consistent manner, then reshape
        # (But for simplicity we can just do a direct apply like so:)
        w_scaled = self.weight / div_factor
        w_floor = torch.floor(w_scaled + 0.5)
        quant_w = self.s1 * w_floor  # Wq

        out = self._conv_forward(x, quant_w, self.bias)

        if self.act_quant is not None:
            out = self.act_quant(out)
        return out


class Conv1dFlexRound(nn.Conv1d):
    """
    A Conv1d layer wrapped with FlexRound quantization using STE.
    Weight quantization:
        Wq = s1 * floor( W / (s1 * S2 * s3) + 0.5 )
    s3 => shape: [out_channels, 1, 1]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, config=None, **kwargs):
        kwargs.pop("padding_mode", None)  # remove unsupported arg
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, **kwargs)
        config = config or {}
        s1_init = config.get("s1_init", 1.0)

        self.s1 = nn.Parameter(torch.tensor(s1_init, dtype=self.weight.dtype))
        self.S2 = nn.Parameter(torch.ones_like(self.weight))
        self.s3 = nn.Parameter(torch.ones(self.out_channels, 1, 1, dtype=self.weight.dtype))

        self.weight_only = config.get("weight_only", False)
        if not self.weight_only:
            act_width = config.get("data_in_width", 8)
            act_frac = config.get("data_in_frac_width", 4)
            self.act_quant = FlexRoundQuantizer(bit_width=act_width, frac_width=act_frac)
        else:
            self.act_quant = None

    def forward(self, x):
        quant_w = flexround_ste(self.weight, self.s1, self.S2, self.s3)
        out = F.conv1d(x, quant_w, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        if self.act_quant is not None:
            out = self.act_quant(out)
        return out
