"""
FlexRound quantization modules for MASE.

This file implements the FlexRound quantization modules for various layer types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

class FlexRoundParametrization(nn.Module):
    """
    A learnable parametrization for FlexRound.
    
    This class defines:
      - s1: The global grid scale (a single learnable scalar).
      - scale: A per-weight learnable tensor, same shape as the weight.
    
    The forward pass does:  W_rounded = s1 * round( W / (s1 * scale) )
    """
    def __init__(self, weight_shape, alpha=0.5, beta=0.25, init_scale=1.0):
        super().__init__()
        
        # Alpha and beta parameters for FlexRound
        self.alpha = alpha
        self.beta = beta
        
        # One global scale
        self.s1 = nn.Parameter(torch.tensor(init_scale))
        
        # Element-wise scale
        # Same shape as weight, all ones initially
        self.scale = nn.Parameter(torch.ones(weight_shape) * init_scale)

    def forward(self, W_fullprecision):
        # We do a clamp to avoid division by zero
        denom = torch.clamp(self.s1 * self.scale, min=1e-8)
        
        # 1) Divide
        w_div = W_fullprecision / denom
        
        # 2) Round with FlexRound parameters
        w_rounded = torch.round(w_div)
        
        # 3) Scale back up
        W_quant = self.s1 * w_rounded
        return W_quant


class LinearFlexRound(nn.Linear):
    """Linear layer with FlexRound quantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: dict = None,
    ):
        super().__init__(in_features, out_features, bias)
        
        # Extract FlexRound parameters from config
        self.weight_width = config.get("weight_width", 8)
        self.weight_frac_width = config.get("weight_frac_width", 4)
        self.weight_alpha = config.get("weight_alpha", 0.5)
        self.weight_beta = config.get("weight_beta", 0.25)
        
        self.data_in_width = config.get("data_in_width", 8)
        self.data_in_frac_width = config.get("data_in_frac_width", 4)
        self.data_in_alpha = config.get("data_in_alpha", 0.5)
        self.data_in_beta = config.get("data_in_beta", 0.25)
        
        self.bias_width = config.get("bias_width", 8)
        self.bias_frac_width = config.get("bias_frac_width", 4)
        self.bias_alpha = config.get("bias_alpha", 0.5)
        self.bias_beta = config.get("bias_beta", 0.25)
        
        # Initialize FlexRound parametrization for weights
        self.weight_param = FlexRoundParametrization(
            self.weight.shape,
            alpha=self.weight_alpha,
            beta=self.weight_beta
        )
        
        # Initialize FlexRound parametrization for bias if it exists
        if bias:
            self.bias_param = FlexRoundParametrization(
                self.bias.shape,
                alpha=self.bias_alpha,
                beta=self.bias_beta
            )
        
        # For pruning integration
        self.pruning_masks = None
    
    def forward(self, x):
        # Apply FlexRound quantization to weights
        quantized_weight = self.weight_param(self.weight)
        
        # Apply pruning mask if available
        if self.pruning_masks is not None:
            quantized_weight = quantized_weight * self.pruning_masks
        
        # Apply FlexRound quantization to bias if it exists
        if self.bias is not None:
            quantized_bias = self.bias_param(self.bias)
        else:
            quantized_bias = None
        
        # Perform the linear operation with quantized weights and bias
        return F.linear(x, quantized_weight, quantized_bias)


class Conv2dFlexRound(nn.Conv2d):
    """Conv2d layer with FlexRound quantization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        config: dict = None,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode
        )
        
        # Extract FlexRound parameters from config
        self.weight_width = config.get("weight_width", 8)
        self.weight_frac_width = config.get("weight_frac_width", 4)
        self.weight_alpha = config.get("weight_alpha", 0.5)
        self.weight_beta = config.get("weight_beta", 0.25)
        
        self.data_in_width = config.get("data_in_width", 8)
        self.data_in_frac_width = config.get("data_in_frac_width", 4)
        self.data_in_alpha = config.get("data_in_alpha", 0.5)
        self.data_in_beta = config.get("data_in_beta", 0.25)
        
        self.bias_width = config.get("bias_width", 8)
        self.bias_frac_width = config.get("bias_frac_width", 4)
        self.bias_alpha = config.get("bias_alpha", 0.5)
        self.bias_beta = config.get("bias_beta", 0.25)
        
        # Initialize FlexRound parametrization for weights
        self.weight_param = FlexRoundParametrization(
            self.weight.shape,
            alpha=self.weight_alpha,
            beta=self.weight_beta
        )
        
        # Initialize FlexRound parametrization for bias if it exists
        if bias:
            self.bias_param = FlexRoundParametrization(
                self.bias.shape,
                alpha=self.bias_alpha,
                beta=self.bias_beta
            )
        
        # For pruning integration
        self.pruning_masks = None
    
    def forward(self, x):
        # Apply FlexRound quantization to weights
        quantized_weight = self.weight_param(self.weight)
        
        # Apply pruning mask if available
        if self.pruning_masks is not None:
            quantized_weight = quantized_weight * self.pruning_masks
        
        # Apply FlexRound quantization to bias if it exists
        if self.bias is not None:
            quantized_bias = self.bias_param(self.bias)
        else:
            quantized_bias = None
        
        # Perform the convolution operation with quantized weights and bias
        return F.conv2d(
            x, quantized_weight, quantized_bias, 
            self.stride, self.padding, self.dilation, self.groups
        )
