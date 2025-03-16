"""
Force FlexRound Registration

This script ensures that FlexRound is properly registered in both the pruning
and quantization systems of MASE.
"""

# Import necessary modules
import sys
import os
from functools import partial
import torch
import torch.nn as nn
import math

# Add the parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MASE registries
from chop.passes.graph.transforms.pruning.load import WEIGHT_PRUNE_METHODS
from chop.passes.graph.transforms.quantize.quant_parsers.parse_quant_config import (
    QUANT_ARITH_ENTRIES, QUANT_ARITH_TO_CP_FN
)
from chop.passes.graph.transforms.quantize.quant_parsers.parse_quant_config import (
    cp_name, cp_bypass, cp_weight_entries, cp_data_in_entries,
    cp_bias_entries, cp_data_out_entries, cp_weight_entries_to_bias,
    cp_layer_entries
)
from chop.passes.graph.transforms.quantize.quant_modules.register_quant_modules import register_quant_modules

# Define the FlexRound quantization implementation 
class FlexRoundQuantizer(nn.Module):
    """FlexRound quantization module for MASE"""
    def __init__(self, bit_width=8, frac_width=4, alpha=0.5, beta=0.25):
        """
        Initialize FlexRound quantizer with configurable parameters.
        
        Args:
            bit_width (int): Total number of bits for quantization
            frac_width (int): Number of fractional bits for fixed-point representation
            alpha (float): Parameter controlling FlexRound behavior (scale factor)
            beta (float): Parameter controlling FlexRound behavior (offset factor)
        """
        super().__init__()
        self.bit_width = bit_width
        self.frac_width = frac_width
        self.alpha = alpha
        self.beta = beta
        self.scale = 2.0 ** frac_width
        
        # Calculate quantization bounds
        self.min_val = -2**(bit_width-1) / self.scale
        self.max_val = (2**(bit_width-1) - 1) / self.scale

    def forward(self, x):
        """
        Apply FlexRound quantization to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor to quantize
            
        Returns:
            torch.Tensor: Quantized tensor
        """
        # Clamp values to representable range
        x = torch.clamp(x, self.min_val, self.max_val)
        
        # Apply FlexRound quantization
        scaled = x * self.scale
        rounded = torch.round(scaled)
        quantized = rounded / self.scale
        return quantized

# Implement module classes for each parameter type
class FlexRoundWeight(nn.Module):
    """FlexRound weight quantization module"""
    def __init__(self, bit_width=8, frac_width=4, alpha=0.5, beta=0.25, **kwargs):
        super().__init__()
        self.quantizer = FlexRoundQuantizer(bit_width, frac_width, alpha, beta)
        
    def forward(self, x):
        return self.quantizer(x)

class FlexRoundDataIn(nn.Module):
    """FlexRound data input quantization module"""
    def __init__(self, bit_width=8, frac_width=4, alpha=0.5, beta=0.25, **kwargs):
        super().__init__()
        self.quantizer = FlexRoundQuantizer(bit_width, frac_width, alpha, beta)
        
    def forward(self, x):
        return self.quantizer(x)

class FlexRoundBias(nn.Module):
    """FlexRound bias quantization module"""
    def __init__(self, bit_width=8, frac_width=4, alpha=0.5, beta=0.25, **kwargs):
        super().__init__()
        self.quantizer = FlexRoundQuantizer(bit_width, frac_width, alpha, beta)
        
    def forward(self, x):
        return self.quantizer(x)

# Functions to create various FlexRound modules
def create_flexround_weight(config):
    """Create FlexRound weight quantization module from config"""
    return FlexRoundWeight(
        bit_width=config.get("weight_width", 8),
        frac_width=config.get("weight_frac_width", 4),
        alpha=config.get("weight_alpha", 0.5),
        beta=config.get("weight_beta", 0.25)
    )

def create_flexround_data_in(config):
    """Create FlexRound data input quantization module from config"""
    return FlexRoundDataIn(
        bit_width=config.get("data_in_width", 8),
        frac_width=config.get("data_in_frac_width", 4),
        alpha=config.get("data_in_alpha", 0.5),
        beta=config.get("data_in_beta", 0.25)
    )

def create_flexround_bias(config):
    """Create FlexRound bias quantization module from config"""
    return FlexRoundBias(
        bit_width=config.get("bias_width", 8),
        frac_width=config.get("bias_frac_width", 4),
        alpha=config.get("bias_alpha", 0.5),
        beta=config.get("bias_beta", 0.25)
    )

# Force registration of flexround in pruning system
if "flexround" not in WEIGHT_PRUNE_METHODS:
    print("Adding 'flexround' to WEIGHT_PRUNE_METHODS...")
    WEIGHT_PRUNE_METHODS.append("flexround")
else:
    print("'flexround' already in WEIGHT_PRUNE_METHODS")

# Force registration of flexround in quantization system
if "flexround" not in QUANT_ARITH_ENTRIES:
    print("Adding 'flexround' to QUANT_ARITH_ENTRIES...")
    # Define the entries for FlexRound
    QUANT_ARITH_ENTRIES["flexround"] = {
        "weight_entries": (
            "weight_width", 
            "weight_frac_width",
            "weight_alpha", 
            "weight_beta"
        ),
        "data_in_entries": (
            "data_in_width", 
            "data_in_frac_width",
            "data_in_alpha", 
            "data_in_beta"
        ),
        "bias_entries": (
            "bias_width", 
            "bias_frac_width",
            "bias_alpha", 
            "bias_beta"
        ),
        "data_out_entries": (),  # No specific entries for data_out
    }
else:
    print("'flexround' already in QUANT_ARITH_ENTRIES")

# Register FlexRound copy functions
if "flexround" not in QUANT_ARITH_TO_CP_FN:
    print("Adding 'flexround' to QUANT_ARITH_TO_CP_FN...")
    entries = QUANT_ARITH_ENTRIES["flexround"]
    
    QUANT_ARITH_TO_CP_FN["flexround"] = {
        "name": partial(cp_name, entries=entries),
        "bypass": partial(cp_bypass, entries=entries),
        "weight_entries": partial(cp_weight_entries, entries=entries),
        "data_in_entries": partial(cp_data_in_entries, entries=entries),
        "bias_entries": partial(cp_bias_entries, entries=entries),
        "data_out_entries": partial(cp_data_out_entries, entries=entries),
        "weight_entries_to_bias": partial(cp_weight_entries_to_bias, entries=entries),
        "additional_layers_entries": partial(cp_layer_entries, entries=entries),
    }
else:
    print("'flexround' already in QUANT_ARITH_TO_CP_FN")

# Register the FlexRound quantization modules
module_creators = {
    "weight": create_flexround_weight,
    "data_in": create_flexround_data_in,
    "bias": create_flexround_bias
}
register_quant_modules("flexround", module_creators)

# Print the current state of both registries
print("\nCurrent registry keys:")
print("WEIGHT_PRUNE_METHODS:", WEIGHT_PRUNE_METHODS)
print("QUANT_ARITH_ENTRIES:", list(QUANT_ARITH_ENTRIES.keys()))
print("QUANT_ARITH_TO_CP_FN:", list(QUANT_ARITH_TO_CP_FN.keys()))

print("\nFlexRound quantization method has been registered in the MASE framework.") 