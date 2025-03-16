"""
Install FlexRound Modules using pip install -e

This script creates the FlexRound module files and then reinstalls the MASE package
using pip install -e to ensure the changes are properly applied.
"""

import os
import sys
import subprocess

def create_flexround_module():
    """Create the FlexRound module file."""
    # Define the directory path
    modules_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "..", "src", "chop", "nn", "quantized", "modules")
    
    # Ensure the directory exists
    os.makedirs(modules_dir, exist_ok=True)
    
    # Define the file path
    flexround_path = os.path.join(modules_dir, "flexround.py")
    
    # Define the content of the flexround.py file
    flexround_content = '''"""
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
'''
    
    # Write the content to the file
    with open(flexround_path, 'w') as f:
        f.write(flexround_content)
    
    print(f"Created FlexRound module at: {flexround_path}")
    
    # Update the __init__.py file to include FlexRound modules
    init_path = os.path.join(modules_dir, "__init__.py")
    
    # Read the current __init__.py
    with open(init_path, 'r') as f:
        init_content = f.read()
    
    # Check if FlexRound is already imported
    if "from .flexround import" not in init_content:
        # Add import statement for FlexRound modules
        import_statement = "\n# Import FlexRound modules\nfrom .flexround import (\n    LinearFlexRound,\n    Conv2dFlexRound,\n)\n"
        
        # Find the position to insert the import statement (after the last import)
        last_import_pos = init_content.rfind("from")
        last_import_end = init_content.find(")", last_import_pos)
        if last_import_end == -1:
            last_import_end = init_content.find("\n\n", last_import_pos)
        
        if last_import_end != -1:
            # Insert the import statement after the last import
            init_content = init_content[:last_import_end+1] + import_statement + init_content[last_import_end+1:]
        else:
            # If we can't find a good position, just append it
            init_content += import_statement
    
    # Check if FlexRound modules are already in the map
    if "linear_flexround" not in init_content:
        # Add FlexRound modules to the quantized_module_map
        map_entries = [
            '    "conv2d_flexround": Conv2dFlexRound,',
            '    "linear_flexround": LinearFlexRound,'
        ]
        
        # Find positions to insert the map entries
        conv2d_pos = init_content.find('"conv2d_logicnets"')
        linear_pos = init_content.find('"linear_logicnets"')
        
        if conv2d_pos != -1 and linear_pos != -1:
            # Find the end of these lines
            conv2d_end = init_content.find("\n", conv2d_pos)
            linear_end = init_content.find("\n", linear_pos)
            
            # Insert the map entries
            init_content = init_content[:conv2d_end+1] + map_entries[0] + "\n" + init_content[conv2d_end+1:]
            # Recalculate linear_end after the first insertion
            linear_end = init_content.find("\n", init_content.find('"linear_logicnets"'))
            init_content = init_content[:linear_end+1] + map_entries[1] + "\n" + init_content[linear_end+1:]
        else:
            print("Warning: Could not find appropriate positions to insert map entries.")
            print("You may need to manually add the following entries to quantized_module_map:")
            for entry in map_entries:
                print(entry)
    
    # Write the updated __init__.py
    with open(init_path, 'w') as f:
        f.write(init_content)
    
    print(f"Updated {init_path} to include FlexRound modules")
    return True

def reinstall_mase():
    """Reinstall the MASE package using pip install -e."""
    # Get the root directory of the MASE project
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Run pip install -e
    print(f"Reinstalling MASE from: {root_dir}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", root_dir], check=True)
        print("Successfully reinstalled MASE.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error reinstalling MASE: {e}")
        return False

def main():
    # Create the FlexRound module
    if not create_flexround_module():
        print("Failed to create FlexRound module.")
        return 1
    
    # Reinstall MASE
    if not reinstall_mase():
        print("Failed to reinstall MASE.")
        return 1
    
    print("Successfully installed FlexRound modules into the MASE framework.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 