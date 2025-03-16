#!/usr/bin/env python
"""
Fixed FlexRound Example Script

This is a fixed version of the flexround_example.py script that ensures FlexRound
is properly registered with the quantization framework before running the examples.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add MASE to path if not already added
mase_root = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(mase_root))

# First, ensure FlexRound is registered in the quantization framework
print("="*80)
print("Registering FlexRound in the quantization framework...")
print("="*80)

# Register FlexRound directly
from functools import partial

# Import the required modules
from chop.passes.graph.transforms.quantize.quant_parsers.parse_quant_config import (
    QUANT_ARITH_ENTRIES, QUANT_ARITH_TO_CP_FN,
    cp_name, cp_bypass, cp_weight_entries, cp_data_in_entries,
    cp_bias_entries, cp_data_out_entries, cp_weight_entries_to_bias,
    cp_layer_entries
)

# Register FlexRound in quantization system
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

print("\nFlexRound quantization method has been registered in the MASE framework.")

# Now import the FlexRound components
from src.chop.passes.graph.transforms.pruning.flexround import (
    FlexRound, FLEXROUND, FlexRoundParameterization
)

# ---------------------------------------------------------------------------------
# The rest of the script is the same as flexround_example.py
# ---------------------------------------------------------------------------------

def test_flexround_standalone():
    """
    Test the FlexRound implementation standalone without MASE
    """
    print("="*80)
    print("Testing FlexRound standalone")
    print("="*80)
    
    # Create a random weight tensor
    torch.manual_seed(42)
    weights = torch.randn(32, 64)
    
    # Parameters for FlexRound
    bit_width = 8
    frac_width = 4
    alpha = 0.5
    beta = 0.25
    
    # Create FlexRound instance
    flexround = FlexRound(
        bit_width=bit_width,
        frac_width=frac_width,
        alpha=alpha,
        beta=beta
    )
    
    # Quantize the weights
    quantized_weights = flexround.quantize(weights)
    
    # Print statistics
    print(f"Original weights shape: {weights.shape}")
    print(f"Quantized weights shape: {quantized_weights.shape}")
    print(f"Original weights min/max: {weights.min().item():.4f}/{weights.max().item():.4f}")
    print(f"Quantized weights min/max: {quantized_weights.min().item():.4f}/{quantized_weights.max().item():.4f}")
    
    # Calculate error metrics
    mae = torch.abs(weights - quantized_weights).mean().item()
    max_error = torch.abs(weights - quantized_weights).max().item()
    print(f"Mean absolute error: {mae:.6f}")
    print(f"Max absolute error: {max_error:.6f}")
    
    # Plot the distribution of original vs quantized weights
    plt.figure(figsize=(10, 6))
    plt.hist(weights.flatten().numpy(), bins=100, alpha=0.5, label='Original')
    plt.hist(quantized_weights.flatten().numpy(), bins=100, alpha=0.5, label='Quantized')
    plt.legend()
    plt.title(f'FlexRound Quantization (bit_width={bit_width}, frac_width={frac_width}, α={alpha}, β={beta})')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.savefig('flexround_quantization.png')
    print(f"Saved plot to flexround_quantization.png")
    plt.close()

def test_flexround_with_pruning():
    """
    Test the FlexRound combined pruning and quantization
    """
    print("\n")
    print("="*80)
    print("Testing FLEXROUND with pruning")
    print("="*80)
    
    # Create random weights
    torch.manual_seed(0)
    weights = torch.randn(32, 128)
    
    # Create FLEXROUND instance with structured sparsity
    quantizer = FLEXROUND(
        bit_width=8,
        frac_width=4,
        alpha=0.5,
        beta=0.25,
        structured_sparsity=True
    )
    
    # Set target sparsity
    sparsity = 0.5
    
    # Prune and quantize the weights
    pruned_weights, mask = quantizer.prune_and_quantize_weights(weights, sparsity)
    
    # Calculate sparsity
    total_weights = weights.numel()
    zeros_before = (weights == 0).sum().item()
    zeros_after = (pruned_weights == 0).sum().item()
    
    actual_sparsity = zeros_after / total_weights
    
    print(f"Sparsity target: {sparsity:.2f}")
    print(f"Actual sparsity: {actual_sparsity:.4f}")
    print(f"Zeros before: {zeros_before}/{total_weights} ({zeros_before/total_weights:.4f})")
    print(f"Zeros after: {zeros_after}/{total_weights} ({zeros_after/total_weights:.4f})")
    
    # Calculate quantization error on non-pruned elements
    error = torch.abs(weights - pruned_weights)
    error_on_nonzero = error[pruned_weights != 0].mean().item()
    print(f"Mean absolute error on non-pruned elements: {error_on_nonzero:.6f}")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Original vs pruned weights as scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(weights.flatten().numpy(), pruned_weights.flatten().numpy(), alpha=0.5, s=1)
    max_val = max(weights.abs().max().item(), pruned_weights.abs().max().item())
    plt.plot([-max_val, max_val], [-max_val, max_val], 'r--')
    plt.title('Original vs Pruned+Quantized')
    plt.xlabel('Original Weights')
    plt.ylabel('Pruned+Quantized Weights')
    plt.grid(True, alpha=0.3)
    
    # Histogram of weights
    plt.subplot(1, 2, 2)
    plt.hist(weights.flatten().numpy(), bins=100, alpha=0.5, label='Original')
    plt.hist(pruned_weights.flatten().numpy(), bins=100, alpha=0.5, label='Pruned+Quantized')
    plt.title(f'Weight Distribution (Sparsity={actual_sparsity:.2f})')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flexround_pruning.png')
    print(f"Saved plot to flexround_pruning.png")
    plt.close()

def test_mase_integration():
    """
    Test FlexRound integration with MASE using a simple model
    """
    print("\n")
    print("="*80)
    print("Testing FlexRound MASE integration")
    print("="*80)
    
    # Create a toy dataset
    torch.manual_seed(0)
    x = torch.randn(100, 10)
    y = torch.randn(100, 2)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 2)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Create and train a model
    model = SimpleModel()
    
    # Create a dataloader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train the model for a few steps
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training model for 10 steps...")
    for epoch in range(10):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Save original weights
    original_fc1_weight = model.fc1.weight.clone()
    
    # Apply FlexRound to the model
    print("\nApplying FlexRound to the model...")
    
    # Apply FlexRound pruning (like MASE would do)
    info = {
        "structured_sparsity": True,
        "alpha": 0.5,
        "beta": 0.25,
        "bit_width": 8,
        "frac_width": 4,
    }
    
    from src.chop.passes.graph.transforms.pruning.flexround import flexround_pruning
    sparsity = 0.5
    
    # Get the mask
    with torch.no_grad():
        mask = flexround_pruning(model.fc1.weight, info, sparsity)
        
        # Apply the mask directly first
        model.fc1.weight.data *= mask
        
        # Now apply parametrization
        model.fc1 = torch.nn.utils.parametrize.register_parametrization(
            model.fc1, 
            "weight", 
            FlexRoundParameterization(
                mask,
                alpha=info["alpha"],
                beta=info["beta"],
                bit_width=info["bit_width"],
                frac_width=info["frac_width"]
            )
        )
    
    # Test the model with FlexRound
    print("\nTesting model with FlexRound...")
    with torch.no_grad():
        test_x = torch.randn(10, 10)
        out_original = model(test_x)
        
        # Remove parametrization to check the quantized weights
        if torch.nn.utils.parametrize.is_parametrized(model.fc1, "weight"):
            torch.nn.utils.parametrize.remove_parametrizations(model.fc1, "weight")
        
        # Print statistics about the pruned and quantized weights
        quantized_weight = model.fc1.weight
        print(f"Weight shape: {quantized_weight.shape}")
        
        nonzeros_original = (original_fc1_weight != 0).sum().item()
        nonzeros_pruned = (quantized_weight != 0).sum().item()
        
        total_params = original_fc1_weight.numel()
        print(f"Original non-zeros: {nonzeros_original}/{total_params}")
        print(f"Final non-zeros: {nonzeros_pruned}/{total_params}")
        print(f"Effective sparsity: {1 - nonzeros_pruned/total_params:.4f}")
        
        # Count unique values to measure quantization effect
        unique_values_original = torch.unique(original_fc1_weight).numel()
        unique_values_pruned = torch.unique(quantized_weight).numel()
        print(f"Unique values before: {unique_values_original}")
        print(f"Unique values after: {unique_values_pruned}")
        print(f"Quantization ratio: {unique_values_pruned/unique_values_original:.4f}")
        
        # Run the model again to check the error
        out_pruned = model(test_x)
        error = torch.abs(out_original - out_pruned).mean()
        print(f"Mean output error after pruning: {error:.6f}")


def get_temp_mase_config_with_flexround():
    """
    Generate a temporary MASE config that includes FlexRound
    """
    print("\n")
    print("="*80)
    print("Generating MASE config with FlexRound")
    print("="*80)
    
    config = {
        "defaults": {
            "linear": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "weight_alpha": 0.5,
                "weight_beta": 0.25,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "data_in_alpha": 0.5,
                "data_in_beta": 0.25,
                "bias_width": 8,
                "bias_frac_width": 4,
                "bias_alpha": 0.5,
                "bias_beta": 0.25,
            },
            "conv2d": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "weight_alpha": 0.5,
                "weight_beta": 0.25,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "data_in_alpha": 0.5,
                "data_in_beta": 0.25,
                "bias_width": 8,
                "bias_frac_width": 4,
                "bias_alpha": 0.5,
                "bias_beta": 0.25,
            }
        }
    }
    
    # Save to a temporary file
    import json
    config_path = "flexround_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved FlexRound MASE config to {config_path}")
    return config_path


def main():
    """Main function"""
    print("FlexRound Example Script")
    
    # Make sure the prints are not buffered
    sys.stdout.flush()
    
    # Test FlexRound standalone
    test_flexround_standalone()
    
    # Test FLEXROUND with pruning
    test_flexround_with_pruning()
    
    # Test MASE integration
    test_mase_integration()
    
    # Generate MASE config
    config_path = get_temp_mase_config_with_flexround()
    
    print("\nDone!")
    print("""
Next steps:
1. You can use the generated config file with MASE's quantize pass.
2. For pruning, use the "flexround" method in the pruning pass.
3. To activate both pruning and quantization, use both passes in sequence.
""")


if __name__ == "__main__":
    main() 