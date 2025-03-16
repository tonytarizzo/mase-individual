"""
FlexRound Example Script

This script demonstrates how to use FlexRound quantization method in MASE.
It shows both simple usage of the FlexRound class directly and integration with MASE.
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

# Force FlexRound registration with MASE
import temp.force_flexround_update

from src.chop.passes.graph.transforms.pruning.flexround import (
    FlexRound, FLEXROUND, FlexRoundParameterization
)

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
    print(f"Original weights min/max: {weights.min():.4f}/{weights.max():.4f}")
    print(f"Quantized weights min/max: {quantized_weights.min():.4f}/{quantized_weights.max():.4f}")
    
    # Calculate error
    error = (weights - quantized_weights).abs()
    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Max absolute error: {error.max():.6f}")
    
    # Plot histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(weights.flatten().numpy(), bins=100, alpha=0.7, label='Original')
    plt.hist(quantized_weights.flatten().numpy(), bins=100, alpha=0.7, label='Quantized')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(error.flatten().numpy(), bins=100)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Quantization Error')
    
    plt.tight_layout()
    plt.savefig("flexround_quantization.png")
    print(f"Saved plot to flexround_quantization.png")


def test_flexround_with_pruning():
    """
    Test FLEXROUND as a combined pruning and quantization method
    """
    print("\n")
    print("="*80)
    print("Testing FLEXROUND with pruning")
    print("="*80)
    
    # Create a random weight tensor
    torch.manual_seed(42)
    weights = torch.randn(32, 128)
    
    # Parameters
    sparsity = 0.5
    bit_width = 8
    frac_width = 4
    alpha = 0.5
    beta = 0.25
    structured_sparsity = True
    
    # Create FLEXROUND instance
    flexround = FLEXROUND(
        alpha=alpha, 
        beta=beta,
        bit_width=bit_width,
        frac_width=frac_width,
        structured_sparsity=structured_sparsity
    )
    
    # Apply pruning and quantization
    pruned_quantized, mask = flexround.prune_and_quantize_weights(weights, sparsity)
    
    # Print statistics
    print(f"Sparsity target: {sparsity:.2f}")
    print(f"Actual sparsity: {1 - (mask.sum() / mask.numel()):.4f}")
    
    # Count zeros before and after
    zeros_before = (weights == 0).sum().item()
    zeros_after = (pruned_quantized == 0).sum().item()
    print(f"Zeros before: {zeros_before}/{weights.numel()} ({zeros_before/weights.numel():.4f})")
    print(f"Zeros after: {zeros_after}/{weights.numel()} ({zeros_after/weights.numel():.4f})")
    
    # Calculate error on non-pruned elements
    non_pruned = mask.bool()
    error = (weights[non_pruned] - pruned_quantized[non_pruned]).abs()
    print(f"Mean absolute error on non-pruned elements: {error.mean():.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(weights.flatten().numpy(), bins=100, alpha=0.7, label='Original')
    plt.hist(pruned_quantized.flatten().numpy(), bins=100, alpha=0.7, label='Pruned+Quantized')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Plot the first row as a scatter plot
    row = 0
    plt.scatter(range(len(weights[row])), weights[row].numpy(), label='Original', alpha=0.7)
    plt.scatter(range(len(pruned_quantized[row])), pruned_quantized[row].numpy(), label='Pruned+Quantized', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Weight Value')
    plt.title(f'Row {row} Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("flexround_pruning.png")
    print(f"Saved plot to flexround_pruning.png")


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
        
        # Get the pruned and quantized weights
        final_weight = model.fc1.weight.clone()
        
        # Print statistics
        weight_before = original_fc1_weight
        weight_after = final_weight
        print(f"Weight shape: {weight_before.shape}")
        print(f"Original non-zeros: {(weight_before != 0).sum().item()}/{weight_before.numel()}")
        print(f"Final non-zeros: {(weight_after != 0).sum().item()}/{weight_after.numel()}")
        print(f"Effective sparsity: {1 - (weight_after != 0).sum().item() / weight_after.numel():.4f}")
        
        # Check if we maintained unique values (a sign of proper quantization)
        unique_before = torch.unique(weight_before).numel()
        unique_after = torch.unique(weight_after).numel()
        print(f"Unique values before: {unique_before}")
        print(f"Unique values after: {unique_after}")
        print(f"Quantization ratio: {unique_after / unique_before:.4f}")
        
        # Final output
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