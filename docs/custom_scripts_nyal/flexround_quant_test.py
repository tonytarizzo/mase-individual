#!/usr/bin/env python
"""
FlexRound Pruning and Quantization Test

This script demonstrates the use of FlexRound for both pruning and quantization in MASE.
It applies FlexRound to all major layers in the Wav2Vec2 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCTC, Wav2Vec2Processor
import sys
import os
import re
import time

# Add MASE to path
mase_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, mase_root)

# Add temp directory to path
temp_path = os.path.join(mase_root, "temp")
sys.path.insert(0, temp_path)

# First, ensure FlexRound is registered in the quantization framework
print("="*80)
print("Registering FlexRound in the quantization framework...")
print("="*80)

# Import FlexRound registration module
import temp.force_flexround_update

# Now import MASE components
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_logger
from chop.passes.module import report_trainable_parameters_analysis_pass

# Set up logger
logger = get_logger(__name__)
logger.setLevel("INFO")

def count_nonzero_parameters(model):
    """Count the actual non-zero parameters in the model"""
    total_params = 0
    nonzero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and 'parametrizations' not in name:
            # Count total parameters
            total_params += param.numel()
            
            # Count non-zero parameters
            nonzero_params += (param != 0).sum().item()
                
    return total_params, nonzero_params

def print_parameter_count(model, description):
    """Helper function to count and print parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Also count non-zero parameters
    total, nonzero = count_nonzero_parameters(model)
    sparsity = 1.0 - (nonzero / total) if total > 0 else 0
    
    print(f"\n===== {description} =====")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total weight parameters: {total:,}")
    print(f"Non-zero weight parameters: {nonzero:,}")
    print(f"Sparsity: {sparsity:.2%}")
    
    return total_params, nonzero, sparsity

def apply_flexround_pruning_and_quantization(model):
    """Apply both FlexRound pruning and quantization to a model"""
    # Create a MASE graph
    mg = MaseGraph(model, hf_input_names=["input_values", "attention_mask"])
    mg, _ = passes.init_metadata_analysis_pass(mg)
    
    # Define dummy input for analysis pass
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    
    mg, _ = passes.add_common_metadata_analysis_pass(mg,
                                                  pass_args={
                                                      "dummy_in": dummy_in,
                                                      "add_value": True,
                                                      "force_device_meta": False,
                                                  })
    
    # Configure FlexRound pruning
    flexround_pruning_config = {
        "weight": {
            "sparsity": 0.1,           # 10% sparsity
            "method": "flexround",     # Use FlexRound method
            "scope": "local",          # Apply locally per layer
            "structured_sparsity": True,  # Use structured sparsity
            "alpha": 0.5,              # FlexRound parameters
            "beta": 0.25,
            "bit_width": 8,
            "frac_width": 4
        },
        "activation": {
            "sparsity": 0.0,           # No activation pruning
            "method": "random",
            "scope": "local",
        },
    }
    
    print("\n===== APPLYING FLEXROUND PRUNING =====")
    mg, _ = passes.prune_transform_pass(mg, pass_args=flexround_pruning_config)
    
    print("\n===== APPLYING FLEXROUND QUANTIZATION =====")
    
    # Define the list of target modules to quantize
    target_modules = [
        # Feature projection
        "feature_projection.projection",
        
        # All attention projections and MLP layers across all encoder layers
    ]
    
    # Add all encoder layers' attention and MLP modules
    for i in range(12):  # 12 encoder layers (0-11)
        # Attention projections
        target_modules.append(f"encoder.layers.{i}.attention.q_proj")
        target_modules.append(f"encoder.layers.{i}.attention.k_proj")
        target_modules.append(f"encoder.layers.{i}.attention.v_proj")
        target_modules.append(f"encoder.layers.{i}.attention.out_proj")
        
        # MLP layers
        target_modules.append(f"encoder.layers.{i}.feed_forward.intermediate_dense")
        target_modules.append(f"encoder.layers.{i}.feed_forward.output_dense")
    
    # Apply FlexRound quantization manually to the target layers
    print("Applying FlexRound quantization to all major layers...")
    quantized_count = 0
    start_time = time.time()
    
    # Create FlexRound quantizer
    from temp.force_flexround_update import FlexRoundWeight
    weight_quantizer = FlexRoundWeight(
        bit_width=8,
        frac_width=4,
        alpha=0.5,
        beta=0.25
    )
    
    # Pattern for matching target module names
    patterns = [re.compile(f"^{re.escape(module)}$") for module in target_modules]
    
    # Apply quantization to the target modules
    with torch.no_grad():
        for name, module in mg.model.named_modules():
            # Check if the module is one of our targets
            if any(pattern.match(name) for pattern in patterns) and hasattr(module, 'weight'):
                # Get the current weight
                original_weight = module.weight.data.clone()
                
                # Quantize the weight
                quantized_weight = weight_quantizer(original_weight)
                
                # Set the weight to the quantized value
                module.weight.data.copy_(quantized_weight)
                
                print(f"Quantized layer: {name}")
                quantized_count += 1
    
    end_time = time.time()
    print(f"\nSuccessfully quantized {quantized_count} layers in {end_time - start_time:.2f} seconds")
    
    # Count pruned parameters
    pruned_params = 0
    total_weight_params = 0
    
    for name, module in mg.model.named_modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            for p in module.parametrizations.weight:
                if hasattr(p, 'mask'):
                    # This is our pruning parametrization
                    weight_shape = module.weight.shape
                    total_in_layer = module.weight.numel()
                    nonzero_in_layer = p.mask.sum().item()
                    pruned_in_layer = total_in_layer - nonzero_in_layer
                    
                    print(f"Layer {name}: pruned {pruned_in_layer}/{total_in_layer} params ({pruned_in_layer/total_in_layer:.2%})")
                    
                    pruned_params += pruned_in_layer
                    total_weight_params += total_in_layer
    
    if total_weight_params > 0:
        overall_sparsity = pruned_params / total_weight_params
        print(f"\nOverall from FlexRound masks: {pruned_params}/{total_weight_params} params pruned ({overall_sparsity:.2%} sparsity)")
    
    # Test the model with a dummy input to make sure it works
    print("\n===== TESTING INFERENCE =====")
    try:
        with torch.no_grad():
            dummy_out = mg.model(**dummy_in)
            print("Inference successful!")
    except Exception as e:
        print(f"Inference error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
    
    return mg.model

def main():
    print("\n===== FlexRound Pruning and Quantization Example =====")
    
    # Load a pretrained model
    checkpoint = "facebook/wav2vec2-base-960h"
    model = AutoModelForCTC.from_pretrained(checkpoint)
    encoder = model.wav2vec2
    
    # Print initial parameter count
    before_params, before_nonzero, _ = print_parameter_count(encoder, "BEFORE PRUNING")
    
    # Apply FlexRound pruning and quantization
    pruned_model = apply_flexround_pruning_and_quantization(encoder)
    
    # Print parameter stats after pruning and quantization
    after_params, after_nonzero, after_sparsity = print_parameter_count(pruned_model, "AFTER PRUNING AND QUANTIZATION")
    
    # Calculate and print change in parameters
    print(f"\n===== PRUNING AND QUANTIZATION SUMMARY =====")
    print(f"Parameters before:            {before_params:,}")
    print(f"Non-zero params before:       {before_nonzero:,}")
    print(f"Non-zero params after:        {after_nonzero:,}")
    print(f"Reduction in parameters:      {before_nonzero - after_nonzero:,}")
    print(f"Overall sparsity achieved:    {after_sparsity:.2%}")
    
    return pruned_model

if __name__ == "__main__":
    main() 