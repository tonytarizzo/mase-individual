#!/usr/bin/env python
"""
Test FlexRound with Quantization

This script ensures FlexRound is properly registered in the quantization framework
before running the testing_flexround.py script.
"""

import sys
import os

# Add the parent directory to the path
mase_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mase_root not in sys.path:
    sys.path.insert(0, mase_root)

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

# Now run the testing script
print("\n" + "="*80)
print("Running the testing_flexround.py script...")
print("="*80 + "\n")

# Execute the testing script
import subprocess
test_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing_flexround.py")
subprocess.run([sys.executable, test_script_path], check=True)

print("\n" + "="*80)
print("Testing completed.")
print("="*80) 