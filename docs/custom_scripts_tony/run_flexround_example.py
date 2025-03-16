#!/usr/bin/env python
"""
Run FlexRound Example with Quantization

This script ensures FlexRound is properly registered in the quantization framework
before running the flexround_example.py script.
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

# Import the force_flexround_update module to register FlexRound
try:
    from temp.force_flexround_update import *
except ImportError:
    # Try an alternate path
    sys.path.insert(0, os.path.join(mase_root, ".."))
    try:
        from temp.force_flexround_update import *
    except ImportError:
        print("Could not import force_flexround_update. Creating the registration directly...")
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
        
        print("\nFlexRound quantization method has been registered in the MASE framework.")

# Now run the example script
print("\n" + "="*80)
print("Running the FlexRound example script...")
print("="*80 + "\n")

# Import the flexround_example module
import importlib.util
example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flexround_example.py")
spec = importlib.util.spec_from_file_location("flexround_example", example_path)
flexround_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flexround_example)

# Run the main function
flexround_example.main()

print("\n" + "="*80)
print("FlexRound example completed.")
print("="*80) 