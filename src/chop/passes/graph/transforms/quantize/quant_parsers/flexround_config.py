"""
FlexRound configuration for MASE quantization system
"""

from .parse_quant_config import QUANT_ARITH_ENTRIES, QUANT_ARITH_TO_CP_FN
from functools import partial
from .parse_quant_config import (
    cp_name, cp_bypass, cp_weight_entries, cp_data_in_entries,
    cp_bias_entries, cp_data_out_entries, cp_weight_entries_to_bias,
    cp_layer_entries
)

# Register FlexRound with the quantization system
if "flexround" not in QUANT_ARITH_ENTRIES:
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

# Print confirmation that FlexRound was registered
print("FlexRound quantization method registered with MASE.") 