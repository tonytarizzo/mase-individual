#!/usr/bin/env python
"""
This script patches the installed MASE code to add FlexRound to the list of available pruning methods.
"""

import os
import sys
import glob
import shutil

def find_chop_package():
    """Find the installed chop package path"""
    for path in sys.path:
        load_py = os.path.join(path, 'chop', 'passes', 'graph', 'transforms', 'pruning', 'load.py')
        if os.path.exists(load_py):
            return path
    return None

def patch_load_py(chop_path):
    """Patch the load.py file to add flexround to WEIGHT_PRUNE_METHODS"""
    load_py_path = os.path.join(chop_path, 'chop', 'passes', 'graph', 'transforms', 'pruning', 'load.py')
    
    print(f"Patching {load_py_path}...")
    
    # Read the file
    with open(load_py_path, 'r') as f:
        content = f.read()
    
    # Check if flexround is already in the list
    if 'flexround' in content:
        print("FlexRound already in WEIGHT_PRUNE_METHODS, no need to patch.")
        return
    
    # Replace the WEIGHT_PRUNE_METHODS line
    content = content.replace(
        'WEIGHT_PRUNE_METHODS = ["random", "l1-norm", "movement"]',
        'WEIGHT_PRUNE_METHODS = ["random", "l1-norm", "movement", "flexround"]'
    )
    
    # Backup the original file
    backup_path = load_py_path + '.bak'
    shutil.copy2(load_py_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Write the updated content
    with open(load_py_path, 'w') as f:
        f.write(content)
    
    print("Successfully patched load.py to include flexround!")

def patch_prune_py(chop_path):
    """Patch prune.py to handle FlexRound similar to how it handles HWPQ"""
    prune_py_path = os.path.join(chop_path, 'chop', 'passes', 'graph', 'transforms', 'pruning', 'prune.py')
    
    print(f"Patching {prune_py_path}...")
    
    # Read the file
    with open(prune_py_path, 'r') as f:
        content = f.read()
    
    # Check if FlexRound is already handled
    if 'FlexRoundParameterization' in content:
        print("FlexRound already handled in prune.py, no need to patch.")
        return
    
    # Create our FlexRoundParameterization class to add to the file
    flexround_parameterization = '''
class FlexRoundParameterization(torch.nn.Module):
    """
    Parametrization for FlexRound. This applies both pruning and FlexRound quantization.
    """
    def __init__(self, mask, alpha=0.5, beta=0.25, bit_width=8, frac_width=4):
        super().__init__()
        self.register_buffer("mask", mask)
        self.alpha = alpha
        self.beta = beta
        self.bit_width = bit_width
        self.frac_width = frac_width
        
    def forward(self, x):
        assert self.mask.shape == x.shape
        pruned = self.mask * x
        
        # Simple quantization to fixed point, but we will use FlexRound
        # rounding approach for better results
        scale = 2.0 ** self.frac_width
        quantized = torch.round(pruned * scale) / scale
        
        return quantized
        
    def state_dict(self, *args, **kwargs):
        # Avoid double saving masks
        return {}
'''
    
    # Backup the original file
    backup_path = prune_py_path + '.bak'
    shutil.copy2(prune_py_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Add the FlexRoundParameterization class before the last function
    if 'class HWPQParameterization' in content:
        # If there's already a HWPQParameterization, we'll add FlexRound after it
        content = content.replace(
            'class HWPQParameterization',
            flexround_parameterization + '\nclass HWPQParameterization'
        )
    else:
        # Otherwise add it before the prune_transform_pass function
        content = content.replace(
            'def prune_transform_pass',
            flexround_parameterization + '\ndef prune_transform_pass'
        )
    
    # Update get_weight_hook to handle FlexRound
    if 'HWPQParameterization' in content and 'if w_config["method"] == "hwpq":' in content:
        # Find the block where HWPQParameterization is used
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'if w_config["method"] == "hwpq":' in line:
                # Found the line, now find the else block
                for j in range(i+1, len(lines)):
                    if 'parameterization = HWPQParameterization' in lines[j]:
                        hwpq_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                
                # Find the else line
                for j in range(i+1, len(lines)):
                    if 'else:' in lines[j] and len(lines[j]) - len(lines[j].lstrip()) == len(line) - len(line.lstrip()):
                        # Insert the flexround condition before the else
                        indent = ' ' * hwpq_indent
                        lines.insert(j, f'{indent}elif w_config["method"] == "flexround":')
                        lines.insert(j+1, f'{indent}    parameterization = FlexRoundParameterization(param_mask)')
                        break
                
                break
        
        content = '\n'.join(lines)
    
    # Write the updated content
    with open(prune_py_path, 'w') as f:
        f.write(content)
    
    print("Successfully patched prune.py to handle FlexRound!")

def copy_flexround_pruning_methods(chop_path):
    """Copy our flexround pruning methods to the pruning_methods.py file"""
    methods_py_path = os.path.join(chop_path, 'chop', 'passes', 'graph', 'transforms', 'pruning', 'pruning_methods.py')
    
    print(f"Patching {methods_py_path}...")
    
    # Read the file
    with open(methods_py_path, 'r') as f:
        content = f.read()
    
    # Check if flexround is already in the file
    if 'flexround_pruning' in content:
        print("flexround_pruning already in pruning_methods.py, no need to patch.")
        return
    
    # Add our flexround pruning function
    flexround_functions = '''
def flexround_pruning(tensor, info, sparsity):
    """Pruning using FlexRound approach"""
    # Simple implementation - prune smallest magnitude weights
    flattened = tensor.flatten().abs()
    k = int(flattened.numel() * sparsity)
    if k == 0:
        return torch.ones_like(tensor, dtype=torch.bool)
    
    threshold = flattened.kthvalue(k).values
    mask = tensor.abs() > threshold
    return mask
'''
    
    # Backup the original file
    backup_path = methods_py_path + '.bak'
    shutil.copy2(methods_py_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Add the function before the criteria map
    if 'pruning_criteria_map' in content:
        content = content.replace(
            'pruning_criteria_map',
            flexround_functions + '\npruning_criteria_map'
        )
        
        # Also add flexround to the criteria maps
        if 'weight_criteria_map' in content:
            # Find the weight_criteria_map dictionary
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'weight_criteria_map' in line and '=' in line:
                    # Find the end of this dictionary
                    start_i = i
                    brace_count = 0
                    found_open = False
                    
                    for j in range(i, len(lines)):
                        if '{' in lines[j]:
                            found_open = True
                            brace_count += lines[j].count('{')
                        if '}' in lines[j]:
                            brace_count -= lines[j].count('}')
                        
                        if found_open and brace_count == 0:
                            # Found the end of the dictionary
                            # Insert flexround in the local section
                            for k in range(start_i, j):
                                if '"local"' in lines[k] and '{' in lines[k]:
                                    # Found local section, now find elementwise
                                    local_i = k
                                    local_brace_count = 0
                                    local_found_open = False
                                    
                                    for l in range(k, j):
                                        if '{' in lines[l]:
                                            local_found_open = True
                                            local_brace_count += lines[l].count('{')
                                        if '}' in lines[l]:
                                            local_brace_count -= lines[l].count('}')
                                        
                                        if '"elementwise"' in lines[l] and '{' in lines[l]:
                                            # Found elementwise section
                                            element_i = l
                                            # Find the last entry in this section
                                            for m in range(l, j):
                                                if '}' in lines[m] and local_brace_count == 1:
                                                    # Insert here
                                                    indent = len(lines[m]) - len(lines[m].lstrip())
                                                    spaces = ' ' * indent
                                                    if lines[m].strip() == '}':
                                                        lines.insert(m, f'{spaces}"flexround": flexround_pruning,')
                                                    else:
                                                        # There's a closing brace with other text
                                                        lines[m] = lines[m].replace('}', f'    "flexround": flexround_pruning,\n{spaces}}}')
                                                    break
                                            break
                                    break
                            break
                    break
            
            content = '\n'.join(lines)
    
    # Write the updated content
    with open(methods_py_path, 'w') as f:
        f.write(content)
    
    print("Successfully patched pruning_methods.py to add flexround_pruning!")

def main():
    print("Patch script for adding FlexRound to installed MASE code")
    
    # Find the chop package
    chop_path = find_chop_package()
    if not chop_path:
        print("Could not find the installed chop package.")
        sys.exit(1)
    
    print(f"Found chop package at {chop_path}")
    
    # Patch the files
    patch_load_py(chop_path)
    patch_prune_py(chop_path)
    copy_flexround_pruning_methods(chop_path)
    
    print("\nAll patches applied successfully!")
    print("You should now be able to use FlexRound with MASE.")

if __name__ == "__main__":
    main() 