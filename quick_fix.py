#!/usr/bin/env python
"""
Quick fix to add FlexRound to the weight_criteria_map in pruning_methods.py
"""

import os
import sys
import shutil

def find_chop_package():
    """Find the installed chop package path"""
    for path in sys.path:
        load_py = os.path.join(path, 'chop', 'passes', 'graph', 'transforms', 'pruning', 'load.py')
        if os.path.exists(load_py):
            return path
    return None

def fix_pruning_methods(chop_path):
    """Directly modify the weight_criteria_map in pruning_methods.py"""
    methods_py_path = os.path.join(chop_path, 'chop', 'passes', 'graph', 'transforms', 'pruning', 'pruning_methods.py')
    
    print(f"Directly modifying {methods_py_path}...")
    
    # Read the whole file content first
    with open(methods_py_path, 'r') as f:
        content = f.read()
        
    # Check if flexround_pruning is already defined
    if 'def flexround_pruning' in content:
        print("flexround_pruning function already exists, skipping function definition")
    else:
        # Define the flexround pruning function to add at the beginning of the file
        flexround_function = '''
# FlexRound pruning method for MASE
import torch

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
        # Add the function at the beginning of the file, after imports
        import_end = content.find('\n\n', content.find('import'))
        if import_end > 0:
            content = content[:import_end+2] + flexround_function + content[import_end+2:]
        else:
            # If can't find end of imports, add to the very beginning
            content = flexround_function + content
        
    # Update weight_criteria_map to include flexround if not already there
    if '"flexround": flexround_pruning' in content:
        print("flexround already in weight_criteria_map, no need to update")
    else:
        # Find weight_criteria_map and add flexround
        
        # Look for the elementwise section in the local section of weight_criteria_map
        elementwise_pos = content.find('"elementwise"', content.find('weight_criteria_map'))
        if elementwise_pos > 0:
            # Find the closing brace of the elementwise section
            open_braces = 1
            close_pos = elementwise_pos
            
            while open_braces > 0 and close_pos < len(content):
                close_pos += 1
                if content[close_pos] == '{':
                    open_braces += 1
                elif content[close_pos] == '}':
                    open_braces -= 1
                    
                    if open_braces == 1:  # Found the closing brace of elementwise
                        # Check if the entry ends with a comma
                        prev_char_pos = close_pos - 1
                        while prev_char_pos > 0 and content[prev_char_pos].isspace():
                            prev_char_pos -= 1
                            
                        indent = len(content[content.rfind('\n', 0, elementwise_pos)+1:elementwise_pos]) - len(content[content.rfind('\n', 0, elementwise_pos)+1:elementwise_pos].lstrip())
                        spaces = ' ' * (indent + 4)
                        
                        # Add flexround entry with proper indentation
                        flexround_entry = f'\n{spaces}"flexround": flexround_pruning,'
                        
                        if content[prev_char_pos] != ',':
                            # Add comma to previous entry
                            content = content[:prev_char_pos+1] + ',' + content[prev_char_pos+1:]
                            
                        # Insert the entry before the closing brace
                        content = content[:close_pos] + flexround_entry + content[close_pos:]
                        break
        else:
            print("Could not find elementwise section in weight_criteria_map")
            return False
    
    # Make a backup of the original file
    backup_path = methods_py_path + '.quickfix.bak'
    shutil.copy2(methods_py_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Write the updated content
    with open(methods_py_path, 'w') as f:
        f.write(content)
    
    print("Successfully fixed pruning_methods.py to include flexround in weight_criteria_map!")
    return True

def main():
    print("Quick fix for adding FlexRound to weight_criteria_map")
    
    # Find the chop package
    chop_path = find_chop_package()
    if not chop_path:
        print("Could not find the installed chop package.")
        sys.exit(1)
    
    print(f"Found chop package at {chop_path}")
    
    # Fix the pruning_methods.py file
    success = fix_pruning_methods(chop_path)
    
    if success:
        print("\nFix applied successfully!")
        print("You should now be able to use FlexRound with MASE.")
    else:
        print("\nFix application failed. Please check the file manually.")

if __name__ == "__main__":
    main() 