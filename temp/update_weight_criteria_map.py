"""
Update Weight Criteria Map

This script updates the weight_criteria_map in pruning_methods.py to include the FlexRound method.
"""

import os
import sys
import importlib.util
import site

def update_weight_criteria_map():
    """Update the weight_criteria_map to include the FlexRound method."""
    # Path to the pruning_methods.py file in our workspace
    pruning_methods_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "chop", "passes", "graph", "transforms", "pruning", "pruning_methods.py"
    )
    
    if not os.path.exists(pruning_methods_path):
        print(f"Error: pruning_methods.py not found at {pruning_methods_path}")
        return False
    
    print(f"Found pruning_methods.py at {pruning_methods_path}")
    
    # Create a backup of the original file
    backup_path = pruning_methods_path + ".bak"
    try:
        with open(pruning_methods_path, 'r') as f:
            content = f.read()
        
        # Save backup
        with open(backup_path, 'w') as f:
            f.write(content)
        
        print(f"Created backup at {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Check if flexround_pruning is already imported
    if "from .flexround import flexround_pruning" not in content:
        # Add import for flexround_pruning
        import_pos = content.find("import torch")
        if import_pos == -1:
            print("Error: Could not find import section in pruning_methods.py")
            return False
        
        # Find the end of the import section
        import_end = content.find("\n\n", import_pos)
        if import_end == -1:
            import_end = content.find("\n", import_pos)
        
        # Add the import statement
        new_content = content[:import_end] + "\nfrom .flexround import flexround_pruning" + content[import_end:]
        content = new_content
    
    # Check if flexround is already in weight_criteria_map
    if '"flexround": flexround_pruning' in content:
        print("FlexRound already in weight_criteria_map, no need to update.")
        return True
    
    # Find the weight_criteria_map dictionary
    map_start = content.find("weight_criteria_map = {")
    if map_start == -1:
        print("Error: Could not find weight_criteria_map in pruning_methods.py")
        return False
    
    # Find the local elementwise section
    local_section = content.find('"local": {', map_start)
    if local_section == -1:
        print("Error: Could not find local section in weight_criteria_map")
        return False
    
    elementwise_section = content.find('"elementwise": {', local_section)
    if elementwise_section == -1:
        print("Error: Could not find elementwise section in weight_criteria_map")
        return False
    
    # Find the end of the elementwise dictionary
    elementwise_end = content.find("}", elementwise_section)
    if elementwise_end == -1:
        print("Error: Could not find end of elementwise section")
        return False
    
    # Add flexround to the elementwise dictionary
    new_content = content[:elementwise_end] + ',\n            "flexround": flexround_pruning' + content[elementwise_end:]
    
    # Write the updated content back to the file
    try:
        with open(pruning_methods_path, 'w') as f:
            f.write(new_content)
        print(f"Updated {pruning_methods_path} to include FlexRound in weight_criteria_map")
        return True
    except Exception as e:
        print(f"Error writing to {pruning_methods_path}: {e}")
        # Restore from backup
        try:
            with open(backup_path, 'r') as f:
                original_content = f.read()
            with open(pruning_methods_path, 'w') as f:
                f.write(original_content)
            print(f"Restored {pruning_methods_path} from backup")
        except Exception as e2:
            print(f"Error restoring from backup: {e2}")
        return False

def main():
    # Update weight_criteria_map
    if update_weight_criteria_map():
        print("Successfully updated weight_criteria_map to include FlexRound.")
        
        # Now reinstall the package to apply the changes
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"Reinstalling MASE from: {root_dir}")
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", root_dir], check=True)
            print("Successfully reinstalled MASE.")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"Error reinstalling MASE: {e}")
            return 1
    else:
        print("Failed to update weight_criteria_map.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 