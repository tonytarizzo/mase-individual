"""
Install FlexRound Modules

This script installs the FlexRound quantization modules into the MASE framework.
It copies the module files to the appropriate locations in the installed package.
"""

import os
import sys
import shutil
import importlib.util
import site

def find_chop_package():
    """Find the installed chop package directory."""
    # Try to find the chop package in site-packages
    for site_dir in site.getsitepackages():
        chop_path = os.path.join(site_dir, "chop")
        if os.path.exists(chop_path):
            return chop_path
    
    # If not found in site-packages, try to find it in the current environment
    try:
        chop_spec = importlib.util.find_spec("chop")
        if chop_spec and chop_spec.origin:
            return os.path.dirname(os.path.dirname(chop_spec.origin))
    except ImportError:
        pass
    
    return None

def install_flexround_modules(chop_path):
    """Install FlexRound modules into the MASE framework."""
    if not chop_path:
        print("Error: Could not find the chop package.")
        return False
    
    # Source files
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "chop")
    flexround_module_path = os.path.join(src_dir, "nn", "quantized", "modules", "flexround.py")
    
    # Destination directories
    dest_modules_dir = os.path.join(chop_path, "nn", "quantized", "modules")
    
    # Check if source files exist
    if not os.path.exists(flexround_module_path):
        print(f"Error: Source file {flexround_module_path} does not exist.")
        return False
    
    # Check if destination directories exist
    if not os.path.exists(dest_modules_dir):
        print(f"Error: Destination directory {dest_modules_dir} does not exist.")
        return False
    
    # Copy flexround.py to modules directory
    dest_flexround_path = os.path.join(dest_modules_dir, "flexround.py")
    try:
        shutil.copy2(flexround_module_path, dest_flexround_path)
        print(f"Copied {flexround_module_path} to {dest_flexround_path}")
    except Exception as e:
        print(f"Error copying {flexround_module_path}: {e}")
        return False
    
    # Update __init__.py to include FlexRound modules
    init_path = os.path.join(dest_modules_dir, "__init__.py")
    if not os.path.exists(init_path):
        print(f"Error: __init__.py not found at {init_path}")
        return False
    
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

def main():
    # Find the chop package
    chop_path = find_chop_package()
    if not chop_path:
        print("Error: Could not find the chop package.")
        return 1
    
    print(f"Found chop package at: {chop_path}")
    
    # Install FlexRound modules
    if install_flexround_modules(chop_path):
        print("Successfully installed FlexRound modules into the MASE framework.")
        return 0
    else:
        print("Failed to install FlexRound modules.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 