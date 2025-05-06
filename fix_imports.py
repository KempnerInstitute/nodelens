#!/usr/bin/env python3
"""
Script to fix import issues in the alignment package.

This script will:
1. Replace "from alignment.plotting import" with "from alignment.utils.plotting import"
2. Remove "from Code.alignment.src.alignment..." imports
"""

import os
import re
import sys
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

def fix_file(filepath):
    """Fix imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace imports
    new_content = content
    
    # Replace alignment.plotting with alignment.utils.plotting
    new_content = new_content.replace("from alignment.plotting import", "from alignment.utils.plotting import")
    
    # Replace Code.alignment.src.alignment imports
    new_content = re.sub(r'from Code\.alignment\.src\.alignment\.([\w_]+) import', r'from alignment.\1 import', new_content)
    
    # Only write the file if changes were made
    if new_content != content:
        print(f"Fixing imports in {filepath}")
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

def fix_imports_in_directory(directory):
    """Fix imports in all Python files in a directory (recursively)."""
    fixed_files = 0
    python_files = Path(directory).glob('**/*.py')
    
    for filepath in python_files:
        if fix_file(filepath):
            fixed_files += 1
    
    return fixed_files

if __name__ == "__main__":
    dir_to_fix = BASE_DIR / "src" / "alignment"
    
    print(f"Fixing imports in {dir_to_fix}")
    fixed = fix_imports_in_directory(dir_to_fix)
    print(f"Fixed imports in {fixed} files") 