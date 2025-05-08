#!/usr/bin/env python

"""
Quick script to fix indentation in alignment_experiments.py
"""

import os
import sys

def fix_indentation():
    # Path to the file with indentation issue
    file_path = "src/alignment/experiments/alignment_experiments.py"
    
    # Read the file content
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Lines to look for and fix
    line_to_check = "            logger.info(f\"Using dataset: {dataset_config.get('name', 'unknown')}"
    
    # Fix indentation by adjusting it
    fixed_lines = []
    for line in lines:
        if line.strip() == line_to_check.strip():
            # Fix the indentation
            fixed_line = "        " + line.strip() + "\n"
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation() 