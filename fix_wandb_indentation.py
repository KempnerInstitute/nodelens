#!/usr/bin/env python

"""
Quick script to fix wandb import indentation in experiment.py
"""

import os

def fix_indentation():
    # Path to the file with indentation issue
    file_path = "src/alignment/experiments/experiment.py"
    
    # Read the file content
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Find and fix the indentation issue
    for i, line in enumerate(lines):
        if "try:" in line:
            try_index = i
            # Check if the next line is properly indented
            if i+1 < len(lines) and not lines[i+1].startswith(" "):
                # Fix indentation for import wandb line
                lines[i+1] = "    " + lines[i+1]
                print(f"Fixed indentation for line {i+1}")
    
    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.writelines(lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation() 