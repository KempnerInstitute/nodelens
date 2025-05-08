#!/usr/bin/env python
"""
Fix indentation issue in alignment_experiments.py
"""

def fix_indentation():
    filepath = "src/alignment/experiments/alignment_experiments.py"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Fix the indentation at line 75 (0-indexed)
    lines[74] = "        logger.info(f\"Using dataset: {dataset_config.get('name', 'unknown')}\")\n"
    
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed indentation in {filepath}")

if __name__ == "__main__":
    fix_indentation() 