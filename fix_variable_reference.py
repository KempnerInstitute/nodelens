#!/usr/bin/env python
"""
Fix variable reference error in alignment_experiments.py
"""

def fix_variable_reference():
    filepath = "src/alignment/experiments/alignment_experiments.py"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the problematic section
    init_start = None
    dataset_config_check = None
    dataset_usage = None
    
    for i, line in enumerate(lines):
        if "def __init__" in line:
            init_start = i
        if init_start and "dataset_config = " in line:
            dataset_config_check = i
        if init_start and "logger.info(f\"Using dataset: {dataset_config.get" in line:
            dataset_usage = i
    
    if all([init_start, dataset_config_check, dataset_usage]):
        # Add initialization of dataset_config variable before the if check
        indentation = "        "  # Assuming consistent indentation in the file
        lines.insert(dataset_config_check, f"{indentation}dataset_config = {{}}\n")
        
        # Fix the conditional section to update dataset_config instead of reassigning
        lines[dataset_config_check + 1] = lines[dataset_config_check + 1].replace(
            "dataset_config = self.config.dataset",
            "dataset_config.update(self.config.dataset)"
        )
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
        
        print(f"Fixed variable reference in {filepath}")
    else:
        print("Could not find the target code section. No changes made.")

if __name__ == "__main__":
    fix_variable_reference() 