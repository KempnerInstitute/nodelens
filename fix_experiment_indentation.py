#!/usr/bin/env python
"""
Fix indentation issue in experiment.py
"""

def fix_indentation():
    filepath = "src/alignment/experiments/experiment.py"
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the indentation issues around the import wandb section
    # The actual fix will depend on the context, but let's ensure proper indentation
    fixed_content = content.replace(
        "try:\nimport wandb", 
        "try:\n    import wandb"
    )
    
    # Save the fixed file
    with open(filepath, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed indentation in {filepath}")

if __name__ == "__main__":
    fix_indentation() 