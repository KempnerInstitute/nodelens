#!/bin/bash
#SBATCH --job-name=isoH_rng_advance
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/logs/iso_rng_advance_%j.out
#SBATCH --error=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/logs/iso_rng_advance_%j.err

set -euo pipefail
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

source ~/.bashrc
mamba activate alignment2

echo "============================================================================"
echo "Testing: Advance RNG by 50 epochs of shuffling before metrics"
echo "============================================================================"

# This Python script advances the RNG state to simulate post-training, then runs metrics
python - << 'PYTHON'
import torch
import numpy as np
import sys
import os

# Add the project to path
sys.path.insert(0, '/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment')

# Set seeds like Jan-20
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Advance RNG by 50 epochs of DataLoader shuffling (CIFAR-10 has 50000 samples)
n_samples = 50000
for epoch in range(50):
    _ = torch.randperm(n_samples)
    
print(f"Advanced RNG by 50 epochs of shuffling")
print(f"First 10 indices of next shuffle: {torch.randperm(n_samples)[:10].tolist()}")

# Now the RNG should be in the same state as Jan-20 after training
# However, we need to integrate this into the experiment somehow...
# The issue is the experiment is launched via run_experiment.py which resets seeds

print("\nNote: This approach won't work directly because run_experiment.py resets seeds.")
print("We need a different approach - either:")
print("1. Save and restore exact RNG state from Jan-20 (not available)")
print("2. Accept that calibration samples differ and focus on understanding the variance")
print("3. Use deterministic indices mode going forward for reproducibility")
PYTHON

echo "Done"
