#!/bin/bash
# ==============================================================================
# Submit AlexNet / CIFAR-10 multi-seed runs to the PAPER folder
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"

echo "Submitting AlexNet / CIFAR-10 multi-seed jobs..."
sbatch run_alexnet_cifar10_seed_array.sh

echo "Done! Use 'squeue -u \$USER' to monitor jobs."
