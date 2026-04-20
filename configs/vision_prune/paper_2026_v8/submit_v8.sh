#!/bin/bash
#SBATCH --job-name=v8_hybrid
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-15%16
#SBATCH --output=/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER/importance_clustering_v8/slurm_%A_%a.out
#SBATCH --error=/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER/importance_clustering_v8/slurm_%A_%a.err

module load cuda/12.2.0-fasrc01
module load gcc/12.2.0-fasrc01

# Activate conda
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Get config for this array task
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/vision_prune/paper_2026_v8/v8_config_list.txt)

echo "========================================="
echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================="

python scripts/run_experiment.py --config "$CONFIG" --allow-dirty
