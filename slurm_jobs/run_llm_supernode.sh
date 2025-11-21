#!/bin/bash
#SBATCH --job-name=llm_supernode_llama3
#SBATCH --output=logs/llm_supernode_%j.out
#SBATCH --error=logs/llm_supernode_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

echo "Starting LLM supernode / SCAR-style pruning experiment at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"

# Environment setup (conda env: networkAlignmentAnalysis)
module purge
module load cuda/12.2.0-fasrc01
source activate networkAlignmentAnalysis || conda activate networkAlignmentAnalysis || true

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/run_experiment.py \
    --config configs/projects/llm_supernode.yaml \
    --device cuda

echo "LLM supernode / SCAR-style pruning experiment completed at $(date)"


