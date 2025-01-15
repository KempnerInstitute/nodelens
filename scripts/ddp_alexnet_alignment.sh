#!/bin/bash
#SBATCH --job-name=ddp-alexnet-align
#SBATCH --output=ddp_output_%j.out
#SBATCH --error=ddp_output_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=kempner           # or whichever GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # total tasks = 4 (one per GPU)
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --account=kempner_dev        # or your HPC account

module load python cuda cudnn

# Activate your conda environment that has PyTorch + alignment_v2 installed
conda activate networkAlignmentAnalysis

# This block sets environment variables for DDP. Slurm usually sets these automatically:
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500         # or choose any free port
export WORLD_SIZE=$SLURM_NTASKS
export NODE_RANK=$SLURM_NODEID

echo "Running on node $HOSTNAME. MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

# Each rank uses local rank [0..3] as the GPU device
# srun will set variables: SLURM_PROCID, SLURM_LOCALID, etc.

# Finally launch the Python script
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     python -u ddp_alexnet_alignment.py