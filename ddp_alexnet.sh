#!/bin/bash
#SBATCH --job-name=alexnet-alignment       # Job name
#SBATCH --output=ddp_%j.out               # Standard output log
#SBATCH --error=ddp_%j.err                # Standard error log
#SBATCH --time=00:30:00                   # Max wall time
#SBATCH --partition=kempner_h100          # GPU partition or queue
#SBATCH --nodes=1                         # 4 nodes total
#SBATCH --ntasks-per-node=4              # 4 tasks per node => 16 tasks total
#SBATCH --gres=gpu:4                     # 4 GPUs on each node
#SBATCH --cpus-per-task=24               # Adjust as needed
#SBATCH --mem=0                          # Memory (0 = all available on node)
#SBATCH --account=kempner_dev            # HPC account if required

module load python cuda cudnn            # or whatever modules your HPC needs
conda activate networkAlignmentAnalysis   # your conda environment

# This sets total WORLD_SIZE = (# of nodes) * (# of GPUs per node)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500  # or any free port number

echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

# srun will launch 4 tasks/node => total 16 tasks across 4 nodes
# Each task picks up its rank automatically (SLURM_PROCID).
srun \
    --ntasks=$WORLD_SIZE \
    python experiment.py alignment_info_stats \
        --save-networks \
        --network MLP \
        --dataset MNIST \
        --use_wandb \
        --dropout_by_layer \
        --epochs 100 \
        --ddp \
        --num-drops 14 \
        --batch-size 1000