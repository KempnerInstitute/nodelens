#!/bin/bash
#SBATCH --job-name=alexnet-alighnemt       # Job name
#SBATCH --output=ddp_local_%j.out      # Standard output log
#SBATCH --error=ddp_local_%j.err       # Standard error log
#SBATCH --time=24:00:00                # Max wall time
#SBATCH --partition=kempner            # Or whichever GPU partition
#SBATCH --nodes=4                      # We only use 1 node
#SBATCH --ntasks=16                     # 1 task total
#SBATCH --gres=gpu:4                   # 4 GPUs on this node
#SBATCH --cpus-per-task=24             # Number of CPU cores, adjust as needed
#SBATCH --mem=0                      # Memory
#SBATCH --account=kempner_dev          # HPC account if required

module load python cuda cudnn          # Or your HPC's modules
conda activate networkAlignmentAnalysis # Activate your environment

python -u scripts/ddp_alexnet.py