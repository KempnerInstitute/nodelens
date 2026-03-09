#!/bin/bash
#SBATCH --job-name=dataset_overlap_run
#SBATCH --output=logs/dataset_overlap_run_%j.out
#SBATCH --error=logs/dataset_overlap_run_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=kempner
#SBATCH --account=kempner_undergrads


echo "============================================================================"
echo "RUN EXPERIMENT"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate alignenv2

cd /n/holylfs06/LABS/kempner_undergrads/Lab/acherilyn/alignment

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/n/home13/hsafaai/.cache/huggingface
# export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)


python scripts/compare_supernode_dataset_overlap.py \
    --config configs/examples/llama3_dataset_supernode_overlap.yaml

echo ""
echo "============================================================================"
echo "Completed at $(date)"
echo "============================================================================"
