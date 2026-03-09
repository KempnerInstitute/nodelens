#!/bin/bash
#SBATCH --job-name=overlap_run_exp
#SBATCH --output=logs/overlap_run_exp_%j.out
#SBATCH --error=logs/overlap_run_exp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=kempner_h100
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


python scripts/run_experiment.py \
    --config configs/examples/llama3_activation_vs_scar_overlap.yaml \
    --device cuda \
    --allow-dirty

echo ""
echo "============================================================================"
echo "Completed at $(date)"
echo "============================================================================"
