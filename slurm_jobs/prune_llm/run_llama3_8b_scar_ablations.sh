#!/bin/bash
#SBATCH --job-name=paper_llama3_scar_ablations
#SBATCH --output=logs/paper_llama3_scar_ablations_%j.out
#SBATCH --error=logs/paper_llama3_scar_ablations_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=kempner_h100_priority3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320GB
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_dev

# SCAR Ablations: Random Supernode + SCAR-Optimal
# Tests:
# 1. Random supernode control (do LP-identified supernodes matter?)
# 2. SCAR-optimal (learned combination of LP, Activation, Taylor, Curvature)

set -e

# Setup environment
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
source ~/.bashrc
conda activate alignment 2>/dev/null || source activate alignment 2>/dev/null || true

# HuggingFace cache
export HF_HOME="/n/netscratch/kempner_dev/Everyone/hf_cache"
mkdir -p "$HF_HOME"

# Output directory
timestamp=$(date +%Y%m%d_%H%M%S)
job_id=${SLURM_JOB_ID:-local}
output_dir="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER/llama3_8b_paper_results_scar_ablations_${timestamp}_${job_id}"
mkdir -p "$output_dir"

echo "=========================================="
echo "SCAR Ablation Experiments"
echo "=========================================="
echo "Output directory: $output_dir"
echo "Job ID: $job_id"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Run the experiment with ablation flags
python -m alignment.experiments.llm_experiments \
    --model_name "meta-llama/Llama-3.1-8B" \
    --output_dir "$output_dir" \
    --experiment_type "paper_sweep" \
    --device "cuda" \
    --calibration_dataset "wikitext" \
    --calibration_num_samples 64 \
    --evaluation_num_samples 128 \
    --do_scar_analysis true \
    --do_supernode_analysis true \
    --do_supernode_connectivity true \
    --do_random_supernode_ablation true \
    --do_scar_optimal true \
    --scar_optimal_granularity 5 \
    --supernode_rho 0.01 \
    --supernode_eta 0.10 \
    --pruning_strategies "['scar_loss_proxy', 'supernode_protection_score', 'random_supernode']" \
    --pruning_sparsities "[0.3, 0.5]" \
    --generate_plots true \
    --save_results true \
    2>&1 | tee "$output_dir/experiment.log"

echo ""
echo "=========================================="
echo "Experiment Complete"
echo "=========================================="
echo "Results saved to: $output_dir"
