#!/bin/bash
#SBATCH --job-name=paper_llama3_two_halo
#SBATCH --output=logs/paper_llama3_two_halo_%j.out
#SBATCH --error=logs/paper_llama3_two_halo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment}"
mkdir -p logs

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/huggingface_cache"
if [[ -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(cat "${HF_HOME}/token")"
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
mkdir -p "$HF_HOME"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"

CAL_N=32
CAL_MAXLEN=512

python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_two_halo_ablation" \
  generate_plots=false \
  alignment_data_num_samples="${CAL_N}" \
  scar_num_samples="${CAL_N}" \
  scar_max_length="${CAL_MAXLEN}" \
  "llm.scar_num_samples=${CAL_N}" \
  "llm.scar_max_length=${CAL_MAXLEN}" \
  "llm.evaluation_metrics=['perplexity']" \
  "llm.evaluation_num_samples=64" \
  "llm.perplexity_protocol=legacy" \
  pruning_strategies="['scar_loss_proxy','supernode_protection_score','supernode_connectivity_score','supernode_read_halo_protect_score','supernode_two_halo_score','wanda']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  do_connectivity_pruning=true \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  "supernode.read_halo_pruning.enabled=true" \
  "supernode.read_halo_pruning.read_halo_fraction=0.10" \
  "supernode.read_halo_pruning.rank_power=8.0" \
  "supernode.read_halo_pruning.protection_floor=0.2" \
  "supernode.read_halo_pruning.random_seed=0" \
  supernode.protect_core=true \
  supernode.protect_core_metrics="['scar_loss_proxy','supernode_protection_score','supernode_connectivity_score','supernode_read_halo_protect_score','supernode_two_halo_score']"

