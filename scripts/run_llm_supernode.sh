#!/usr/bin/env bash

# Run the LLaMA-3 supernode / SCAR-style pruning experiment.
# This uses the project config at configs/projects/llm_supernode.yaml.
#
# NOTE:
#   - This experiment expects access to a GPU and sufficient memory to host
#     the specified HF model (e.g. meta-llama/Llama-3.1-8B).
#   - Make sure you have the correct environment (transformers, datasets, peft)
#     and any necessary authentication for Hugging Face models.
#
# Usage:
#   bash scripts/run_llm_supernode.sh
#   bash scripts/run_llm_supernode.sh --device cuda:0
#
# Any extra arguments are forwarded to scripts/run_experiment.py.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

CONFIG="configs/projects/llm_supernode.yaml"

echo "Running LLM supernode (SCAR-style) experiment with config: ${CONFIG}"
echo "Working directory: ${ROOT_DIR}"
echo

python scripts/run_experiment.py --config "${CONFIG}" "$@"


