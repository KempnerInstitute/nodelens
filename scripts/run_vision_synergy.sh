#!/usr/bin/env bash

# Run the vision synergy / redundancy experiment on ResNet-18 + CIFAR-10.
# This uses the config at configs/projects/vision_synergy.yaml.
#
# Usage:
#   bash scripts/run_vision_synergy.sh
#   bash scripts/run_vision_synergy.sh --device cuda:1
#
# Any extra arguments are forwarded to scripts/run_experiment.py.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

CONFIG="configs/projects/vision_synergy.yaml"

echo "Running vision synergy experiment with config: ${CONFIG}"
echo "Working directory: ${ROOT_DIR}"
echo

python scripts/run_experiment.py --config "${CONFIG}" "$@"


