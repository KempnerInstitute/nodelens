#!/usr/bin/env bash

# Quick sanity-check experiment on MNIST with a small MLP.
# Uses configs/examples/mnist_basic.yaml.
#
# Usage:
#   bash scripts/run_mnist_basic.sh
#   bash scripts/run_mnist_basic.sh --device cpu
#
# Any extra arguments are forwarded to scripts/run_experiment.py.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

CONFIG="configs/examples/mnist_basic.yaml"

echo "Running MNIST basic alignment experiment with config: ${CONFIG}"
echo "Working directory: ${ROOT_DIR}"
echo

python scripts/run_experiment.py --config "${CONFIG}" "$@"


