#!/bin/bash
# Run comprehensive MNIST MLP pruning experiments using unified_experiment.py

echo "🚀 Starting MNIST MLP Pruning Experiments"
echo "Using configs/mnist_mlp_experiments.yaml"
echo ""

echo "=================================================================================="
echo "Experiment 1: Standard Pruning (90% sparsity)"
echo "=================================================================================="
python examples/unified_experiment.py \
    --config configs/mnist_mlp_experiments.yaml \
    --name mnist_mlp_standard_pruning \
    --pruning_experiment standard \
    --pruning_config.amount 0.9

echo ""
echo "=================================================================================="
echo "Experiment 2: Layer-Isolated Pruning (High/Low/Random)"
echo "=================================================================================="
python examples/unified_experiment.py \
    --config configs/mnist_mlp_experiments.yaml \
    --name mnist_mlp_layer_isolated \
    --pruning_experiment layer_isolated \
    --dropout_rates 0.0 0.2 0.4 0.6 0.8 0.9

echo ""
echo "=================================================================================="
echo "Experiment 3: Cascading Layer Pruning"
echo "=================================================================================="
python examples/unified_experiment.py \
    --config configs/mnist_mlp_experiments.yaml \
    --name mnist_mlp_cascading \
    --pruning_experiment cascading_layer \
    --dropout_rates 0.0 0.2 0.4 0.6 0.8 0.9

echo ""
echo "✨ All experiments completed!"
echo "Check the logs/ directory for results and visualizations" 