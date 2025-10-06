"""
Quick demo of the alignment module functionality.

This script provides a minimal example showing the basic workflow of the alignment framework.
It demonstrates model wrapping, metric computation, and basic pruning on a simple MLP.

Usage:
    python quick_demo.py

No configuration needed - this script runs with default settings and creates its own model.

Requirements:
    - PyTorch
    - alignment package installed

Output:
    - Console output showing:
        * Model structure
        * Tracked layers
        * Activation shapes
        * Alignment metrics (Rayleigh Quotient, Weight Cosine Similarity)
        * Pruning results
"""

import torch
import torch.nn as nn

from alignment.metrics import get_metric
from alignment.models import ModelWrapper


def main():
    # 1. Create a simple neural network
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    print("Model created with 3 linear layers")

    # 2. Wrap the model to track activations
    wrapped_model = ModelWrapper(model)
    print(f"\nTracked layers: {wrapped_model.tracked_layers}")

    # 3. Generate some dummy data
    batch_size = 32
    inputs = torch.randn(batch_size, 784)

    # 4. Forward pass with activation tracking
    outputs, activations = wrapped_model.forward_with_activations(inputs)
    print(f"\nCollected activations from {len(activations)} points")
    print(f"Activation keys: {list(activations.keys())}")

    # 5. Get layer weights
    weights = wrapped_model.get_layer_weights()
    print(f"Extracted weights from {len(weights)} layers")

    # 6. Compute alignment metrics
    print("\n" + "="*60)
    print("Computing Alignment Metrics")
    print("="*60)

    # Rayleigh Quotient
    RQMetric = get_metric('rayleigh_quotient')
    rq_metric = RQMetric()  # Instantiate the metric

    for layer_name in wrapped_model.tracked_layers:
        # Get inputs and weights for this layer
        layer_inputs = activations[f"{layer_name}_input"]
        layer_weights = weights[layer_name]

        # Compute RQ scores
        scores = rq_metric.compute(inputs=layer_inputs, weights=layer_weights)

        print(f"\nLayer: {layer_name}")
        print(f"  Input shape: {layer_inputs.shape}")
        print(f"  Weight shape: {layer_weights.shape}")
        print(f"  RQ scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
        print(f"  Min neuron score: {scores.min():.4f}")
        print(f"  Max neuron score: {scores.max():.4f}")

    # 7. Try other metrics
    print("\n" + "="*60)
    print("Trying Other Metrics")
    print("="*60)

    # Weight cosine similarity
    WeightSimMetric = get_metric('weight_cosine_similarity')
    weight_sim = WeightSimMetric()  # Instantiate
    for layer_name in wrapped_model.tracked_layers:
        layer_weights = weights[layer_name]
        sim_scores = weight_sim.compute(weights=layer_weights)
        print(f"\nWeight Cosine Similarity ({layer_name}): mean={sim_scores.mean():.4f}")

    # 8. Demonstrate pruning
    print("\n" + "="*60)
    print("Pruning Demo")
    print("="*60)

    from alignment.pruning import PruningConfig, get_pruning_strategy

    # Use magnitude-based pruning
    config = PruningConfig(amount=0.5, pruning_mode='low')
    strategy = get_pruning_strategy('magnitude', config=config)

    # Apply pruning to first layer
    first_layer = model[0]  # First linear layer
    mask = strategy.prune(first_layer)

    print("\nPruning 50% of weights in first layer")
    print(f"Sparsity achieved: {(mask == 0).float().mean():.2%}")

    # Apply mask
    first_layer.weight.data *= mask

    # Test forward pass after pruning
    pruned_outputs = model(inputs)
    print(f"\nOutput shape after pruning: {pruned_outputs.shape}")
    print("Pruning applied successfully!")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
