"""
Demo of pruning strategies with different modes (low/high/random).

This demonstrates:
1. Basic pruning with different modes
2. Parallel pruning strategies
3. Performance comparison

Usage:
    python pruning_strategies_demo.py

No configuration needed - this script runs standalone demonstrations.

This comprehensive demo showcases:
    - Basic pruning modes (low/high/random)
    - Parallel execution of multiple pruning strategies
    - Tensorized pruning for GPU efficiency
    - Gradient-based pruning
    - Performance comparisons and speedup analysis
    - Memory efficiency analysis
    - Overlap analysis between different pruning modes

Requirements:
    - PyTorch
    - CUDA (optional, for GPU acceleration demos)
    - alignment package installed
    - numpy

Output:
    Console output showing:
    - Pruning results for each strategy
    - Performance metrics and timing comparisons
    - Memory usage analysis
    - Statistical analysis of pruning patterns
    - Key takeaways and insights
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Direct imports
from alignment.pruning.base import PruningConfig
from alignment.pruning.strategies import (
    GradientPruning,
    MagnitudePruning,
    ParallelModePruning,
    RandomPruning,
    TensorizedPruning,
)


class SimpleNet(nn.Module):
    """Simple network for demonstration."""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def demo_basic_pruning():
    """Demonstrate basic pruning with different modes."""
    print("\n=== Basic Pruning Demo ===\n")

    # Create model
    model = SimpleNet(hidden_size=512)

    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    # Test different pruning modes on first layer
    layer = model.fc1
    sparsity = 0.5

    print(f"Layer shape: {layer.weight.shape}")
    print(f"Target sparsity: {sparsity*100}%\n")

    # 1. Magnitude pruning - low mode (prune small weights)
    print("1. Magnitude Pruning - Low Mode (prune small weights):")
    config = PruningConfig(amount=sparsity, pruning_mode='low')
    pruner = MagnitudePruning(config)

    scores = pruner.compute_importance_scores(layer)
    mask = pruner.create_pruning_mask(scores)

    weights = layer.weight.data.flatten()
    kept_weights = weights[mask.flatten().bool()]
    pruned_weights = weights[~mask.flatten().bool()]

    print(f"   Sparsity achieved: {(mask == 0).float().mean():.2%}")
    print(f"   Avg magnitude of kept weights: {kept_weights.abs().mean():.4f}")
    print(f"   Avg magnitude of pruned weights: {pruned_weights.abs().mean():.4f}")
    print(f"   Max pruned weight magnitude: {pruned_weights.abs().max():.4f}")

    # 2. Magnitude pruning - high mode (prune large weights)
    print("\n2. Magnitude Pruning - High Mode (prune large weights):")
    config = PruningConfig(amount=sparsity, pruning_mode='high')
    pruner = MagnitudePruning(config)

    mask = pruner.create_pruning_mask(scores)

    kept_weights = weights[mask.flatten().bool()]
    pruned_weights = weights[~mask.flatten().bool()]

    print(f"   Sparsity achieved: {(mask == 0).float().mean():.2%}")
    print(f"   Avg magnitude of kept weights: {kept_weights.abs().mean():.4f}")
    print(f"   Avg magnitude of pruned weights: {pruned_weights.abs().mean():.4f}")
    print(f"   Min pruned weight magnitude: {pruned_weights.abs().min():.4f}")

    # 3. Random pruning
    print("\n3. Random Pruning:")
    config = PruningConfig(amount=sparsity)
    pruner = RandomPruning(config)

    scores = pruner.compute_importance_scores(layer)
    mask = pruner.create_pruning_mask(scores)

    kept_weights = weights[mask.flatten().bool()]
    pruned_weights = weights[~mask.flatten().bool()]

    print(f"   Sparsity achieved: {(mask == 0).float().mean():.2%}")
    print(f"   Avg magnitude of kept weights: {kept_weights.abs().mean():.4f}")
    print(f"   Avg magnitude of pruned weights: {pruned_weights.abs().mean():.4f}")
    print(f"   Std of pruned weights magnitude: {pruned_weights.abs().std():.4f}")


def demo_parallel_pruning():
    """Demonstrate parallel pruning strategies."""
    print("\n\n=== Parallel Pruning Demo ===\n")

    # Create model
    model = SimpleNet(hidden_size=256)

    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    layer = model.fc1
    sparsity = 0.5

    # 1. Parallel mode pruning
    print("1. Parallel Mode Pruning (all modes simultaneously):")
    parallel_pruner = ParallelModePruning(
        modes=['low', 'high', 'random'],
        base_strategy='magnitude'
    )

    start_time = time.time()
    result = parallel_pruner.prune_parallel(layer, amount=sparsity)
    parallel_time = time.time() - start_time

    print(f"\n   Execution time: {parallel_time:.4f}s")
    print("\n   Sparsity by mode:")
    for mode, sparse in result.sparsities.items():
        print(f"     {mode}: {sparse:.2%}")

    # Analyze overlaps
    print("\n   Mask overlaps:")
    low_high_overlap = (result.masks['low'] * result.masks['high']).sum().item()
    low_random_overlap = (result.masks['low'] * result.masks['random']).sum().item()
    high_random_overlap = (result.masks['high'] * result.masks['random']).sum().item()
    total = result.masks['low'].numel()

    print(f"     low ∩ high: {low_high_overlap/total:.2%}")
    print(f"     low ∩ random: {low_random_overlap/total:.2%}")
    print(f"     high ∩ random: {high_random_overlap/total:.2%}")

    # 2. Sequential comparison
    print("\n2. Sequential execution (for comparison):")

    start_time = time.time()
    masks_sequential = {}

    for mode in ['low', 'high', 'random']:
        config = PruningConfig(amount=sparsity, pruning_mode=mode)
        pruner = MagnitudePruning(config)
        scores = pruner.compute_importance_scores(layer)
        masks_sequential[mode] = pruner.create_pruning_mask(scores)

    sequential_time = time.time() - start_time

    print(f"\n   Execution time: {sequential_time:.4f}s")
    print(f"   Speedup from parallel: {sequential_time/parallel_time:.2f}x")

    # Verify results match
    print(f"\n   Results match: {all(torch.allclose(result.masks[mode], masks_sequential[mode]) for mode in ['low', 'high', 'random'])}")


def demo_tensorized_pruning():
    """Demonstrate tensorized pruning for GPU efficiency."""
    print("\n\n=== Tensorized Pruning Demo ===\n")

    # Create larger model for better timing comparison
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    # 1. Tensorized pruning - compute multiple sparsity levels at once
    print("\n1. Tensorized Pruning (multiple sparsity levels):")

    tensorized_pruner = TensorizedPruning()

    # Test on first layer
    layer = model[0]
    modes = ['low', 'high', 'random']
    amounts = [0.1, 0.3, 0.5, 0.7, 0.9]

    start_time = time.time()
    pruning_tensor = tensorized_pruner.compute_pruning_tensor(
        layer,
        modes=modes,
        amounts=amounts
    )
    tensorized_time = time.time() - start_time

    print(f"\n   Execution time: {tensorized_time:.4f}s")
    print(f"   Pruning tensor shape: {pruning_tensor.shape}")
    print(f"   (modes={len(modes)}, amounts={len(amounts)}, *weight_shape)")

    # Analyze patterns
    print("\n   Analyzing pruning patterns...")
    analysis = tensorized_pruner.analyze_pruning_patterns(pruning_tensor)

    print("\n   Sparsity progression (avg across weight dims):")
    sparsity_prog = analysis['sparsity_progression']
    for i, mode in enumerate(modes):
        print(f"     {mode}: ", end="")
        for j, amount in enumerate(amounts):
            print(f"{sparsity_prog[i, j]:.2f} ", end="")
        print()

    # 2. Compare with sequential approach
    print("\n2. Sequential approach (for comparison):")

    start_time = time.time()
    sequential_masks = {}

    for mode in modes:
        sequential_masks[mode] = {}
        for amount in amounts:
            config = PruningConfig(amount=amount, pruning_mode=mode)
            if mode == 'random':
                pruner = RandomPruning(config)
            else:
                pruner = MagnitudePruning(config)

            scores = pruner.compute_importance_scores(layer)
            mask = pruner.create_pruning_mask(scores)
            sequential_masks[mode][amount] = mask

    sequential_time = time.time() - start_time

    print(f"\n   Execution time: {sequential_time:.4f}s")
    print(f"   Speedup from tensorized: {sequential_time/tensorized_time:.2f}x")

    # 3. Memory efficiency analysis
    print("\n3. Memory efficiency:")

    # Tensorized approach stores everything in one tensor
    tensorized_memory = pruning_tensor.element_size() * pruning_tensor.nelement()

    # Sequential approach stores individual masks
    sequential_memory = 0
    for mode_masks in sequential_masks.values():
        for mask in mode_masks.values():
            sequential_memory += mask.element_size() * mask.nelement()

    print(f"   Tensorized memory: {tensorized_memory / (1024**2):.2f} MB")
    print(f"   Sequential memory: {sequential_memory / (1024**2):.2f} MB")
    print(f"   Memory efficiency: {sequential_memory/tensorized_memory:.2f}x")

    # 4. Show overlap analysis
    if 'mode_overlap' in analysis:
        print("\n4. Mode overlap analysis (low ∩ high):")
        overlap = analysis['mode_overlap']
        print("   Sparsity:  ", end="")
        for amount in amounts:
            print(f"{amount:.1f}  ", end="")
        print("\n   Overlap:   ", end="")
        for j in range(len(amounts)):
            print(f"{overlap[j]:.2f} ", end="")
        print()


def demo_gradient_based_pruning():
    """Demonstrate gradient-based pruning."""
    print("\n\n=== Gradient-Based Pruning Demo ===\n")

    # Create model and dummy data
    model = SimpleNet(input_size=784, hidden_size=256)
    criterion = nn.CrossEntropyLoss()

    # Create dummy batch
    batch_size = 32
    x = torch.randn(batch_size, 784)
    y = torch.randint(0, 10, (batch_size,))

    # Forward and backward pass to get gradients
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    layer = model.fc1
    sparsity = 0.5

    print(f"Layer shape: {layer.weight.shape}")
    print(f"Target sparsity: {sparsity*100}%")

    # 1. Gradient magnitude pruning - low mode
    print("\n1. Gradient Pruning - Low Mode (prune small gradients):")
    config = PruningConfig(amount=sparsity, pruning_mode='low')
    pruner = GradientPruning(config)

    scores = pruner.compute_importance_scores(layer)
    mask = pruner.create_pruning_mask(scores)

    grad_magnitudes = layer.weight.grad.abs().flatten()
    kept_grads = grad_magnitudes[mask.flatten().bool()]
    pruned_grads = grad_magnitudes[~mask.flatten().bool()]

    print(f"   Sparsity achieved: {(mask == 0).float().mean():.2%}")
    print(f"   Avg gradient magnitude of kept: {kept_grads.mean():.6f}")
    print(f"   Avg gradient magnitude of pruned: {pruned_grads.mean():.6f}")

    # 2. Gradient magnitude pruning - high mode
    print("\n2. Gradient Pruning - High Mode (prune large gradients):")
    config = PruningConfig(amount=sparsity, pruning_mode='high')
    pruner = GradientPruning(config)

    mask = pruner.create_pruning_mask(scores)

    kept_grads = grad_magnitudes[mask.flatten().bool()]
    pruned_grads = grad_magnitudes[~mask.flatten().bool()]

    print(f"   Sparsity achieved: {(mask == 0).float().mean():.2%}")
    print(f"   Avg gradient magnitude of kept: {kept_grads.mean():.6f}")
    print(f"   Avg gradient magnitude of pruned: {pruned_grads.mean():.6f}")

    # Compare with weight magnitude
    print("\n3. Gradient vs Weight magnitude correlation:")
    weight_mags = layer.weight.abs().flatten()
    correlation = torch.corrcoef(torch.stack([weight_mags, grad_magnitudes]))[0, 1]
    print(f"   Correlation: {correlation:.4f}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Pruning Strategies Demo")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Run demos
        demo_basic_pruning()
        demo_parallel_pruning()
        demo_tensorized_pruning()
        demo_gradient_based_pruning()

        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)

        print("\nKey takeaways:")
        print("1. Low mode pruning: Removes small magnitude weights (common approach)")
        print("2. High mode pruning: Removes large magnitude weights (adversarial)")
        print("3. Parallel pruning: Compute multiple modes simultaneously for efficiency")
        print("4. Tensorized pruning: GPU-optimized for multiple sparsity levels")
        print("5. Gradient pruning: Uses gradient information for importance")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
