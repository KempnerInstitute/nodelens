"""
Example: Redundancy-Aware Pruning with Information-Theoretic Metrics

This example demonstrates the complete workflow for intelligent pruning using:
- RayleighQuotient (alignment)
- PairwiseRedundancyGaussian (redundancy)
- SynergyGaussianMMI (synergy)
- Class-conditioned RQ (task-relevance)
- NodeScoringService (composite scoring)

The workflow:
1. Load a pretrained model
2. Capture activations
3. Compute multiple metrics
4. Create composite importance scores
5. Prune with redundancy awareness
6. Compare against baseline methods
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from alignment.metrics import get_metric

# Alignment framework imports
from alignment.models import BaseModelWrapper
from alignment.services import (
    ActivationCaptureService,
    MaskOperations,
    NodeScoringService,
)


def create_simple_cnn():
    """Create a simple CNN for demonstration."""
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return model


def evaluate_model(model, dataloader, device="cpu"):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return 100 * correct / total


def main():
    """Main demonstration."""
    print("=" * 80)
    print("Redundancy-Aware Pruning Demonstration")
    print("=" * 80)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load MNIST
    print("\n1. Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Create and wrap model
    print("\n2. Creating model...")
    model = create_simple_cnn().to(device)

    # Simple training (or load pretrained)
    # For demo, we'll use random weights
    print("   (Using random initialization for demo)")

    wrapper = BaseModelWrapper(model, tracked_layers=["3", "6"], track_inputs=True, track_outputs=True)  # Conv2 and Linear1

    print(f"   Tracking layers: {wrapper.tracked_layers}")

    # Initialize services
    print("\n3. Initializing services...")
    capture_service = ActivationCaptureService(wrapper)

    # Initialize metrics
    print("\n4. Initializing metrics...")
    rq_metric = get_metric("rayleigh_quotient", relative=True, regularization=1e-6)
    redundancy_metric = get_metric("pairwise_redundancy_gaussian", num_pairs=10)
    synergy_metric = get_metric("synergy_gaussian_mmi", num_pairs=10)

    scoring_service = NodeScoringService(
        metrics={"rq": rq_metric, "redundancy": redundancy_metric, "synergy": synergy_metric},
        alpha_mi=0.0,  # No MI (would need proper MI metric)
        beta_synergy=0.3,  # Synergy weight
        gamma_redundancy=0.4,  # Redundancy weight (negative)
        delta_rq=0.3,  # RQ weight
    )

    # Capture activations on a subset
    print("\n5. Capturing activations...")
    batch_inputs, batch_targets = next(iter(test_loader))
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)

    activation_data = capture_service.capture(batch_inputs, layers=wrapper.tracked_layers, include_weights=True)

    print(f"   Captured data from {len(activation_data.layer_names)} layers")

    # Compute composite scores
    print("\n6. Computing composite scores...")
    layerwise_scores = scoring_service.compute_layerwise_scores(activation_data, targets=batch_targets, include_redundancy=True, include_synergy=True)

    # Display results for first layer
    layer_name = wrapper.tracked_layers[0]
    if layer_name in layerwise_scores:
        scores = layerwise_scores[layer_name]
        print(f"\n   Results for layer '{layer_name}':")

        if scores.rq is not None:
            print(f"   - RQ:         mean={scores.rq.mean():.4f}, std={scores.rq.std():.4f}")
        if scores.redundancy is not None:
            print(f"   - Redundancy: mean={scores.redundancy.mean():.4f}, std={scores.redundancy.std():.4f}")
        if scores.synergy is not None:
            print(f"   - Synergy:    mean={scores.synergy.mean():.4f}, std={scores.synergy.std():.4f}")
        if scores.composite is not None:
            print(f"   - Composite:  mean={scores.composite.mean():.4f}, std={scores.composite.std():.4f}")

    # Demonstrate class-conditioned RQ
    print("\n7. Computing class-conditioned RQ (ΔRQ)...")
    layer_name = wrapper.tracked_layers[-1]  # Linear layer
    if layer_name in activation_data.inputs:
        delta_rq_results = rq_metric.compute_class_conditioned(
            inputs=activation_data.inputs[layer_name], weights=activation_data.weights[layer_name], targets=batch_targets, return_delta_rq=True
        )

        print(f"   Layer '{layer_name}':")
        print(f"   - RQ (unconditional): {delta_rq_results['rq_uncond'].mean():.4f}")
        print(f"   - RQ (class-cond):    {delta_rq_results['rq_cond'].mean():.4f}")
        print(f"   - ΔRQ:                {delta_rq_results['delta_rq'].mean():.4f}")
        print("   ΔRQ measures task-relevant alignment (larger = more discriminative)")

    # Create pruning masks
    print("\n8. Creating pruning masks...")
    pruning_amount = 0.3  # Prune 30%

    for layer_name, scores in layerwise_scores.items():
        if scores.composite is None:
            continue

        # Create mask using composite scores
        mask = MaskOperations.create_structured_mask(scores.composite, amount=pruning_amount, mode="low")  # Prune low-importance neurons

        stats = MaskOperations.get_mask_statistics(mask)
        print(f"   Layer '{layer_name}': {stats['kept_elements']}/{stats['total_elements']} neurons kept")

    # Compare methods
    print("\n9. Comparing pruning methods...")
    print("   Method comparison (on same layer):")
    layer_name = wrapper.tracked_layers[-1]
    if layer_name in layerwise_scores:
        scores = layerwise_scores[layer_name]

        # Method 1: Random
        mask_random = MaskOperations.create_structured_mask(torch.rand_like(scores.composite), amount=pruning_amount, mode="random")

        # Method 2: RQ only
        mask_rq = MaskOperations.create_structured_mask(scores.rq, amount=pruning_amount, mode="low")

        # Method 3: Composite (redundancy-aware)
        mask_composite = MaskOperations.create_structured_mask(scores.composite, amount=pruning_amount, mode="low")

        # Check overlap
        overlap_rq_composite = (mask_rq & mask_composite).sum().item() / mask_rq.sum().item()
        overlap_random_composite = (mask_random & mask_composite).sum().item() / mask_random.sum().item()

        print(f"   - Overlap (RQ vs Composite):     {overlap_rq_composite:.2%}")
        print(f"   - Overlap (Random vs Composite): {overlap_random_composite:.2%}")
        print(f"   → Redundancy-awareness changes {(1-overlap_rq_composite)*100:.1f}% of pruning decisions")

    # Evaluate baseline
    print("\n10. Evaluating model...")
    baseline_acc = evaluate_model(model, test_loader, device)
    print(f"    Baseline accuracy: {baseline_acc:.2f}%")
    print("    (Would apply masks and fine-tune for full experiment)")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        """
Key Features Demonstrated:
✓ ActivationCaptureService - Clean API for activation capture
✓ PairwiseRedundancyGaussian - Identifies redundant neurons
✓ SynergyGaussianMMI - Identifies complementary neurons
✓ Class-conditioned RQ - Measures task-relevant alignment (ΔRQ)
✓ NodeScoringService - Composite scoring with configurable weights
✓ MaskOperations - Flexible mask creation and analysis

Benefits of Redundancy-Aware Pruning:
• Preserves complementary (synergistic) neurons
• Removes redundant (overlapping) neurons
• Uses task-relevant alignment (ΔRQ) when targets available
• Expected: +3-5% accuracy retention vs magnitude at same sparsity

Next Steps:
1. Train model to convergence
2. Apply masks and fine-tune
3. Compare accuracy vs magnitude/random baselines
4. Repeat across multiple sparsity levels
    """
    )


if __name__ == "__main__":
    main()
