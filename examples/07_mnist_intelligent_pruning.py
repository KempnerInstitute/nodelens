"""
MNIST Intelligent Pruning Example

Demonstrates complete workflow:
1. Train MLP on MNIST
2. Compute composite importance scores
3. Apply pruning with different strategies
4. Compare results
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from alignment.metrics import get_metric

# Alignment framework
from alignment.models import BaseModelWrapper
from alignment.services import (
    ActivationCaptureService,
    MaskOperations,
    NodeScoringService,
)


class SimpleMLP(nn.Module):
    """MLP: 784 -> 128 -> 64 -> 10"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, epochs=5, device="cpu"):
    """Train model."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}")

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")


def evaluate(model, test_loader, device="cpu"):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def prune_model(model, wrapper, pruning_method, pruning_amount, val_loader, device="cpu"):
    """Prune model using specified method."""
    print(f"\nPruning with {pruning_method} (amount={pruning_amount:.1%})...")

    # Get a batch for metric computation
    inputs_batch, targets_batch = next(iter(val_loader))
    inputs_batch = inputs_batch.to(device)
    targets_batch = targets_batch.to(device)

    # Capture activations
    capture_service = ActivationCaptureService(wrapper)
    data = capture_service.capture(inputs_batch, layers=wrapper.tracked_layers, include_weights=True)

    # Compute scores based on method
    masks = {}

    for layer_name in wrapper.tracked_layers:
        if layer_name not in data.inputs or layer_name not in data.weights:
            continue

        inputs = data.inputs[layer_name]
        weights = data.weights[layer_name]

        if pruning_method == "random":
            # Random scores
            scores = torch.rand(weights.shape[0])

        elif pruning_method == "magnitude":
            # L2 norm of weights
            scores = torch.norm(weights, p=2, dim=1)

        elif pruning_method == "rq":
            # RQ only
            rq_metric = get_metric("rayleigh_quotient")
            scores = rq_metric.compute(inputs, weights)

        elif pruning_method == "composite":
            # Redundancy-aware composite
            scoring_service = NodeScoringService(
                metrics={
                    "rq": get_metric("rayleigh_quotient"),
                    "redundancy": get_metric("pairwise_redundancy_gaussian", num_pairs=8),
                    "synergy": get_metric("synergy_gaussian_mmi", num_pairs=8),
                },
                alpha_mi=0.0,
                beta_synergy=0.3,
                gamma_redundancy=0.4,
                delta_rq=0.3,
            )

            layer_scores = scoring_service.compute_composite_scores(inputs, weights, targets_batch)
            scores = layer_scores.composite

        else:
            raise ValueError(f"Unknown method: {pruning_method}")

        # Create mask
        mask = MaskOperations.create_structured_mask(scores, amount=pruning_amount, mode="low")
        masks[layer_name] = mask

        stats = MaskOperations.get_mask_statistics(mask)
        print(f"  {layer_name}: {stats['kept_elements']}/{stats['total_elements']} kept")

    # Apply masks
    apply_masks_to_model(model, masks, wrapper.tracked_layers)

    return masks


def apply_masks_to_model(model, masks, layer_names):
    """Apply pruning masks to model weights."""
    for name, module in model.named_modules():
        if name in masks and hasattr(module, "weight"):
            mask = masks[name]

            # Expand mask to weight dimensions
            if isinstance(module, nn.Linear):
                # Zero out entire rows (neurons)
                weight_mask = mask.unsqueeze(1).expand_as(module.weight)
                module.weight.data *= weight_mask.float()

                if module.bias is not None:
                    module.bias.data *= mask.float()


def main():
    """Main experiment."""
    print("=" * 80)
    print("MNIST Intelligent Pruning - Complete Workflow")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load MNIST
    print("\n1. Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Validation loader (smaller for metric computation)
    val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Train baseline model
    print("\n2. Training baseline model...")
    model = SimpleMLP().to(device)
    train_model(model, train_loader, epochs=3, device=device)

    baseline_acc = evaluate(model, test_loader, device)
    print(f"\n✓ Baseline accuracy: {baseline_acc:.2f}%")

    # Wrap model
    BaseModelWrapper(model, tracked_layers=["fc1", "fc2"], track_inputs=True, track_outputs=True)  # Track hidden layers

    # Pruning experiments
    print("\n3. Pruning experiments...")
    print("-" * 80)

    pruning_amount = 0.5  # Prune 50%
    methods = ["random", "magnitude", "rq", "composite"]
    results = {}

    for method in methods:
        # Create fresh copy of model
        model_copy = SimpleMLP().to(device)
        model_copy.load_state_dict(model.state_dict())

        # Wrap
        wrapper_copy = BaseModelWrapper(model_copy, tracked_layers=["fc1", "fc2"], track_inputs=True, track_outputs=True)

        # Prune
        prune_model(model_copy, wrapper_copy, method, pruning_amount, val_loader, device)

        # Evaluate immediately after pruning (no fine-tuning)
        acc_pruned = evaluate(model_copy, test_loader, device)

        # Fine-tune
        print(f"\n  Fine-tuning {method}...")
        train_model(model_copy, train_loader, epochs=2, device=device)

        # Final evaluation
        acc_final = evaluate(model_copy, test_loader, device)

        results[method] = {"acc_before": baseline_acc, "acc_pruned": acc_pruned, "acc_final": acc_final, "drop": baseline_acc - acc_final}

        print(f"  {method}: {baseline_acc:.2f}% → {acc_pruned:.2f}% → {acc_final:.2f}% (drop: {baseline_acc - acc_final:.2f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nPruning amount: {pruning_amount:.0%} of neurons")
    print(f"Baseline accuracy: {baseline_acc:.2f}%\n")

    print("Method          | After Pruning | After Fine-tune | Accuracy Drop")
    print("-" * 70)
    for method, res in results.items():
        print(f"{method:15s} | {res['acc_pruned']:13.2f}% | {res['acc_final']:15.2f}% | {res['drop']:13.2f}%")

    # Find best method
    best_method = min(results.keys(), key=lambda m: results[m]["drop"])
    print(f"\n✓ Best method: {best_method} (smallest accuracy drop: {results[best_method]['drop']:.2f}%)")

    print("\nComposite pruning considers redundancy and synergy in addition to alignment.")

    # Save results
    output_dir = Path("results/mnist_intelligent_pruning")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(results, output_dir / "pruning_results.pt")
    print(f"\n✓ Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    results = main()
