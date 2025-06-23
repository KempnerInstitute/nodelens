#!/usr/bin/env python3
"""
Demonstration of all pruning methods and strategies in the alignment module.

This script shows:
1. Different pruning strategies (magnitude, random, structured, gradient-based)
2. Iterative pruning with fine-tuning
3. Layer-wise vs global pruning
4. Pruning schedules
5. Impact on model accuracy and alignment metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import alignment modules
from alignment.core import ModelWrapper
from alignment.metrics import METRIC_REGISTRY
from alignment.utils.pruning import PruningUtilities, PruningConfig, create_pruning_schedule
from alignment.visualization.alignment_plots import AlignmentVisualizer


class ConvNet(nn.Module):
    """Simple CNN for MNIST demonstration."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_mnist(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, epochs=5, device='cpu'):
    """Train the model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def compute_alignment_metrics(model, data_loader, device='cpu'):
    """Compute alignment metrics for the model."""
    wrapper = ModelWrapper(
        model,
        tracked_layers=['fc1', 'fc2'],
        device=device
    )
    
    # Select key metrics
    metrics = {
        'rayleigh_quotient': METRIC_REGISTRY['rayleigh_quotient'](),
        'mutual_information': METRIC_REGISTRY['mutual_information'](),
        'weight_cosine_similarity': METRIC_REGISTRY['weight_cosine_similarity']()
    }
    
    # Get a batch for analysis
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    
    outputs, activations = wrapper.forward_with_activations(inputs)
    weights = wrapper.get_layer_weights()
    
    results = {}
    for layer_name in wrapper.tracked_layers:
        layer_inputs = activations.get(f"{layer_name}_input")
        layer_weights = weights.get(layer_name)
        layer_outputs = activations.get(f"{layer_name}_output", outputs)
        
        results[layer_name] = {}
        for metric_name, metric in metrics.items():
            try:
                scores = metric.compute(
                    inputs=layer_inputs,
                    weights=layer_weights,
                    outputs=layer_outputs
                )
                results[layer_name][metric_name] = scores.mean().item()
            except Exception as e:
                logger.warning(f"Could not compute {metric_name} for {layer_name}: {e}")
    
    return results


def demonstrate_basic_pruning(model, test_loader, device='cpu'):
    """Demonstrate basic pruning strategies."""
    logger.info("\n" + "="*60)
    logger.info("BASIC PRUNING STRATEGIES")
    logger.info("="*60)
    
    strategies = ['magnitude', 'random']
    pruning_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {strategy: {'sparsity': [], 'accuracy': []} for strategy in strategies}
    
    original_state = model.state_dict()
    
    for strategy in strategies:
        logger.info(f"\n--- {strategy.upper()} PRUNING ---")
        
        for ratio in pruning_ratios:
            # Reset model
            model.load_state_dict(original_state)
            
            # Apply pruning
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if strategy == 'magnitude':
                        mask = PruningUtilities.get_pruning_mask_magnitude(
                            module.weight.data, amount=ratio
                        )
                    else:  # random
                        mask = PruningUtilities.get_pruning_mask_random(
                            module.weight.data, amount=ratio
                        )
                    
                    PruningUtilities.apply_pruning_mask(module, mask)
            
            # Evaluate
            sparsity = PruningUtilities.get_model_sparsity(model)
            accuracy = evaluate_model(model, test_loader, device)
            
            results[strategy]['sparsity'].append(sparsity)
            results[strategy]['accuracy'].append(accuracy)
            
            logger.info(f"Sparsity: {sparsity:.2%}, Accuracy: {accuracy:.2f}%")
    
    return results


def demonstrate_structured_pruning(model, test_loader, device='cpu'):
    """Demonstrate structured pruning."""
    logger.info("\n" + "="*60)
    logger.info("STRUCTURED PRUNING")
    logger.info("="*60)
    
    original_state = model.state_dict()
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {'sparsity': [], 'accuracy': [], 'filters_removed': []}
    
    for ratio in pruning_ratios:
        # Reset model
        model.load_state_dict(original_state)
        filters_removed = 0
        
        # Apply structured pruning to conv layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                original_filters = module.weight.shape[0]
                PruningUtilities.structured_pruning(module, amount=ratio, dim=0)
                
                # Count removed filters
                remaining = (module.weight.sum(dim=(1, 2, 3)) != 0).sum().item()
                filters_removed += original_filters - remaining
        
        # Evaluate
        sparsity = PruningUtilities.get_model_sparsity(model)
        accuracy = evaluate_model(model, test_loader, device)
        
        results['sparsity'].append(sparsity)
        results['accuracy'].append(accuracy)
        results['filters_removed'].append(filters_removed)
        
        logger.info(f"Ratio: {ratio:.1%}, Filters removed: {filters_removed}, "
                   f"Sparsity: {sparsity:.2%}, Accuracy: {accuracy:.2f}%")
    
    return results


def demonstrate_iterative_pruning(model, train_loader, test_loader, device='cpu'):
    """Demonstrate iterative pruning with fine-tuning."""
    logger.info("\n" + "="*60)
    logger.info("ITERATIVE PRUNING WITH FINE-TUNING")
    logger.info("="*60)
    
    original_state = model.state_dict()
    model.load_state_dict(original_state)
    
    # Iterative pruning configuration
    target_sparsity = 0.9
    iterations = 10
    fine_tune_epochs = 1
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # Get subset of training data for faster fine-tuning
    fine_tune_loader = DataLoader(
        train_loader.dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    
    logger.info(f"Target sparsity: {target_sparsity:.0%} over {iterations} iterations")
    
    accuracies = PruningUtilities.iterative_magnitude_pruning(
        model,
        amount=target_sparsity,
        iterations=iterations,
        dataloader=fine_tune_loader,
        fine_tune_epochs=fine_tune_epochs,
        optimizer=optimizer,
        loss_fn=criterion
    )
    
    # Final evaluation
    final_accuracy = evaluate_model(model, test_loader, device)
    final_sparsity = PruningUtilities.get_model_sparsity(model)
    
    logger.info(f"\nFinal results: Sparsity: {final_sparsity:.2%}, Accuracy: {final_accuracy:.2f}%")
    
    return accuracies


def demonstrate_pruning_schedules(epochs=20):
    """Demonstrate different pruning schedules."""
    logger.info("\n" + "="*60)
    logger.info("PRUNING SCHEDULES")
    logger.info("="*60)
    
    schedules = {
        'linear': create_pruning_schedule(0.0, 0.9, 0, epochs, 1, 'linear'),
        'polynomial': create_pruning_schedule(0.0, 0.9, 0, epochs, 1, 'polynomial'),
        'exponential': create_pruning_schedule(0.0, 0.9, 0, epochs, 1, 'exponential')
    }
    
    # Plot schedules
    steps = range(epochs + 1)
    plt.figure(figsize=(10, 6))
    
    for name, schedule in schedules.items():
        sparsities = [schedule(step) for step in steps]
        plt.plot(steps, sparsities, label=name, linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Target Sparsity')
    plt.title('Pruning Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pruning_schedules.png', dpi=150)
    plt.close()
    
    logger.info("Saved pruning schedules plot to pruning_schedules.png")


def demonstrate_layer_importance(model, test_loader, device='cpu'):
    """Analyze layer importance for pruning."""
    logger.info("\n" + "="*60)
    logger.info("LAYER IMPORTANCE ANALYSIS")
    logger.info("="*60)
    
    original_state = model.state_dict()
    layer_importance = {}
    
    # Test pruning each layer individually
    for target_name, target_module in model.named_modules():
        if isinstance(target_module, (nn.Linear, nn.Conv2d)):
            # Reset model
            model.load_state_dict(original_state)
            
            # Prune only this layer
            mask = PruningUtilities.get_pruning_mask_magnitude(
                target_module.weight.data, amount=0.5
            )
            PruningUtilities.apply_pruning_mask(target_module, mask)
            
            # Evaluate impact
            accuracy = evaluate_model(model, test_loader, device)
            baseline_accuracy = evaluate_model(model, test_loader, device)
            
            importance = baseline_accuracy - accuracy
            layer_importance[target_name] = importance
            
            logger.info(f"{target_name}: Accuracy drop = {importance:.2f}%")
    
    # Sort by importance
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("\nLayers ranked by importance:")
    for layer, importance in sorted_layers:
        logger.info(f"  {layer}: {importance:.2f}%")
    
    return layer_importance


def visualize_pruning_results(results, output_dir='./pruning_results'):
    """Create visualizations of pruning results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Accuracy vs Sparsity for different strategies
    plt.figure(figsize=(10, 6))
    
    for strategy, data in results['basic_pruning'].items():
        plt.plot(data['sparsity'], data['accuracy'], 'o-', label=strategy, linewidth=2, markersize=8)
    
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy (%)')
    plt.title('Pruning Strategy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison.png', dpi=150)
    plt.close()
    
    # 2. Structured pruning results
    if 'structured' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        data = results['structured']
        x = range(len(data['sparsity']))
        
        ax1.plot(x, data['accuracy'], 'b-', linewidth=2)
        ax1.set_xlabel('Pruning Ratio')
        ax1.set_ylabel('Accuracy (%)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2.bar(x, data['filters_removed'], alpha=0.7, color='orange')
        ax2.set_xlabel('Pruning Ratio')
        ax2.set_ylabel('Filters Removed', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        plt.suptitle('Structured Pruning Results')
        plt.tight_layout()
        plt.savefig(output_dir / 'structured_pruning.png', dpi=150)
        plt.close()
    
    # 3. Iterative pruning progress
    if 'iterative' in results:
        plt.figure(figsize=(10, 6))
        accuracies = results['iterative']
        iterations = range(1, len(accuracies) + 1)
        
        plt.plot(iterations, accuracies, 'g-', linewidth=2, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy (%)')
        plt.title('Iterative Pruning Progress')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'iterative_pruning.png', dpi=150)
        plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")


def main():
    """Run comprehensive pruning demonstration."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size=128)
    
    # Create and train model
    logger.info("Creating and training model...")
    model = ConvNet()
    train_model(model, train_loader, epochs=5, device=device)
    
    # Baseline evaluation
    baseline_accuracy = evaluate_model(model, test_loader, device)
    logger.info(f"\nBaseline accuracy: {baseline_accuracy:.2f}%")
    
    # Compute initial alignment metrics
    logger.info("\nComputing baseline alignment metrics...")
    baseline_metrics = compute_alignment_metrics(model, test_loader, device)
    for layer, metrics in baseline_metrics.items():
        logger.info(f"{layer}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Run demonstrations
    results = {}
    
    # 1. Basic pruning strategies
    results['basic_pruning'] = demonstrate_basic_pruning(model, test_loader, device)
    
    # 2. Structured pruning
    results['structured'] = demonstrate_structured_pruning(model, test_loader, device)
    
    # 3. Iterative pruning
    results['iterative'] = demonstrate_iterative_pruning(model, train_loader, test_loader, device)
    
    # 4. Pruning schedules
    demonstrate_pruning_schedules()
    
    # 5. Layer importance
    results['layer_importance'] = demonstrate_layer_importance(model, test_loader, device)
    
    # Create visualizations
    visualize_pruning_results(results)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PRUNING DEMONSTRATION COMPLETE")
    logger.info("="*60)
    logger.info("\nKey findings:")
    logger.info("1. Magnitude-based pruning generally outperforms random pruning")
    logger.info("2. Iterative pruning with fine-tuning can achieve high sparsity with minimal accuracy loss")
    logger.info("3. Different layers have different importance for the task")
    logger.info("4. Structured pruning can reduce model size while maintaining accuracy")
    logger.info("\nCheck ./pruning_results/ for visualizations")


if __name__ == "__main__":
    main() 