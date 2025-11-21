"""
Standard Alignment Experiment

This script demonstrates a complete workflow:
1. Train a neural network on MNIST
2. Compute alignment metrics
3. Apply pruning based on different strategies
4. Generate visualizations
5. Compare results

This serves as a template for alignment analysis experiments.

Usage:
    python standard_alignment_experiment.py
    
No configuration needed - this script runs with default settings.
The script will:
    - Download MNIST dataset automatically (if not present)
    - Train a simple MLP model for 5 epochs
    - Compute alignment metrics (Rayleigh Quotient, Weight Similarity)
    - Test pruning at various sparsity levels (10%, 30%, 50%, 70%, 90%)
    - Compare magnitude vs random pruning strategies
    - Generate comprehensive visualizations

Requirements:
    - PyTorch with CUDA (optional, will use CPU if not available)
    - torchvision
    - alignment package installed
    - matplotlib for visualizations

Output:
    Results are saved to: results/standard_experiment/
    - training_history.json: Training metrics
    - alignment_metrics.json: Layer-wise alignment scores
    - pruning_results.json: Pruning experiment results
    - pruning_performance.png: Performance comparison plot
    - comparison_grid.png: Comprehensive analysis grid
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any

# Alignment framework imports
from alignment.models import ModelWrapper
from alignment.metrics import get_metric
from alignment.pruning import get_pruning_strategy, PruningConfig
from alignment.analysis.visualization import PruningVisualizer


class SimpleNet(nn.Module):
    """Simple feedforward network for MNIST."""
    def __init__(self, hidden_sizes=[784, 256, 128, 10]):
        super().__init__()
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < len(hidden_sizes) - 2:  # No ReLU after last layer
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def train_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        val_acc = 100. * correct / len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {train_loss:.4f}, Acc: {val_acc:.2f}%')
    
    return history


def compute_alignment_metrics(model, data_loader, device='cuda'):
    """Compute various alignment metrics for the model."""
    model = model.to(device)
    wrapped_model = ModelWrapper(model.model)  # Wrap the sequential model
    
    # Get a batch of data for metric computation
    data_iter = iter(data_loader)
    inputs, _ = next(data_iter)
    inputs = inputs.to(device)
    inputs = inputs.view(inputs.size(0), -1)  # Flatten for the model
    
    # Forward pass to collect activations
    _, activations = wrapped_model.forward_with_activations(inputs)
    weights = wrapped_model.get_layer_weights()
    
    # Initialize metrics
    metrics = {
        'rayleigh_quotient': get_metric('rayleigh_quotient'),
        'weight_cosine_similarity': get_metric('weight_cosine_similarity'),
    }
    
    results = {}
    
    # Compute metrics for each layer
    for layer_name in wrapped_model.tracked_layers:
        results[layer_name] = {}
        
        # Rayleigh Quotient
        if f"{layer_name}_input" in activations:
            layer_inputs = activations[f"{layer_name}_input"]
            layer_weights = weights[layer_name]
            
            rq_scores = metrics['rayleigh_quotient'].compute(
                inputs=layer_inputs, 
                weights=layer_weights
            )
            results[layer_name]['rayleigh_quotient'] = {
                'mean': float(rq_scores.mean()),
                'std': float(rq_scores.std()),
                'min': float(rq_scores.min()),
                'max': float(rq_scores.max())
            }
        
        # Weight Cosine Similarity
        layer_weights = weights[layer_name]
        sim_scores = metrics['weight_cosine_similarity'].compute(weights=layer_weights)
        results[layer_name]['weight_similarity'] = {
            'mean': float(sim_scores.mean()),
            'std': float(sim_scores.std())
        }
    
    return results


def apply_pruning_experiment(model, val_loader, strategies=['magnitude', 'random'], 
                           sparsities=[0.1, 0.3, 0.5, 0.7, 0.9], device='cuda'):
    """Apply different pruning strategies and evaluate."""
    results = {}
    
    for strategy_name in strategies:
        results[strategy_name] = {}
        
        for sparsity in sparsities:
            # Clone model for this experiment
            pruned_model = SimpleNet()
            pruned_model.load_state_dict(model.state_dict())
            pruned_model = pruned_model.to(device)
            
            # Apply pruning
            config = PruningConfig(amount=sparsity, pruning_mode='low')
            strategy = get_pruning_strategy(strategy_name, config=config)
            
            # Prune each linear layer
            for name, module in pruned_model.named_modules():
                if isinstance(module, nn.Linear):
                    mask = strategy.prune(module)
                    module.weight.data *= mask
            
            # Evaluate
            pruned_model.eval()
            correct = 0
            total_loss = 0
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = pruned_model(data)
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
            
            accuracy = 100. * correct / len(val_loader.dataset)
            avg_loss = total_loss / len(val_loader)
            
            results[strategy_name][sparsity] = {
                'accuracy': accuracy,
                'loss': avg_loss
            }
            
            print(f"{strategy_name} - Sparsity {sparsity:.1%}: "
                  f"Acc: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
    
    return results


def visualize_results(pruning_results, output_dir='results/standard_experiment'):
    """Generate visualizations of the results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = PruningVisualizer()
    
    # Create performance comparison plot
    fig1 = visualizer.plot_pruning_performance(
        pruning_results,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / 'pruning_performance.png',
        title='Pruning Strategy Comparison on MNIST',
        show_confidence=False
    )
    
    # Create comprehensive comparison grid
    fig2 = visualizer.plot_pruning_comparison_grid(
        pruning_results,
        save_path=output_dir / 'comparison_grid.png'
    )
    
    print(f"\nVisualizations saved to {output_dir}")


def main():
    """Run the complete experiment."""
    print("=" * 60)
    print("Standard Alignment Experiment")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data loading
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    model = SimpleNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, train_loader, val_loader, epochs=5, device=device)
    
    # Save training history
    output_dir = Path('results/standard_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Compute alignment metrics
    print("\nComputing alignment metrics...")
    metrics = compute_alignment_metrics(model, val_loader, device=device)
    
    print("\nAlignment Metrics:")
    for layer, layer_metrics in metrics.items():
        print(f"\nLayer {layer}:")
        for metric_name, values in layer_metrics.items():
            if 'mean' in values:
                print(f"  {metric_name}: mean={values['mean']:.4f}, std={values['std']:.4f}")
    
    # Save metrics
    with open(output_dir / 'alignment_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Pruning experiments
    print("\n" + "=" * 60)
    print("Pruning Experiments")
    print("=" * 60)
    
    pruning_results = apply_pruning_experiment(
        model, val_loader,
        strategies=['magnitude', 'random'],
        sparsities=[0.1, 0.3, 0.5, 0.7, 0.9],
        device=device
    )
    
    # Save pruning results
    with open(output_dir / 'pruning_results.json', 'w') as f:
        json.dump(pruning_results, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(pruning_results, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - training_history.json: Training loss and accuracy")
    print("  - alignment_metrics.json: Layer-wise alignment metrics")
    print("  - pruning_results.json: Pruning experiment results")
    print("  - pruning_performance.png: Performance comparison plot")
    print("  - comparison_grid.png: Comprehensive analysis grid")
    
    # Print key findings
    print("\nKey Findings:")
    baseline_acc = history['val_acc'][-1]
    print(f"  - Baseline accuracy: {baseline_acc:.2f}%")
    
    for strategy in pruning_results:
        acc_50 = pruning_results[strategy][0.5]['accuracy']
        acc_90 = pruning_results[strategy][0.9]['accuracy']
        print(f"  - {strategy.capitalize()} pruning:")
        print(f"      50% sparsity: {acc_50:.2f}% (drop: {baseline_acc - acc_50:.2f}%)")
        print(f"      90% sparsity: {acc_90:.2f}% (drop: {baseline_acc - acc_90:.2f}%)")


if __name__ == "__main__":
    main() 