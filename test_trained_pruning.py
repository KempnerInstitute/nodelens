"""
Test script with proper network training and pruning evaluation.
This script demonstrates the correct approach for the pruning experiment.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import from the alignment package
from alignment.config import ExperimentConfig
from alignment.datasets import load_dataset
from alignment.models.registry import create_model

def train_network(network, dataset, device="cuda", epochs=10):
    """Train a network on the given dataset."""
    print(f"\nTraining network for {epochs} epochs:")
    print("-" * 40)
    
    network.train()
    network = network.to(device)
    
    # Get train loader
    train_loader = dataset.train_loader
    
    # Create optimizer
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = network(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Print epoch statistics
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Evaluate on test set
        test_acc, test_loss = evaluate_network(network, dataset, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    # Create a training curve plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    os.makedirs("debug_output", exist_ok=True)
    plt.savefig("debug_output/training_curves.png")
    print(f"Saved training curves to debug_output/training_curves.png")
    
    return network

def evaluate_network(network, dataset, device="cuda"):
    """Evaluate a network's accuracy and loss on a dataset."""
    network.eval()
    network = network.to(device)
    
    # Get test loader
    test_loader = dataset.test_loader
    
    # Track metrics
    correct = 0
    total = 0
    total_loss = 0.0
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Compute loss
            loss = F.cross_entropy(outputs, targets, reduction='sum')
            total_loss += loss.item()
    
    # Calculate metrics
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return accuracy, avg_loss

def test_pruning_strategies(network, dataset, device="cuda"):
    """Test different pruning strategies on a trained network."""
    print("\nTesting pruning strategies on trained network")
    print("=" * 50)
    
    # Original network evaluation
    print(f"Original network (0% pruning):")
    orig_accuracy, orig_loss = evaluate_network(network, dataset, device)
    print(f"Accuracy: {orig_accuracy:.2f}%, Loss: {orig_loss:.4f}")
    
    # Define pruning percentages to test
    pruning_percents = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Initialize results dictionary
    results = {
        'pruning_percents': pruning_percents,
        'high_rq_acc': [],
        'low_rq_acc': [],
        'random_acc': [],
        'high_rq_loss': [],
        'low_rq_loss': [],
        'random_loss': []
    }
    
    # Save original weights
    original_weights = {}
    original_biases = {}
    
    for i, layer in enumerate(network.alignment_layers):
        if hasattr(layer, "weight") and layer.weight is not None:
            original_weights[i] = layer.weight.data.clone()
            if hasattr(layer, "bias") and layer.bias is not None:
                original_biases[i] = layer.bias.data.clone()
    
    # Test each pruning percentage
    for prune_percent in pruning_percents:
        print(f"\nPruning {prune_percent*100:.1f}% of neurons:")
        print("-" * 40)
        
        # Test each strategy
        for strategy in ["high_rq", "low_rq", "random"]:
            # Restore original weights
            for i, layer in enumerate(network.alignment_layers):
                if i in original_weights:
                    layer.weight.data = original_weights[i].clone()
                    if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias.data = original_biases[i].clone()
            
            # Skip pruning for 0% case
            if prune_percent == 0.0:
                acc = orig_accuracy
                loss = orig_loss
            else:
                # Apply pruning based on strategy
                total_neurons = 0
                total_pruned = 0
                
                for i, layer in enumerate(network.alignment_layers):
                    if i not in original_weights:
                        continue
                    
                    # Compute neuron importance scores (using weight magnitude as proxy)
                    weights = layer.weight.data
                    input_dim = weights.shape[1]
                    total_neurons += input_dim
                    
                    neuron_scores = [torch.norm(weights[:, j]).item() for j in range(input_dim)]
                    
                    # Calculate how many neurons to prune
                    num_to_drop = max(1, int(input_dim * prune_percent)) if prune_percent > 0 else 0
                    total_pruned += num_to_drop
                    
                    if num_to_drop > 0:
                        # Get indices to drop based on strategy
                        if strategy == "high_rq":  # Drop highest alignment neurons
                            sorted_indices = np.argsort(neuron_scores)[::-1]  # Sort descending
                            to_drop = sorted_indices[:num_to_drop]
                        elif strategy == "low_rq":  # Drop lowest alignment neurons
                            sorted_indices = np.argsort(neuron_scores)  # Sort ascending
                            to_drop = sorted_indices[:num_to_drop]
                        else:  # Random pruning
                            all_indices = list(range(input_dim))
                            np.random.shuffle(all_indices)
                            to_drop = all_indices[:num_to_drop]
                        
                        # Zero out weights for these neurons
                        for idx in to_drop:
                            if idx < weights.shape[1]:
                                layer.weight.data[:, idx] = 0.0
                                if hasattr(layer, "bias") and layer.bias is not None and idx < layer.bias.data.shape[0]:
                                    layer.bias.data[idx] = 0.0
                
                # Evaluate the pruned network
                acc, loss = evaluate_network(network, dataset, device)
            
            # Store results
            if strategy == "high_rq":
                results['high_rq_acc'].append(acc)
                results['high_rq_loss'].append(loss)
            elif strategy == "low_rq":
                results['low_rq_acc'].append(acc)
                results['low_rq_loss'].append(loss)
            else:
                results['random_acc'].append(acc)
                results['random_loss'].append(loss)
            
            # Print results
            print(f"{strategy.ljust(8)}: Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
    
    # Restore original weights
    for i, layer in enumerate(network.alignment_layers):
        if i in original_weights:
            layer.weight.data = original_weights[i].clone()
            if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data = original_biases[i].clone()
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(pruning_percents, results['high_rq_acc'], 'o-', label='high_rq (prune highest)')
    plt.plot(pruning_percents, results['low_rq_acc'], 's-', label='low_rq (prune lowest)')
    plt.plot(pruning_percents, results['random_acc'], '^-', label='random')
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Accuracy (%)')
    plt.title('Impact of Pruning on Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(pruning_percents, results['high_rq_loss'], 'o-', label='high_rq (prune highest)')
    plt.plot(pruning_percents, results['low_rq_loss'], 's-', label='low_rq (prune lowest)')
    plt.plot(pruning_percents, results['random_loss'], '^-', label='random')
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Loss')
    plt.title('Impact of Pruning on Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("debug_output", exist_ok=True)
    plt.savefig("debug_output/pruning_comparison.png")
    print(f"Saved pruning comparison to debug_output/pruning_comparison.png")
    
    return results

def main():
    """Main function to run the test script."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config_path = "configs/config_alignment_experiment.yaml"
    config = ExperimentConfig.load(config_path)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {config.dataset.dataset_name}")
    dataset = load_dataset(config.dataset)
    
    # Create a model
    print(f"Creating model: {config.model.model_name}")
    network = create_model(config.model)
    
    # Train the network first (this is key!)
    network = train_network(network, dataset, device, epochs=10)
    
    # Now test pruning strategies on the trained network
    test_pruning_strategies(network, dataset, device)

if __name__ == "__main__":
    main() 