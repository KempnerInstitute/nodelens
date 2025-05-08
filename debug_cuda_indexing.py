#!/usr/bin/env python

"""
Debug script to identify indexing or activation issues with the dropout implementation.
Tests very small dropout percentages to see where the accuracy starts degrading.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add the project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import from the alignment package
from alignment.config import ExperimentConfig
from alignment.datasets import load_dataset
from alignment.models.registry import create_model

def train_network(network, dataset, device="cuda", epochs=5):
    """Train a network on the given dataset."""
    network.train()
    network = network.to(device)
    
    # Get train loader
    train_loader = dataset.train_loader
    
    # Create optimizer
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
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
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate on test set
        test_acc, test_loss = evaluate_network(network, dataset, device)
        print(f"Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
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

def test_small_pruning_percentages(network, dataset, device="cuda"):
    """Test a spectrum of small pruning percentages to find where accuracy breaks."""
    print("\nTesting small dropout percentages to identify issues")
    print("=====================================================")
    
    # Original network evaluation
    print(f"Original network (0% pruning):")
    accuracy, loss = evaluate_network(network, dataset, device)
    print(f"Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
    
    # Print network details
    print("\nNetwork alignment layer details:")
    for i, layer in enumerate(network.alignment_layers):
        if hasattr(layer, "weight") and layer.weight is not None:
            print(f"Layer {i}: Shape {layer.weight.shape}, Device {layer.weight.device}")
    
    # Test very small percentages
    dropout_percents = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    for prune_percent in dropout_percents:
        print(f"\nPruning {prune_percent*100:.2f}% of neurons:")
        print("-" * 40)
        
        # Save original weights for each test
        original_weights = {}
        original_biases = {}
        
        for i, layer in enumerate(network.alignment_layers):
            if hasattr(layer, "weight") and layer.weight is not None:
                original_weights[i] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    original_biases[i] = layer.bias.data.clone()
        
        # Apply pruning strategy by strategy
        strategies = ["high_rq", "low_rq", "random"]
        
        for strategy in strategies:
            # Apply pruning to each layer based on strategy
            print(f"\nStrategy: {strategy}")
            
            # Track pruning statistics without verbose logging
            total_neurons = 0
            total_pruned = 0
            
            for i, layer in enumerate(network.alignment_layers):
                if i not in original_weights:
                    continue
                    
                # Restore original weights for each strategy test
                layer.weight.data = original_weights[i].clone()
                if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data = original_biases[i].clone()
                
                # Compute magnitude as alignment proxy
                weights = layer.weight.data
                input_dim = weights.shape[1]
                total_neurons += input_dim
                
                # Compute neuron scores
                neuron_scores = [torch.norm(weights[:, j]).item() for j in range(input_dim)]
                
                # Calculate how many to drop (ensure at least 1 for non-zero percentages)
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
                    
                    # Zero out weights for these neurons with safe indexing
                    for idx in to_drop:
                        if idx < weights.shape[1]:
                            layer.weight.data[:, idx] = 0.0
                            if hasattr(layer, "bias") and layer.bias is not None and idx < layer.bias.data.shape[0]:
                                layer.bias.data[idx] = 0.0
            
            # Report pruning statistics
            if prune_percent > 0:
                print(f"  Pruned {total_pruned}/{total_neurons} neurons ({total_pruned/total_neurons*100:.2f}%)")
            
            # Evaluate after pruning
            accuracy, loss = evaluate_network(network, dataset, device)
            print(f"  Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            
            # Verification: check if zeros were actually applied
            zero_verification = {}
            total_zeros = 0
            
            for i, layer in enumerate(network.alignment_layers):
                        if hasattr(layer, "weight") and layer.weight is not None:
                    # Count how many truly zero columns we have
                    weights = layer.weight.data
                    zero_cols = 0
                    for j in range(weights.shape[1]):
                        if torch.all(weights[:, j] == 0).item():
                            zero_cols += 1
                    
                    zero_verification[i] = zero_cols
                    total_zeros += zero_cols
            
            print(f"  Verification: {total_zeros} neurons were zeroed out")
            
            # Restore original weights after evaluation
            for i, layer in enumerate(network.alignment_layers):
                if i in original_weights:
                    layer.weight.data = original_weights[i].clone()
                    if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias.data = original_biases[i].clone()
    
    return

def main():
    """Main function to run the debug script."""
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
    
    # Train the network first
    print("\nTraining network before testing pruning:")
    network = train_network(network, dataset, device, epochs=3)
    
    # Now test pruning on the trained network
    test_small_pruning_percentages(network, dataset, device)

if __name__ == "__main__":
    main() 