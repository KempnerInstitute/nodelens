"""
Debug script that properly trains networks before testing pruning strategies.
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

def test_high_low_random_pruning(network, dataset, device="cuda"):
    """Test all three pruning strategies on a properly trained network."""
    print("\nEvaluating trained network with different pruning strategies")
    print("===========================================================")
    
    # Original network evaluation
    print(f"Original network:")
    accuracy, loss = evaluate_network(network, dataset, device)
    print(f"Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
    
    # Test different pruning percentages
    for prune_percent in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"\nPruning {prune_percent*100:.0f}% of neurons:")
        print("----------------------------------")
        
        # Strategy names
        strategies = ["high_rq", "low_rq", "random"]
        
        for strategy_idx, strategy in enumerate(strategies):
            # Store original weights
            original_weights = {}
            original_biases = {}
            
            for i, layer in enumerate(network.alignment_layers):
                if hasattr(layer, "weight") and layer.weight is not None:
                    original_weights[i] = layer.weight.data.clone()
                    if hasattr(layer, "bias") and layer.bias is not None:
                        original_biases[i] = layer.bias.data.clone()
            
            # Apply pruning to each layer based on strategy
            for i, layer in enumerate(network.alignment_layers):
                if i not in original_weights:
                    continue
                    
                # Compute magnitude as simple alignment proxy
                weights = layer.weight.data
                neuron_scores = [torch.norm(weights[:, j]).item() for j in range(weights.shape[1])]
                
                # Calculate how many to drop
                num_to_drop = int(len(neuron_scores) * prune_percent)
                
                if num_to_drop > 0:
                    # Get indices to drop based on strategy
                    if strategy == "high_rq":  # Drop highest alignment neurons
                        sorted_indices = np.argsort(neuron_scores)[::-1]  # Sort descending
                        to_drop = sorted_indices[:num_to_drop]
                    elif strategy == "low_rq":  # Drop lowest alignment neurons
                        sorted_indices = np.argsort(neuron_scores)  # Sort ascending
                        to_drop = sorted_indices[:num_to_drop]
                    else:  # Random pruning
                        all_indices = list(range(len(neuron_scores)))
                        np.random.shuffle(all_indices)
                        to_drop = all_indices[:num_to_drop]
                    
                    # Zero out weights for these neurons
                    for idx in to_drop:
                        if idx < layer.weight.data.shape[1]:
                            layer.weight.data[:, idx] = 0.0
                            if hasattr(layer, "bias") and layer.bias is not None and idx < layer.bias.data.shape[0]:
                                layer.bias.data[idx] = 0.0
            
            # Evaluate with pruning
            accuracy, loss = evaluate_network(network, dataset, device)
            print(f"{strategy}: Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            
            # Restore original weights
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
    network = train_network(network, dataset, device, epochs=5)
    
    # Now test pruning on the trained network
    test_high_low_random_pruning(network, dataset, device)

if __name__ == "__main__":
    main() 