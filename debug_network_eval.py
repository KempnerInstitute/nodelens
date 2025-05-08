"""
Debug script to test if networks are properly evaluated.
This helps isolate issues with the progressive dropout implementation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add the project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import from the alignment package
from alignment.config import ExperimentConfig
from alignment.datasets import get_dataset, load_dataset
from alignment.models.registry import create_model

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

def debug_network_pruning(network, dataset, device="cuda"):
    """Debug network pruning by evaluating the network with different pruning levels."""
    print(f"Original network evaluation:")
    accuracy, loss = evaluate_network(network, dataset, device)
    print(f"Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
    
    # Now test with simulated pruning of 10%, 50%, and 90%
    for prune_percent in [0.1, 0.5, 0.9]:
        print(f"\nTesting with {prune_percent*100:.0f}% pruning:")
        
        # Store original weights
        original_weights = {}
        original_biases = {}
        
        # Get alignment layers
        if not hasattr(network, "alignment_layers"):
            print("Network doesn't have alignment_layers attribute")
            return
        
        # Store original weights
        for i, layer in enumerate(network.alignment_layers):
            if hasattr(layer, "weight") and layer.weight is not None:
                original_weights[i] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    original_biases[i] = layer.bias.data.clone()
        
        # Apply pruning to each layer
        for i, layer in enumerate(network.alignment_layers):
            if i not in original_weights:
                continue
                
            # Compute magnitude as simple alignment proxy
            weights = layer.weight.data
            neuron_scores = [torch.norm(weights[:, j]).item() for j in range(weights.shape[1])]
            
            # Calculate how many to drop
            num_to_drop = int(len(neuron_scores) * prune_percent)
            
            if num_to_drop > 0:
                # Sort indices by score (ascending)
                sorted_indices = np.argsort(neuron_scores)
                
                # Get indices to drop (lowest scores)
                to_drop = sorted_indices[:num_to_drop]
                
                # Zero out weights for these neurons
                for idx in to_drop:
                    if idx < layer.weight.data.shape[1]:
                        layer.weight.data[:, idx] = 0.0
                        if hasattr(layer, "bias") and layer.bias is not None and idx < layer.bias.data.shape[0]:
                            layer.bias.data[idx] = 0.0
        
        # Evaluate with pruning
        accuracy, loss = evaluate_network(network, dataset, device)
        print(f"Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        
        # Restore original weights
        for i, layer in enumerate(network.alignment_layers):
            if i in original_weights:
                layer.weight.data = original_weights[i].clone()
                if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data = original_biases[i].clone()

def main():
    """Main function to run the debug script."""
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
    
    # Debug the network pruning
    debug_network_pruning(network, dataset, device)

if __name__ == "__main__":
    main() 