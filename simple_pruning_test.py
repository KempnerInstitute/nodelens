#!/usr/bin/env python
"""
Simple Pruning Test

This script tests pruning functionality with a minimal standalone implementation.
"""

import os
import sys
import copy
import logging
import numpy as np
import torch
import torch.nn as nn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pruning_test_output.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Create a simple model with alignment layers
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        # Define alignment layers for pruning
        self.alignment_layers = [self.fc1, self.fc2, self.fc3]
        self.alignment_names = ['layer1', 'layer2', 'layer3']
        self.hidden = {}
        
    def forward(self, x):
        # First layer
        self.hidden['layer1'] = x
        x = self.relu(self.fc1(x))
        
        # Second layer
        self.hidden['layer2'] = x
        x = self.relu(self.fc2(x))
        
        # Final layer
        self.hidden['layer3'] = x
        out = self.fc3(x)
        
        return out

# Generate synthetic data
def generate_data(batch_size=32, input_dim=784, num_classes=10):
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y

# Count zero and non-zero weights in a model
def count_weights(model):
    zero_weights = 0
    total_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            zero_count = (param.data == 0).sum().item()
            total_count = param.data.numel()
            zero_weights += zero_count
            total_weights += total_count
            logger.info(f"{name}: {zero_count}/{total_count} zeros "
                      f"({100.0*zero_count/total_count:.2f}%)")
    
    if total_weights > 0:
        logger.info(f"Total: {zero_weights}/{total_weights} zeros "
                  f"({100.0*zero_weights/total_weights:.2f}%)")
    
    return zero_weights, total_weights

# Directly prune weights in a model
def prune_model(model, strategy='low_rq', fraction=0.5, is_scaled=False):
    """Apply pruning to a model directly."""
    logger.info(f"Pruning with strategy={strategy}, fraction={fraction}, scaled={is_scaled}")
    
    # Clone the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)
    
    # Apply pruning to each alignment layer
    for layer_idx, layer in enumerate(pruned_model.alignment_layers):
        if not hasattr(layer, 'weight') or layer.weight is None:
            continue
        
        # Get layer dimensions
        weights = layer.weight.data
        output_dim = weights.shape[0]  # Number of neurons (rows)
        
        # Calculate how many neurons to prune
        num_to_drop = int(output_dim * fraction)
        logger.info(f"Layer {layer_idx}: Pruning {num_to_drop}/{output_dim} neurons ({fraction*100:.1f}%)")
        
        if num_to_drop == 0:
            continue
            
        # Calculate neuron importance (L2 norm of each neuron's weights)
        neuron_scores = [torch.norm(weights[j, :]).item() for j in range(output_dim)]
        
        # Determine which neurons to drop based on strategy
        if strategy == 'high_rq':  # Highest weight magnitudes
            sorted_indices = np.argsort(neuron_scores)[::-1]
            to_drop = sorted_indices[:num_to_drop]
        elif strategy == 'low_rq':  # Lowest weight magnitudes
            sorted_indices = np.argsort(neuron_scores)
            to_drop = sorted_indices[:num_to_drop]
        else:  # Random
            all_indices = list(range(output_dim))
            np.random.shuffle(all_indices)
            to_drop = all_indices[:num_to_drop]
        
        # Apply pruning - zero out weights for selected neurons
        if is_scaled:
            # Create a mask tensor
            mask = torch.ones_like(weights)
            mask[to_drop, :] = 0
            
            # Calculate scaling factor
            scaling_factor = len(to_drop) / output_dim
            scale = 1.0 / (1.0 - scaling_factor) if scaling_factor < 0.9 else 10.0
            
            # Apply mask and scaling
            layer.weight.data = layer.weight.data * mask * scale
            
            # Zero out biases if present
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_mask = torch.ones_like(layer.bias)
                bias_mask[to_drop] = 0
                layer.bias.data = layer.bias.data * bias_mask * scale
        else:
            # Simply zero out weights (no scaling)
            for idx in to_drop:
                layer.weight.data[idx, :] = 0.0
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data[idx] = 0.0
    
    return pruned_model

# Evaluate a model on data
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return accuracy, avg_loss

# Main testing function
def test_pruning(pruning_fractions=[0.0, 0.3, 0.5, 0.7, 0.9],
                strategy='low_rq', scaled=False):
    """Test pruning functionality with various fractions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create a simple model
    input_dim = 784
    hidden_dim = 100
    output_dim = 10
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    
    # Generate synthetic data
    batch_size = 64
    num_batches = 10
    data_loader = [generate_data(batch_size, input_dim, output_dim) for _ in range(num_batches)]
    data_loader = [(x.to(device), y.to(device)) for x, y in data_loader]
    
    # Evaluate original model
    logger.info("Evaluating original model...")
    orig_accuracy, orig_loss = evaluate(model, data_loader, device)
    logger.info(f"Original model: accuracy={orig_accuracy:.2f}%, loss={orig_loss:.4f}")
    
    # Count weights in original model
    logger.info("Weights in original model:")
    count_weights(model)
    
    # Test each pruning fraction
    results = []
    for fraction in pruning_fractions:
        logger.info(f"\nTesting pruning fraction: {fraction}")
        
        # Apply pruning
        pruned_model = prune_model(model, strategy, fraction, scaled)
        
        # Count weights in pruned model
        logger.info("Weights in pruned model:")
        zero_weights, total_weights = count_weights(pruned_model)
        
        # Evaluate pruned model
        pruned_accuracy, pruned_loss = evaluate(pruned_model, data_loader, device)
        logger.info(f"Pruned model: accuracy={pruned_accuracy:.2f}%, loss={pruned_loss:.4f}")
        
        # Calculate weight change
        pruning_percent = 100.0 * zero_weights / total_weights if total_weights > 0 else 0.0
        
        # Store results
        results.append({
            'fraction': fraction,
            'accuracy': pruned_accuracy,
            'loss': pruned_loss,
            'pruning_percent': pruning_percent
        })
    
    # Print summary
    logger.info("\nSUMMARY:")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Scaled: {scaled}")
    logger.info(f"{'-'*40}")
    for result in results:
        logger.info(f"{result['fraction']:10.2f} {result['accuracy']:10.2f} {result['loss']:10.4f} {result['pruning_percent']:10.2f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pruning functionality")
    parser.add_argument("--strategy", type=str, default="low_rq", 
                      choices=["high_rq", "low_rq", "random"],
                      help="Pruning strategy")
    parser.add_argument("--scaled", action="store_true", 
                      help="Use scaled pruning (rescale remaining weights)")
    
    args = parser.parse_args()
    
    # Run test
    test_pruning(strategy=args.strategy, scaled=args.scaled) 