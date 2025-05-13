#!/usr/bin/env python
"""
Test script for the cascading layer pruning method.

This script tests the new cascading layer pruning method.
It creates a small network and applies the cascading pruning method.
"""

import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from alignment.dropout import progressive_dropout
from alignment.datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a simple alignment metric for testing
class SimpleAlignmentMetric:
    """Simple alignment metric that uses L2 norm of weights."""
    
    def __init__(self):
        self.name = "simple_norm"
        
    def compute_per_node_scores(self, layer_input, layer_weights, device=None):
        """Compute score based on L2 norm of weights."""
        if device is None:
            device = layer_input.device
        
        # If weights are not provided, use random weights for testing
        if layer_weights is None:
            layer_weights = torch.randn(layer_input.size(1), layer_input.size(1), device=device)
        
        # Return L2 norm of each weight vector
        return torch.norm(layer_weights, dim=1)
    
    def measure(self, hidden, targets=None, num_classes=10):
        """Measure alignment for multiple layers."""
        return [torch.mean(torch.norm(h, dim=1)).item() for h in hidden]
    
    def compute_alignment(self, weights):
        """Compute alignment score using L2 norm for weight vector."""
        return float(np.linalg.norm(weights))
    
    def measure_neuron(self, weights):
        """Measure neuron importance based on weight vector."""
        return float(np.linalg.norm(weights))

# Define a simple network for testing
class SimpleNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[32, 16], output_size=10):
        super(SimpleNetwork, self).__init__()
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Output layer
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Define alignment layers for pruning
        self.alignment_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        self.alignment_names = [f"linear_{i}" for i in range(len(self.alignment_layers))]
        
        # Storage for activations
        self.hidden = {}
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Pass through layers
        for i, layer in enumerate(self.layers):
            # For linear layers, save input for alignment computation
            if isinstance(layer, nn.Linear):
                layer_name = self.alignment_names[self.alignment_layers.index(layer)]
                self.hidden[layer_name] = x
            
            # Forward pass through layer
            x = layer(x)
        
        return x

def test_cascading_pruning():
    """
    Test the cascading layer pruning method.
    """
    logger.info("Testing cascading layer pruning method")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create a dataset
    logger.info("Loading MNIST dataset")
    transform_params = {
        "center_crop": None,
        "resize": None,
        "flatten": True,  # Flatten for MLP
        "normalize": False
    }
    dataset_config = {
        "dataset_name": "MNIST", 
        "batch_size": 32, 
        "data_path": "./data",
        "transform_params": transform_params
    }
    dataset = load_dataset(dataset_config)
    
    # Create networks
    logger.info("Creating networks")
    num_networks = 1
    networks = [SimpleNetwork().to(device) for _ in range(num_networks)]
    
    # Initialize networks with different weights
    for i, net in enumerate(networks):
        torch.manual_seed(i)  # For reproducibility
        for layer in net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    # Define dropout fractions
    dropout_fractions = [0.0, 0.5]
    
    # Create alignment metric
    metric = SimpleAlignmentMetric()
    
    # Apply progressive dropout with cascading layer pruning
    logger.info("Testing cascading layer pruning")
    
    # Time the method
    start_time = time.time()
    
    # Apply progressive dropout
    accuracies, losses = progressive_dropout(
        networks=networks,
        dataset=dataset,
        dropout_fractions=dropout_fractions,
        metric=metric,
        device=device,
        pruning_mode="cascading_layer",
        dropout_mode="scaled",
        strategy="low_rq",
        show_progress=True
    )
    
    # Record time
    elapsed_time = time.time() - start_time
    
    # Print results
    logger.info(f"Cascading layer pruning completed in {elapsed_time:.2f} seconds")
    
    for net_idx, accs in accuracies.items():
        logger.info(f"Network {net_idx} accuracies:")
        for i, acc in enumerate(accs):
            fraction = dropout_fractions[i] if i < len(dropout_fractions) else "N/A"
            logger.info(f"  {fraction*100 if isinstance(fraction, float) else fraction}% dropout: {acc:.2f}% accuracy")
    
    # Return success
    return True

if __name__ == "__main__":
    success = test_cascading_pruning()
    sys.exit(0 if success else 1) 