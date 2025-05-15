"""
Standalone benchmark script for evaluating ML metrics.

This script bypasses the package structure to directly implement and test
the alignment metrics in isolation.
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Direct implementation of RQMetric
def compute_rq_metric(inputs, weights, relative=True, epsilon=1e-8):
    """Compute Representation Quality metric."""
    # Ensure inputs have at least 2 dimensions
    if inputs.dim() < 2:
        inputs = inputs.unsqueeze(0)
        
    # Move weights to same device as inputs
    weights = weights.to(inputs.device)
    
    # Center the inputs
    X = inputs - inputs.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov = torch.matmul(X.t(), X) / (X.size(0) - 1)
    
    # Add small value to diagonal for stability
    cov = cov + torch.eye(cov.size(0), device=cov.device) * epsilon
    
    # Compute the RQ values
    numerator = torch.sum(weights * torch.matmul(weights, cov), dim=1)
    denominator = (torch.norm(weights, dim=1) ** 2) * (torch.norm(weights @ cov, dim=1) + epsilon)
    
    # Calculate RQ as cosine similarity between weight vectors and weight @ covariance
    rq = numerator / denominator
    
    if relative:
        # Make RQ values relative to random vectors in high dimensions (expected value is 1/sqrt(d))
        d = weights.size(1)
        rq = rq * np.sqrt(d)
        
    return rq

# Create a simple MLP that we'll analyze
class MLP(torch.nn.Module):
    """Simple multi-layer perceptron."""
    
    def __init__(self, input_dim=784, hidden_dims=[100, 50], output_dim=10):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create layers
        self.layers = torch.nn.ModuleList()
        
        # Store linear layer indices for hooks
        self.linear_indices = {}
        linear_count = 0
        
        # Input layer
        curr_dim = input_dim
        
        # Hidden layers
        for h_dim in hidden_dims:
            linear_layer = torch.nn.Linear(curr_dim, h_dim)
            self.layers.append(linear_layer)
            self.linear_indices[linear_count] = len(self.layers) - 1
            linear_count += 1
            
            self.layers.append(torch.nn.ReLU())
            curr_dim = h_dim
            
        # Output layer
        linear_layer = torch.nn.Linear(curr_dim, output_dim)
        self.layers.append(linear_layer)
        self.linear_indices[linear_count] = len(self.layers) - 1
        
        # Register hooks to capture activations
        self.activations = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = input[0].detach()
            return hook
            
        # Register hooks for linear layers
        for linear_idx, layer_idx in self.linear_indices.items():
            self.layers[layer_idx].register_forward_hook(get_activation(f"linear_{linear_idx}"))
        
    def forward(self, x):
        """Forward pass."""
        # Reset activations
        self.activations = {}
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        return x
        
    def get_weight_layers(self):
        """Get layers with weights."""
        return [self.layers[idx] for idx in self.linear_indices.values()]
        
    def get_weights(self):
        """Get all weight matrices."""
        return [layer.weight.data for layer in self.get_weight_layers()]
    
    def get_activations(self):
        """Get activations for each layer."""
        activations = []
        
        for i in range(len(self.linear_indices)):
            layer_name = f"linear_{i}"
            if layer_name in self.activations:
                activations.append(self.activations[layer_name])
            else:
                # If no activation found (unlikely), add a placeholder
                activations.append(None)
                
        return activations

def benchmark_rq_metric(batch_size=64, input_dim=784, hidden_dims=[100, 50], output_dim=10, num_iterations=10):
    """Benchmark the RQ metric implementation."""
    # Create model
    model = MLP(input_dim, hidden_dims, output_dim)
    
    # Get weights
    weights = model.get_weights()
    
    # Create random input
    inputs = torch.randn(batch_size, input_dim)
    
    # Time the computation
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Forward pass to get activations
        _ = model(inputs)
        
        # Get activations for each layer
        activations = model.get_activations()
        
        # Compute RQ for each layer
        for i, (weight, activation) in enumerate(zip(weights, activations)):
            if activation is not None:
                rq = compute_rq_metric(activation, weight)
                
                # Just to make sure it's used
                if rq.mean().item() < 0:
                    print("Unlikely negative RQ value")
    
    elapsed = time.time() - start_time
    print(f"RQ metric benchmark complete: {elapsed:.4f} seconds ({elapsed/num_iterations:.4f} sec/iter)")

def compare_batched_vs_sequential(input_dim=784, hidden_dims=[256, 128, 64], output_dim=10, 
                                  batch_size=32, num_networks=10, num_iterations=5):
    """Compare batched vs. sequential processing."""
    print(f"Creating {num_networks} networks with architecture: {input_dim}->{hidden_dims}->{output_dim}")
    
    # Create multiple networks
    networks = [MLP(input_dim, hidden_dims, output_dim) for _ in range(num_networks)]
    
    # Create random input
    inputs = torch.randn(batch_size, input_dim)
    
    # Sequential approach
    print("\nTesting sequential approach:")
    seq_start = time.time()
    
    for _ in tqdm(range(num_iterations)):
        for net_idx, network in enumerate(networks):
            # Forward pass
            _ = network(inputs)
            
            # Get weights and activations
            weights = network.get_weights()
            activations = network.get_activations()
            
            # Compute RQ for each layer
            for weight, activation in zip(weights, activations):
                if activation is not None:
                    rq = compute_rq_metric(activation, weight)
    
    seq_time = time.time() - seq_start
    print(f"Sequential approach: {seq_time:.4f} seconds ({seq_time/num_iterations:.4f} sec/iter)")
    
    # Batched approach
    print("\nTesting batched approach (process networks in small batches):")
    batch_start = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Process networks in smaller batches
        network_batch_size = 3  # Process 3 networks at a time
        for i in range(0, num_networks, network_batch_size):
            batch_nets = networks[i:min(i+network_batch_size, num_networks)]
            
            for network in batch_nets:
                # Forward pass
                _ = network(inputs)
                
                # Get weights and activations
                weights = network.get_weights()
                activations = network.get_activations()
                
                # Compute RQ for each layer
                for weight, activation in zip(weights, activations):
                    if activation is not None:
                        rq = compute_rq_metric(activation, weight)
    
    batch_time = time.time() - batch_start
    print(f"Batched approach: {batch_time:.4f} seconds ({batch_time/num_iterations:.4f} sec/iter)")
    
    # Tensorized approach (if applicable)
    print("\nTesting tensorized approach (simplified version):")
    tensor_start = time.time()
    
    for _ in tqdm(range(num_iterations)):
        # Get all networks' outputs first
        for network in networks:
            _ = network(inputs)
        
        # Collect all weights and activations
        all_weights = []
        all_activations = []
        
        for network in networks:
            weights = network.get_weights()
            activations = network.get_activations()
            
            for w, a in zip(weights, activations):
                if a is not None:
                    all_weights.append(w)
                    all_activations.append(a)
            
        # Process all weights at once
        for weight, activation in zip(all_weights, all_activations):
            rq = compute_rq_metric(activation, weight)
    
    tensor_time = time.time() - tensor_start
    print(f"Tensorized approach: {tensor_time:.4f} seconds ({tensor_time/num_iterations:.4f} sec/iter)")
    
    # Print summary
    print("\nBenchmark Results:")
    print(f"Sequential: {seq_time:.4f}s ({1.0:.2f}x speedup)")
    print(f"Batched:    {batch_time:.4f}s ({seq_time/batch_time:.2f}x speedup)")
    print(f"Tensorized: {tensor_time:.4f}s ({seq_time/tensor_time:.2f}x speedup)")

if __name__ == "__main__":
    # Basic RQ benchmark
    print("\n=== Basic RQ Metric Benchmark ===")
    benchmark_rq_metric(batch_size=64, num_iterations=5)
    
    # Compare different processing approaches
    print("\n=== Comparing Processing Approaches ===")
    compare_batched_vs_sequential(num_networks=6, num_iterations=2) 