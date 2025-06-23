#!/usr/bin/env python
"""
Benchmark script for comparing network training methods.

This script benchmarks different training methods for multiple networks:
1. Sequential training (original approach)
2. Tensorized training (improved parallel approach)
3. Fully tensorized training (optimized ensemble approach)
"""

import os
import time
import argparse
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

from alignment.metrics import get_metric
from alignment.datasets import load_dataset
from alignment.models.models import MLP, create_mlp
from alignment.models.base import AlignmentNetwork
from alignment.training import (
    train_networks_sequential,
    train_networks_tensorized,
    train_networks_fully_tensorized,
    train_networks
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_networks(num_networks, input_size=784, hidden_sizes=[512, 256], output_size=10, seed=42):
    """Create multiple networks with identical architecture but different initializations."""
    networks = []
    
    for i in range(num_networks):
        # Set seed for reproducibility, but different for each network
        torch.manual_seed(seed + i)
        
        # Create a simple MLP network
        base_model = MLP(
            input_dim=input_size,
            output_dim=output_size,
            num_hidden=hidden_sizes,
            dropout_rate=0.0
        )
        
        # Get all linear layers for alignment
        linear_layers = {}
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                if name != "layers.0":  # Skip the first layer
                    linear_layers[name] = None  # Use the layer's own input
        
        # Create AlignmentNetwork wrapper
        network = AlignmentNetwork(base_model=base_model, alignment_layer_names=linear_layers)
        
        # Move to device (will be moved again in training, but good practice)
        if torch.cuda.is_available():
            network.to('cuda')
        
        networks.append(network)
    
    logger.info(f"Created {num_networks} networks with architecture: {input_size} -> {hidden_sizes} -> {output_size}")
    return networks

def run_benchmark(networks, dataset, num_epochs=2, learning_rate=0.001, device=None, show_progress=True):
    """Run benchmark comparing different training methods."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Running benchmark on device: {device}")
    
    # Define methods to benchmark
    methods = [
        ("Sequential", train_networks_sequential),
        ("Tensorized", train_networks_tensorized),
        ("Fully Tensorized", train_networks_fully_tensorized),
        ("Auto-select", train_networks)
    ]
    
    results = {}
    
    # Clone networks for each method to ensure fair comparison
    for method_name, method_func in methods:
        logger.info(f"Benchmarking {method_name} training...")
        
        # Create copies of networks for this method
        method_networks = []
        
        # For each network in the original list
        for i, source_net in enumerate(networks):
            # Get the architecture of the source network
            input_dim = 784  # MNIST standard
            output_dim = 10  # MNIST standard
            
            # Extract hidden sizes by examining the linear layers
            linear_layers = [
                module for module in source_net.base_model.modules() 
                if isinstance(module, nn.Linear)
            ]
            
            # Get all but the last layer's output dimensions for hidden sizes
            hidden_sizes = [layer.out_features for layer in linear_layers[:-1]]
            
            logger.debug(f"Network {i}: input_dim={input_dim}, hidden_sizes={hidden_sizes}, output_dim={output_dim}")
            
            # Create base model with identical architecture
            base_model = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                num_hidden=hidden_sizes,
                dropout_rate=0.0
            )
            
            # Create alignment layer mapping identical to source network
            # Use the layer_to_input_names attribute which is the internal representation
            layer_to_input_names = source_net.layer_to_input_names
            
            # Create new network with same architecture and alignment layers
            target_net = AlignmentNetwork(
                base_model=base_model, 
                alignment_layer_names=layer_to_input_names
            )
            
            # Copy parameters from source to target
            # This would be the most reliable way to ensure identical networks
            target_net.load_state_dict(source_net.state_dict())
            
            # Move to device
            target_net.to(device)
            
            # Add to method networks
            method_networks.append(target_net)
        
        # Measure time
        start_time = time.time()
        
        # Train networks using this method
        training_history = method_func(
            networks=method_networks,
            dataset=dataset,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            show_progress=show_progress
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Store results
        results[method_name] = {
            "time": elapsed_time,
            "history": training_history,
            "final_acc": training_history["test_acc"][-1] if training_history["test_acc"] else 0.0
        }
        
        logger.info(f"{method_name} training completed in {elapsed_time:.2f} seconds (final acc: {results[method_name]['final_acc']:.2f}%)")
    
    # Print summary
    print("\nBenchmark Results:")
    print("==================")
    print(f"Networks: {len(networks)}, Epochs: {num_epochs}")
    print("------------------")
    
    # Find the fastest method to calculate speedup
    fastest_time = min(results[method]["time"] for method in results)
    
    for method_name in results:
        elapsed_time = results[method_name]["time"]
        final_acc = results[method_name]["final_acc"]
        speedup = fastest_time / elapsed_time if elapsed_time > 0 else 0
        
        print(f"{method_name:16s}: {elapsed_time:.2f}s ({speedup:.2f}x), Acc: {final_acc:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark network training methods.")
    parser.add_argument("--num_networks", type=int, default=5, help="Number of networks to train")
    parser.add_argument("--hidden_sizes", type=str, default="512,256", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run benchmark on")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Parse hidden sizes
    hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create networks
    networks = create_networks(
        num_networks=args.num_networks,
        hidden_sizes=hidden_sizes,
        seed=args.seed
    )
    
    # Load dataset
    logger.info("Loading MNIST dataset")
    transform_params = {
        "center_crop": None,
        "resize": None,
        "flatten": True,  # Flatten for MLP
        "normalize": False
    }
    dataset_config = {
        "dataset_name": "MNIST",
        "batch_size": args.batch_size,
        "data_path": "./data",
        "transform_params": transform_params
    }
    dataset = load_dataset(dataset_config)
    
    # Run benchmark
    run_benchmark(
        networks=networks,
        dataset=dataset,
        num_epochs=args.epochs,
        device=device
    )

if __name__ == "__main__":
    main() 