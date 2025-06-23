"""
Example usage of the refactored alignment metrics framework.

This script demonstrates the key features and improvements in the new architecture.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List
import logging

# Import all components
from alignment import (
    ModelWrapper,
    DatasetWrapper,
    RayleighQuotient,
    MutualInformationGaussian,
    discover_metrics,
    list_metrics
)

from alignment.experiments import ProgressiveDropoutExperiment
from alignment.analysis import ResultAnalyzer, MetricVisualizer

# Import from the refactored framework
from alignment.core import get_metric, get_experiment
from alignment.metrics.rayleigh import RayleighQuotient
from alignment.models import ModelWrapper
from alignment.data import MNISTDataset
from alignment.experiments import ProgressiveDropoutExperiment
from alignment.analysis import AlignmentPlotter


def example_basic_metric_computation():
    """Example 1: Basic metric computation."""
    print("=== Example 1: Basic Metric Computation ===")
    
    # Create synthetic data
    batch_size, input_features, output_features = 128, 784, 256
    inputs = torch.randn(batch_size, input_features)
    weights = torch.randn(output_features, input_features)
    
    # Method 1: Direct instantiation
    metric = RayleighQuotient(relative=True)
    scores = metric.compute(inputs=inputs, weights=weights)
    print(f"RQ scores shape: {scores.shape}")
    print(f"Mean RQ: {scores.mean():.4f}")
    
    # Method 2: Using registry
    metric = get_metric("rayleigh_quotient", relative=True)
    scores = metric.compute(inputs=inputs, weights=weights)
    print(f"Registry-based RQ mean: {scores.mean():.4f}\n")


def example_model_wrapper():
    """Example 2: Model wrapper with activation tracking."""
    print("=== Example 2: Model Wrapper ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Wrap model to track specific layers
    tracked_layers = ['0', '2', '4']  # Linear layers
    wrapper = ModelWrapper(model, tracked_layers=tracked_layers)
    
    # Forward pass with activation collection
    inputs = torch.randn(32, 784)
    outputs, activations = wrapper.forward_with_activations(inputs)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Tracked activations: {list(activations.keys())}")
    
    # Get layer weights
    weights = wrapper.get_layer_weights()
    print(f"Weight shapes: {[(k, v.shape) for k, v in weights.items()]}\n")


def example_distributed_computation():
    """Example 3: Distributed metric computation."""
    print("=== Example 3: Distributed Computation ===")
    
    # Simulate distributed environment
    world_size = 4
    rank = 0  # Current process rank
    
    # Create metric
    metric = RayleighQuotient()
    
    # Create local data (each process has its own subset)
    local_batch_size = 32
    inputs = torch.randn(local_batch_size, 512)
    weights = torch.randn(256, 512)
    
    # Compute with automatic distributed reduction
    scores = metric.compute_distributed(
        inputs=inputs,
        weights=weights,
        world_size=world_size,
        rank=rank
    )
    
    print(f"Distributed RQ scores shape: {scores.shape}")
    print(f"This simulates rank {rank} of {world_size} processes\n")


def example_memory_aware_computation():
    """Example 4: Memory-aware computation with CPU offloading."""
    print("=== Example 4: Memory-Aware Computation ===")
    
    # Create large tensors that might benefit from CPU computation
    large_size = 10000
    inputs = torch.randn(1000, large_size, device='cuda')
    weights = torch.randn(500, large_size, device='cuda')
    
    # Metric with automatic CPU offloading for large operations
    metric = RayleighQuotient(
        force_cpu_for_large_ops=True,
        cpu_threshold=1e7  # Move to CPU if > 10M elements
    )
    
    # This will automatically use CPU for covariance computation
    scores = metric.compute(inputs=inputs, weights=weights)
    print(f"Computed RQ for large tensors: {scores.shape}")
    print(f"Scores are on device: {scores.device}\n")


def example_experiment_runner():
    """Example 5: Running a complete experiment."""
    print("=== Example 5: Experiment Runner ===")
    
    # Configuration for the experiment
    config = {
        'metrics': ['rayleigh_quotient', 'mutual_information'],
        'pruning': {
            'mode': 'layer_wise',
            'dropout_range': [0.0, 0.9],
            'steps': 10,
            'exclude_final_layer': True
        },
        'num_batches': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create experiment
    experiment = ProgressiveDropoutExperiment(
        name="example_pruning",
        config=config,
        output_dir="results/example"
    )
    
    # Create model and dataset (pseudo-code, would use actual implementations)
    model = models.resnet18(pretrained=True)
    dataset = MNISTDataset()
    
    print("Would run progressive dropout experiment with:")
    print(f"  - Metrics: {config['metrics']}")
    print(f"  - Pruning mode: {config['pruning']['mode']}")
    print(f"  - Dropout steps: {config['pruning']['steps']}")
    
    # In practice:
    # results = experiment.run(model, dataset)
    # experiment.save_results(results)


def example_custom_metric():
    """Example 6: Creating a custom metric."""
    print("=== Example 6: Custom Metric ===")
    
    from alignment.core import BaseMetric, register_metric
    
    @register_metric("cosine_alignment")
    class CosineAlignment(BaseMetric):
        """Custom metric measuring cosine similarity between weights and input PCs."""
        
        @property
        def requires_inputs(self) -> bool:
            return True
        
        @property
        def requires_weights(self) -> bool:
            return True
        
        @property
        def requires_outputs(self) -> bool:
            return False
        
        def compute(self, inputs=None, weights=None, outputs=None, **kwargs):
            # Compute input covariance and get first PC
            cov = torch.cov(inputs.T)
            _, eigvecs = torch.linalg.eigh(cov)
            first_pc = eigvecs[:, -1]
            
            # Compute cosine similarity with each weight vector
            weight_norms = torch.norm(weights, dim=1, keepdim=True)
            normalized_weights = weights / (weight_norms + 1e-8)
            
            cosine_sim = torch.matmul(normalized_weights, first_pc)
            return torch.abs(cosine_sim)
    
    # Use the custom metric
    metric = get_metric("cosine_alignment")
    inputs = torch.randn(100, 64)
    weights = torch.randn(32, 64)
    scores = metric.compute(inputs=inputs, weights=weights)
    print(f"Custom metric scores: {scores.shape}, mean: {scores.mean():.4f}\n")


def example_metric_aggregation():
    """Example 7: Computing and aggregating multiple metrics."""
    print("=== Example 7: Metric Aggregation ===")
    
    from alignment.metrics import MetricComputer
    
    # Create metric computer with multiple metrics
    computer = MetricComputer(
        metrics=['rayleigh_quotient', 'weight_cosine_similarity'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        show_progress=True
    )
    
    # Create a model wrapper (pseudo-code)
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    wrapper = ModelWrapper(model, tracked_layers=['0', '2', '4'])
    
    # Compute metrics on a batch
    batch = torch.randn(64, 784)
    results = computer.compute_on_batch(wrapper, batch)
    
    print("Computed metrics:")
    for metric_name, layer_results in results.items():
        print(f"  {metric_name}:")
        for layer_name, values in layer_results.items():
            print(f"    Layer {layer_name}: mean={values.mean():.4f}")


if __name__ == "__main__":
    # Run all examples
    examples = [
        example_basic_metric_computation,
        example_model_wrapper,
        example_distributed_computation,
        example_memory_aware_computation,
        example_experiment_runner,
        example_custom_metric,
        example_metric_aggregation
    ]
    
    for example_func in examples:
        try:
            example_func()
        except ImportError as e:
            print(f"Skipping {example_func.__name__} - not all modules implemented yet")
            print(f"  Error: {e}\n")
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}\n")
    
    print("=== Examples Complete ===")
    print("This demonstrates the key features of the refactored framework:")
    print("1. Clean, modular API")
    print("2. Registry-based component discovery")
    print("3. Built-in distributed computing support")
    print("4. Memory-aware computation")
    print("5. Easy extensibility")
    print("6. Type-safe interfaces") 