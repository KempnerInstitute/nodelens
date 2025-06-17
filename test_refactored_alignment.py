#!/usr/bin/env python3
"""
Test script for the refactored alignment module.
Tests model wrapping, metric computation, and dropout-based pruning functionality.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add alignment module to path
sys.path.insert(0, 'src')

from alignment.models import ModelWrapper
from alignment.metrics.rayleigh import RayleighQuotient
from alignment.metrics.information import MutualInformationGaussian
from alignment.data.datasets import CIFAR10Dataset
from alignment.experiments.layer_isolated import LayerIsolatedPruningExperiment, LayerIsolatedConfig
from alignment.experiments.progressive_dropout import ProgressiveDropoutExperiment
from alignment.experiments.base import ExperimentConfig


class SimpleNet(nn.Module):
    """Simple test network."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test_model_wrapper():
    """Test ModelWrapper functionality."""
    print("\n=== Testing ModelWrapper ===")
    
    # Create model
    model = SimpleNet()
    print(f"Created model: {model.__class__.__name__}")
    
    # Print model structure to see layer names
    print("\nModel structure:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            print(f"  {name}: {module}")
    
    # Wrap model - specify layers explicitly
    tracked_layers = ['features.0', 'features.3', 'classifier.0', 'classifier.2']
    wrapped_model = ModelWrapper(model, tracked_layers=tracked_layers)
    print(f"\nWrapped model with tracked layers: {wrapped_model.tracked_layers}")
    
    # Test auto-discovery
    auto_wrapped = ModelWrapper(model)
    print(f"Auto-discovered layers: {auto_wrapped.tracked_layers}")
    
    # Test forward pass with activation tracking
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    
    outputs, activations = wrapped_model.forward_with_activations(x)
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Activations collected: {list(activations.keys())}")
    
    # Test weight retrieval
    weights = wrapped_model.get_layer_weights()
    print(f"Weights collected: {list(weights.keys())}")
    
    # Verify shapes
    for layer_name in wrapped_model.tracked_layers:
        if f"{layer_name}_input" in activations:
            print(f"{layer_name} input shape: {activations[f'{layer_name}_input'].shape}")
        if f"{layer_name}_output" in activations:
            print(f"{layer_name} output shape: {activations[f'{layer_name}_output'].shape}")
        if layer_name in weights:
            print(f"{layer_name} weight shape: {weights[layer_name].shape}")
    
    return wrapped_model, x


def test_metrics(wrapped_model, x):
    """Test metric computation."""
    print("\n=== Testing Metrics ===")
    
    # Get activations and weights
    outputs, activations = wrapped_model.forward_with_activations(x)
    weights = wrapped_model.get_layer_weights()
    
    # Test Rayleigh Quotient
    rq_metric = RayleighQuotient()
    print("\nTesting Rayleigh Quotient:")
    
    for layer_name in wrapped_model.tracked_layers:
        layer_inputs = activations.get(f"{layer_name}_input")
        layer_weights = weights.get(layer_name)
        
        if layer_inputs is not None and layer_weights is not None:
            try:
                # Debug shapes
                print(f"  {layer_name}: input shape = {layer_inputs.shape}, weight shape = {layer_weights.shape}")
                
                # Compute RQ
                rq_score = rq_metric.compute(layer_inputs, layer_weights)
                print(f"  {layer_name}: RQ = {rq_score.mean().item():.4f}")
            except Exception as e:
                print(f"  {layer_name}: Error - {e}")
    
    # Test Mutual Information
    mi_metric = MutualInformationGaussian()
    print("\nTesting Mutual Information:")
    
    # For MI, we need two sets of activations
    x2 = torch.randn_like(x)
    outputs2, activations2 = wrapped_model.forward_with_activations(x2)
    
    for layer_name in wrapped_model.tracked_layers:
        layer_acts1 = activations.get(f"{layer_name}_output") or activations.get(layer_name)
        layer_acts2 = activations2.get(f"{layer_name}_output") or activations2.get(layer_name)
        
        if layer_acts1 is not None and layer_acts2 is not None:
            try:
                # Reshape activations if needed
                if layer_acts1.dim() > 2:
                    layer_acts1 = layer_acts1.view(layer_acts1.size(0), -1)
                    layer_acts2 = layer_acts2.view(layer_acts2.size(0), -1)
                
                mi_score = mi_metric.compute(layer_acts1, layer_acts2)
                print(f"  {layer_name}: MI = {mi_score:.4f}")
            except Exception as e:
                print(f"  {layer_name}: Error - {e}")


def test_layer_isolated_pruning():
    """Test layer-isolated pruning functionality."""
    print("\n=== Testing Layer-Isolated Pruning ===")
    
    # Create a simple config with correct parameters
    config = LayerIsolatedConfig(
        name="test_layer_isolated",
        model_name="simplenet",
        dataset_name="cifar10",
        data_path="./data",
        # Dropout configuration (correct parameter name)
        dropout_rates=[0.1, 0.3, 0.5],
        dropout_mode="scaled",
        pruning_metric="rayleigh_quotient",
        pruning_strategy="low",
        # Training
        train_before_dropout=False,  # Skip initial training for quick test
        training_epochs=1,  # Quick test
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Evaluation
        eval_batches=2  # Use only 2 batches for quick test
    )
    
    # Create model
    model = SimpleNet()
    config.model = model  # Add model to config
    
    print(f"Device: {config.device}")
    print(f"Dropout rates: {config.dropout_rates}")
    print(f"Pruning metric: {config.pruning_metric}")
    print(f"Pruning strategy: {config.pruning_strategy}")
    
    # Create experiment
    try:
        experiment = LayerIsolatedPruningExperiment(config)
        
        # Run experiment
        print("\nRunning experiment...")
        results = experiment.run()
        
        # Print results
        print("\nLayer-isolated pruning results:")
        
        # Print accuracies at different dropout rates
        if 'accuracies' in results:
            print("\nAccuracies at different dropout rates:")
            for strategy in ['low', 'high', 'random']:
                if strategy in results['accuracies']:
                    print(f"\n{strategy.capitalize()} pruning:")
                    for i, rate in enumerate(results['dropout_rates']):
                        acc = results['accuracies'][strategy][i]
                        print(f"  Dropout {rate}: {acc:.2f}%")
        
        # Print layer scores
        if 'layer_scores' in results:
            print("\nLayer scores:")
            for layer_name, scores in results['layer_scores'].items():
                print(f"  {layer_name}: {len(scores)} neurons")
                
    except Exception as e:
        print(f"Layer-isolated pruning test failed: {e}")
        import traceback
        traceback.print_exc()


def test_progressive_dropout():
    """Test progressive dropout functionality."""
    print("\n=== Testing Progressive Dropout ===")
    
    # Create config using base ExperimentConfig
    config = ExperimentConfig(
        name="test_progressive_dropout",
        model_name="simplenet", 
        dataset_name="cifar10",
        data_path="./data",
        # Training
        train_before_dropout=False,  # Skip initial training for test
        training_epochs=1,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Add progressive dropout specific settings
    config.dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    config.num_samples = 100  # Small number for testing
    config.dropout_structure = 'random'
    
    # Create model
    model = SimpleNet()
    config.model = model
    
    print(f"Device: {config.device}")
    print(f"Dropout rates: {config.dropout_rates}")
    
    # Create experiment
    try:
        experiment = ProgressiveDropoutExperiment(config)
        
        # Run experiment
        print("\nRunning experiment...")
        results = experiment.run()
        
        # Print results
        print("\nProgressive dropout results:")
        if 'metrics_by_rate' in results:
            for rate, metrics in results['metrics_by_rate'].items():
                print(f"\nDropout rate {rate}:")
                if isinstance(metrics, dict):
                    for metric_name, layer_results in metrics.items():
                        if not metric_name.startswith('_') and isinstance(layer_results, dict):
                            # Print average metric across layers
                            values = [v for v in layer_results.values() if isinstance(v, (int, float))]
                            if values:
                                avg_value = sum(values) / len(values)
                                print(f"  {metric_name}: avg={avg_value:.4f} (across {len(values)} layers)")
                    
    except Exception as e:
        print(f"Progressive dropout test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("Testing Refactored Alignment Module")
    print("=" * 50)
    
    # Test 1: Model Wrapper
    wrapped_model, x = test_model_wrapper()
    
    # Test 2: Metrics
    test_metrics(wrapped_model, x)
    
    # Test 3: Layer-Isolated Pruning
    test_layer_isolated_pruning()
    
    # Test 4: Progressive Dropout
    test_progressive_dropout()
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main() 