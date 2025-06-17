"""
Example: Using Standard Models in the Codebase

This example shows how to use the standard models (MLP, CNN2P2) that match
the functionality from the old alignment codebase.
"""

import torch
from torchvision import datasets, transforms

# Import standard models
from alignment_refactor.models.architectures.standard_models import (
    MLP, CNN2P2, create_model, DATASET_PARAMETERS
)
from alignment_refactor.models import ModelWrapper
from alignment_refactor.metrics import RayleighQuotient


def example_mlp_mnist():
    """Example: Creating and using an MLP for MNIST."""
    print("=" * 60)
    print("Example 1: MLP on MNIST")
    print("=" * 60)
    
    # Method 1: Direct instantiation
    model = MLP(
        input_dim=784,
        hidden_dims=[300, 200, 100],
        output_dim=10,
        dropout_rate=0.5,
        activation_type="relu"
    )
    
    print(f"Created MLP with architecture:")
    print(f"  Input: 784")
    print(f"  Hidden: [300, 200, 100]")
    print(f"  Output: 10")
    
    # Method 2: Using create_model with dataset parameters
    model2 = create_model('mlp', 'mnist', hidden_dims=[300, 200])
    print(f"\nCreated MLP using create_model function")
    
    # Test forward pass
    dummy_input = torch.randn(32, 784)
    output = model(dummy_input)
    print(f"\nForward pass successful: input {dummy_input.shape} -> output {output.shape}")


def example_cnn2p2_cifar():
    """Example: Creating and using CNN2P2 for CIFAR-10."""
    print("\n" + "=" * 60)
    print("Example 2: CNN2P2 on CIFAR-10")
    print("=" * 60)
    
    # Create CNN2P2 with CIFAR-10 parameters
    model = create_model(
        'cnn2p2',
        'cifar10',
        conv_channels=[32, 64],
        hidden_fc_dim=256,
        dropout_rate=0.3
    )
    
    print(f"Created CNN2P2 for CIFAR-10")
    print(f"  Input channels: 3")
    print(f"  Conv channels: [32, 64]")
    print(f"  FC hidden: 256")
    print(f"  Output: 10 classes")
    
    # Test forward pass
    dummy_input = torch.randn(16, 3, 32, 32)
    output = model(dummy_input)
    print(f"\nForward pass successful: input {dummy_input.shape} -> output {output.shape}")


def example_with_model_wrapper():
    """Example: Using ModelWrapper for activation tracking."""
    print("\n" + "=" * 60)
    print("Example 3: Using ModelWrapper with Standard Models")
    print("=" * 60)
    
    # Create a simple MLP
    base_model = MLP(
        input_dim=784,
        hidden_dims=[128, 64],
        output_dim=10,
        dropout_rate=0.2
    )
    
    # Wrap it for activation tracking
    # First, identify layer names
    print("\nModel structure:")
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  Linear layer: {name}")
    
    # Track specific layers
    tracked_layers = ['network.0', 'network.3', 'network.6']  # The Linear layers
    wrapped_model = ModelWrapper(base_model, tracked_layers=tracked_layers)
    
    print(f"\nWrapped model tracking layers: {wrapped_model.tracked_layers}")
    
    # Forward pass with activation tracking
    dummy_input = torch.randn(8, 784)
    output, activations = wrapped_model.forward_with_activations(dummy_input)
    
    print("\nActivations captured:")
    for key, tensor in activations.items():
        print(f"  {key}: shape {tensor.shape}")
    
    # Compute metrics
    metric = RayleighQuotient()
    weights = wrapped_model.get_layer_weights()
    
    print("\nComputing Rayleigh Quotient for each layer:")
    for layer_name in tracked_layers:
        if f"{layer_name}_input" in activations and layer_name in weights:
            inputs = activations[f"{layer_name}_input"].flatten(start_dim=1)
            layer_weights = weights[layer_name]
            
            rq_scores = metric.compute(inputs=inputs, weights=layer_weights)
            print(f"  {layer_name}: mean RQ = {rq_scores.mean().item():.4f}")


def example_dataset_parameters():
    """Example: Viewing available dataset parameters."""
    print("\n" + "=" * 60)
    print("Example 4: Dataset-Specific Parameters")
    print("=" * 60)
    
    print("\nAvailable dataset parameters:")
    print("\nFor MLP:")
    for dataset, params in DATASET_PARAMETERS['mlp'].items():
        print(f"  {dataset}: {params}")
    
    print("\nFor CNN2P2:")
    for dataset, params in DATASET_PARAMETERS['cnn2p2'].items():
        print(f"  {dataset}: {params}")


def main():
    """Run all examples."""
    print("Standard Models Usage Examples")
    print("==============================\n")
    
    # Set up for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    example_mlp_mnist()
    example_cnn2p2_cifar()
    example_with_model_wrapper()
    example_dataset_parameters()
    
    print("\n" + "=" * 60)
    print("Migration Tip:")
    print("=" * 60)
    print("These models match the old codebase functionality.")
    print("You can use them as drop-in replacements when migrating.")
    print("\nKey differences from old codebase:")
    print("- No AlignmentNetwork wrapper (use ModelWrapper instead)")
    print("- No model registry (use create_model function)")
    print("- Layer naming might differ slightly")


if __name__ == "__main__":
    main() 