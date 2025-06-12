#!/usr/bin/env python
"""
Quick test script to evaluate AlexNet accuracy on CIFAR-10
"""

import sys
import torch
import torch.nn as nn
from alignment.config import ExperimentConfig
from alignment.models.registry import create_model
from alignment.datasets import load_dataset

def test_alexnet_accuracy():
    """Test baseline accuracy of pretrained AlexNet on ImageNet"""
    
    # Load config
    config = ExperimentConfig.load("configs/config_alignment_experiment.yaml")
    
    # Create model
    print("Creating AlexNet model...")
    model = create_model(config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load dataset with correct transform parameters
    print("Loading ImageNet dataset...")
    from alignment.models.models import get_transform_parameters
    
    # Get correct transform parameters for AlexNet + CIFAR-10
    model_name = config.model.model_name.lower()
    if model_name.startswith("torchvision_"):
        model_name = model_name.replace("torchvision_", "")
    
    transform_params = get_transform_parameters(model_name, config.dataset.dataset_name)
    print(f"Using transform parameters: {transform_params}")
    
    dataset = load_dataset(
        config.dataset,
        batch_size=config.dataset.batch_size,
        device=device,
        transform_params=transform_params
    )
    
    # Test on a smaller subset first (fast check)
    print("\n=== Quick Test (5 batches) ===")
    accuracy_quick = dataset.evaluate(model, device=device, num_batches=5, show_progress=True)
    print(f"Quick test accuracy: {accuracy_quick[0]:.2f}%")
    
    # Full test set evaluation
    print("\n=== Full Test Set Evaluation ===")
    accuracy_full, loss_full = dataset.evaluate(model, device=device, num_batches=None, show_progress=True)
    print(f"Full test accuracy: {accuracy_full:.2f}%")
    print(f"Average loss: {loss_full:.4f}")
    
    # Print some dataset info
    print(f"\nDataset info:")
    print(f"- Dataset: {config.dataset.dataset_name}")
    print(f"- Train samples: {len(dataset.train_dataset)}")
    print(f"- Test samples: {len(dataset.test_dataset)}")
    print(f"- Classes: {dataset.num_classes}")
    
    # Sample a batch to check data shape
    sample_batch = next(iter(dataset.test_loader))
    inputs, targets = dataset.unwrap_batch(sample_batch, device)
    print(f"- Sample batch shape: {inputs.shape}")
    print(f"- Sample targets shape: {targets.shape}")
    
    return accuracy_full, loss_full

if __name__ == "__main__":
    accuracy, loss = test_alexnet_accuracy()
    print(f"\n🎯 Final Result: {accuracy:.2f}% accuracy on ImageNet") 