#!/usr/bin/env python3
"""Test script to verify the experiments module functionality."""

import torch
from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig

def test_experiments_module():
    """Test that the experiments module can be imported and used."""
    print("Testing experiments module...")
    
    # Create a simple config
    config = GeneralAlignmentConfig(
        name="test_experiment",
        dataset_name="mnist",
        model_name="mlp",
        alignment_metrics=["rayleigh_quotient"],
        pruning_strategy="magnitude",
        pruning_config={"amount": 0.5},
        training_config={
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001
        },
        train_model=False,  # Skip training for quick test
        compute_initial_metrics=True,
        apply_pruning=False,
        fine_tune_after_pruning=False
    )
    
    print(f"Created config: {config.name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Model: {config.model_name}")
    print(f"Metrics: {config.alignment_metrics}")
    
    # Create experiment
    experiment = GeneralAlignmentExperiment(config)
    print("✓ Successfully created GeneralAlignmentExperiment")
    
    # Test that we can access the model
    model = experiment.get_model()
    print(f"✓ Model created: {type(model).__name__}")
    
    # Test wrapped model
    wrapped = experiment.wrapped_model
    print(f"✓ Wrapped model created: {type(wrapped).__name__}")
    
    print("\nExperiments module is working correctly!")
    return True

if __name__ == "__main__":
    try:
        test_experiments_module()
    except Exception as e:
        print(f"Error testing experiments module: {e}")
        import traceback
        traceback.print_exc() 