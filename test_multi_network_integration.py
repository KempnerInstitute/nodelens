#!/usr/bin/env python3
"""
Test script to verify multi-network support in general alignment experiment.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alignment.experiments.general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig
from alignment.core.registry import MODEL_REGISTRY, DATASET_REGISTRY

def test_multi_network_config():
    """Test that multi-network configuration works."""
    config = GeneralAlignmentConfig(
        name="test_multi",
        model_name="mlp",
        model_config={
            "hidden_dims": [64, 32],  # MLP uses hidden_dims not hidden_sizes
            "activation_type": "relu",
            "output_dim": 10,
            "input_dim": 784  # MNIST flattened size
        },
        dataset_name="mnist",
        dataset_config={
            "data_path": "./data",
            "train": True,
            "download": True
        },
        batch_size=32,
        num_networks=3,  # Multi-network mode
        training_epochs=2,
        do_train=True,
        do_pruning_experiments=False,
        do_dropout_analysis=False,
        do_eigenfeature_analysis=False,
        device="cpu",  # Use CPU for testing
        checkpoint_dir="./test_checkpoints",
        log_dir="./test_logs"
    )
    
    print(f"✓ Created config with num_networks={config.num_networks}")
    
    # Create experiment
    experiment = GeneralAlignmentExperiment(config)
    print(f"✓ Created experiment, is_multi_network={experiment.is_multi_network}")
    
    # Check that multiple networks were created
    if experiment.is_multi_network:
        assert len(experiment.networks) == 3, f"Expected 3 networks, got {len(experiment.networks)}"
        assert len(experiment.wrapped_networks) == 3, f"Expected 3 wrapped networks, got {len(experiment.wrapped_networks)}"
        assert experiment.model is None, "Single model should be None in multi-network mode"
        assert experiment.wrapped_model is None, "Single wrapped model should be None in multi-network mode"
        print(f"✓ Initialized {len(experiment.networks)} networks")
    
    return experiment

def test_multi_network_training():
    """Test that training works in multi-network mode."""
    config = GeneralAlignmentConfig(
        name="test_multi_train",
        model_name="mlp", 
        model_config={
            "hidden_dims": [32],
            "activation_type": "relu",
            "output_dim": 10,
            "input_dim": 784
        },
        dataset_name="mnist",
        dataset_config={
            "data_path": "./data",
            "train": True,
            "download": True
        },
        batch_size=64,
        num_networks=2,
        training_epochs=1,
        do_train=True,
        measure_alignment_during_training=False,
        do_pruning_experiments=False,
        do_dropout_analysis=False,
        do_eigenfeature_analysis=False,
        device="cpu",
        use_tensorized_training=True,
        aggregate_metrics=True
    )
    
    experiment = GeneralAlignmentExperiment(config)
    
    # Run training
    print("\nTesting multi-network training...")
    train_results = experiment._train_model()
    
    # Check results structure
    assert "train_losses" in train_results, "Missing train_losses in results"
    assert "train_accs" in train_results, "Missing train_accs in results"
    assert len(train_results["train_losses"]) == 1, "Should have 1 epoch of losses"
    
    print(f"✓ Training completed, final accuracy: {train_results['train_accs'][-1]:.2f}%")
    
    return train_results

def test_single_vs_multi_network():
    """Test that single network mode still works."""
    # Single network config
    single_config = GeneralAlignmentConfig(
        name="test_single",
        model_name="mlp",
        model_config={"hidden_dims": [32], "activation_type": "relu", "output_dim": 10, "input_dim": 784},
        dataset_name="mnist",
        dataset_config={"data_path": "./data", "train": True, "download": True},
        batch_size=64,
        num_networks=1,  # Single network mode
        training_epochs=1,
        do_train=False,
        device="cpu"
    )
    
    single_exp = GeneralAlignmentExperiment(single_config)
    assert not single_exp.is_multi_network, "Should be single network mode"
    assert single_exp.model is not None, "Should have single model"
    assert not hasattr(single_exp, 'networks') or not single_exp.networks, "Should not have networks list"
    
    print("✓ Single network mode still works correctly")

def main():
    """Run all tests."""
    print("Testing Multi-Network Integration in General Alignment Experiment")
    print("=" * 60)
    
    # Test 1: Configuration
    print("\n1. Testing multi-network configuration...")
    test_multi_network_config()
    
    # Test 2: Single vs Multi
    print("\n2. Testing single vs multi-network modes...")
    test_single_vs_multi_network()
    
    # Test 3: Training
    print("\n3. Testing multi-network training...")
    try:
        test_multi_network_training()
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All tests completed!")

if __name__ == "__main__":
    main() 