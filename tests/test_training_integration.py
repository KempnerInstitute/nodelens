#!/usr/bin/env python
"""
Test script to verify the integration of tensorized training methods in AlignmentExperiment.
"""

import os
import sys
import logging
import time
import torch
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.config import ExperimentConfig
from alignment.experiments.alignment_experiments import AlignmentExperiment
from alignment.datasets import load_dataset
from alignment.models.registry import create_model

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_methods():
    """
    Test all training methods to ensure they are properly integrated.
    """
    # Create base config
    config_dict = {
        "experiment_type": "alignment",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": {
            "model_name": "MLP",
            "dropout_rate": 0.0,
        },
        "dataset": {
            "dataset_name": "MNIST",
            "batch_size": 64,
        },
        "training": {
            "epochs": 1,
            "replicates": 5,
            "learning_rate": 0.001,
        },
        "alignment": {
            "metric": "RQ",
            "run_progressive": False,
            "run_eigenvector": False,
        },
        "extra": {},
    }

    # Create dataset
    dataset_config = {
        "dataset_name": "MNIST",
        "batch_size": 64,
        "data_path": "./data",
        "transform_params": {
            "flatten": True,
        }
    }
    dataset = load_dataset(dataset_config)

    # Test each training method
    training_methods = ["auto", "sequential", "tensorized", "fully_tensorized"]
    
    for method in training_methods:
        logger.info(f"\n==== Testing {method} training method ====")
        
        # Update config with current method
        test_config = config_dict.copy()
        test_config["extra"] = {"training_method": method}
        
        # Create experiment
        config = ExperimentConfig.from_dict(test_config)
        experiment = AlignmentExperiment(config)
        
        # Create networks
        networks = experiment.create_networks()
        
        # Measure training time
        start_time = time.time()
        
        # Train networks
        result = experiment.train_networks(networks, dataset)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Check results
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        logger.info(f"Final training accuracy: {result['train_acc'][-1]:.2f}%")
        logger.info(f"Final test accuracy: {result['test_acc'][-1]:.2f}%")
        
        # Verify result has expected keys
        assert "train_loss" in result
        assert "train_acc" in result
        assert "test_loss" in result
        assert "test_acc" in result
        
        logger.info(f"✅ {method} training method works correctly")

if __name__ == "__main__":
    logger.info("Testing tensorized training integration")
    test_training_methods()
    logger.info("All tests passed!") 