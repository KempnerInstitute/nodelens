#!/usr/bin/env python
"""
Test script for running alignment experiments with cascading layer pruning
"""

import os
import sys
import logging
import time

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

try:
    from alignment.config import ExperimentConfig
    from alignment.experiments.alignment_experiments import AlignmentExperiment
    from alignment.experiments.alignment_experiments import set_logging_level
    print("Successfully imported alignment modules")
except ImportError as e:
    print(f"Error importing alignment modules: {e}")
    sys.exit(1)

def main():
    # Define experiment parameters
    config_file = "configs/config_alignment_experiment.yaml"
    experiment_type = "progressive_dropout"
    pruning_mode = "cascading_layer"
    run_name = "cascading_test"
    
    print(f"Starting alignment experiment with cascading layer pruning at {time.ctime()}")
    start_time = time.time()
    
    # Set up logging
    set_logging_level(logging.INFO)
    
    # Load configuration
    config_path = os.path.join(script_dir, config_file)
    config = ExperimentConfig.load(config_path)
    
    # Set experiment parameters
    config.experiment_type = experiment_type
    
    # Make sure the extra attribute exists
    if not hasattr(config, 'extra'):
        config.extra = type('ExtraConfig', (), {})()
    
    config.extra.dropout_pruning_mode = pruning_mode
    
    # Set experiment name
    config.experiment_name = run_name
    
    # Modify configuration for faster testing
    config.training.replicates = 3  # Use fewer networks
    config.alignment.dropout_steps = 3  # Use fewer dropout steps
    
    # Print configuration for debugging
    print(f"Using pruning mode: {pruning_mode}")
    print(f"Number of replicates: {config.training.replicates}")
    print(f"Number of dropout steps: {config.alignment.dropout_steps}")
    
    # Create and run experiment
    experiment = AlignmentExperiment(config)
    results, networks = experiment.run()
    
    end_time = time.time()
    print(f"Alignment experiment finished at {time.ctime()}")
    print(f"Total duration: {end_time - start_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 