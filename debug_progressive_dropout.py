#!/usr/bin/env python

"""
Debug script for testing the progressive_dropout function directly.
"""

import os
import sys
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Setup basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import alignment modules
from alignment.config import ExperimentConfig
from alignment.metrics import get_metric
from alignment.models.registry import create_model
from alignment.datasets import load_dataset
from alignment.dropout import progressive_dropout

def main():
    """Run a direct test of progressive_dropout with verbose output."""
    logger.info("Starting progressive_dropout debug script")
    
    # Load configuration
    config_path = "configs/config_alignment_experiment.yaml"
    config = ExperimentConfig.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create metric
    metric_name = config.alignment.metric
    logger.info(f"Using alignment metric: {metric_name}")
    metric = get_metric(metric_name)
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset.dataset_name}")
    dataset = load_dataset(config.dataset)
    
    # Create networks (just 2 for testing)
    num_networks = 2
    logger.info(f"Creating {num_networks} test networks")
    networks = []
    for i in range(num_networks):
        network = create_model(config.model)
        network.to(device)
        networks.append(network)
        logger.info(f"Created network {i+1}/{num_networks}: {type(network).__name__}")
    
    # Set up dropout parameters
    logger.info("Setting up dropout parameters")
    dropout_min = config.alignment.dropout_min
    dropout_max = config.alignment.dropout_max
    num_steps = 5  # Use fewer steps for debugging
    dropout_fractions = np.linspace(dropout_min, dropout_max, num_steps).tolist()
    logger.info(f"Testing dropout fractions: {dropout_fractions}")
    
    # Get pruning and dropout modes
    pruning_mode = config.extra.dropout_pruning_mode
    dropout_mode = config.extra.dropout_mode
    logger.info(f"Pruning mode: {pruning_mode}, Dropout mode: {dropout_mode}")
    
    # Run progressive dropout with detailed logging
    logger.info("Starting progressive_dropout call")
    try:
        results = progressive_dropout(
            networks,
            dataset,
            dropout_fractions,
            metric,
            device=device,
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode,
        )
        
        # Print results
        logger.info("Progressive dropout completed successfully")
        logger.info(f"Results keys: {results.keys() if hasattr(results, 'keys') else type(results)}")
        
        # Print detailed structure
        if hasattr(results, 'network_accuracies'):
            logger.info(f"network_accuracies keys: {results.network_accuracies.keys()}")
            for key in results.network_accuracies:
                if results.network_accuracies[key]:
                    logger.info(f"Strategy {key} type: {type(results.network_accuracies[key])}")
                    logger.info(f"Strategy {key} shape: {np.array(results.network_accuracies[key]).shape}")
                    logger.info(f"Strategy {key} values: {results.network_accuracies[key]}")
                else:
                    logger.info(f"Strategy {key} accuracies are empty")
        else:
            logger.warning("No network_accuracies in results")
            
        if hasattr(results, 'alignment_values'):
            logger.info(f"alignment_values keys: {results.alignment_values.keys()}")
            for key in results.alignment_values:
                if results.alignment_values[key]:
                    logger.info(f"Alignment {key} type: {type(results.alignment_values[key])}")
                    logger.info(f"Alignment {key} shape: {np.array(results.alignment_values[key]).shape if isinstance(results.alignment_values[key], (list, np.ndarray)) else 'not array'}")
                else:
                    logger.info(f"Alignment {key} values are empty")
        else:
            logger.warning("No alignment_values in results")
        
        # Create a results directory
        os.makedirs("debug_output", exist_ok=True)
        
        # Save result details
        import json
        with open("debug_output/progressive_dropout_debug.json", "w") as f:
            # Convert results to dict if it's not already
            if not isinstance(results, dict):
                # Map numeric indices to strategy names if needed
                strategy_map = {0: "high_rq", 1: "low_rq", 2: "random"}
                
                # Extract network accuracies with proper keys
                accuracies = {}
                if hasattr(results, "network_accuracies"):
                    for idx, strategy in strategy_map.items():
                        if idx in results.network_accuracies:
                            accuracies[strategy] = results.network_accuracies[idx]
                        else:
                            accuracies[strategy] = []
                
                # Extract alignment values with proper keys
                alignment_values = {}
                if hasattr(results, "alignment_values"):
                    for idx, strategy in strategy_map.items():
                        if idx in results.alignment_values:
                            alignment_values[strategy] = results.alignment_values[idx]
                        else:
                            alignment_values[strategy] = []
                
                results_dict = {
                    "dropout_fractions": dropout_fractions,
                    "accuracies": accuracies,
                    "alignment_values": alignment_values,
                }
                json.dump(results_dict, f, indent=2)
            else:
                json.dump(results, f, indent=2)
        
        logger.info("Saved debug results to debug_output/progressive_dropout_debug.json")
        
    except Exception as e:
        logger.error(f"Error in progressive_dropout: {str(e)}", exc_info=True)
    
    logger.info("Debug script completed")

if __name__ == "__main__":
    main() 