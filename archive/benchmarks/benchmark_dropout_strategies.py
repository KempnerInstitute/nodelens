#!/usr/bin/env python
"""
Benchmark script to compare original sequential dropout approach vs. multi-strategy approach.

This script measures the performance difference between:
1. Processing strategies (high_rq, low_rq, random) sequentially
2. Processing all strategies simultaneously with the new tensorized multi-strategy approach
"""

import os
import sys
import time
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm

from alignment.experiments.alignment_experiments import AlignmentExperiment
from alignment.config import ExperimentConfig
from alignment.datasets import load_dataset
from alignment.dropout import progressive_dropout, progressive_dropout_multi_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_benchmark(config_path, num_runs=3):
    """Run benchmark comparing sequential vs. multi-strategy dropout."""
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = ExperimentConfig.load(config_path)
    
    # Force the device to be cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    
    # Create experiment
    experiment = AlignmentExperiment(config)
    
    # Create networks
    logger.info("Creating networks...")
    networks = experiment.create_networks()
    logger.info(f"Created {len(networks)} networks")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(config.dataset)
    
    # Setup dropout parameters
    dropout_fractions = np.linspace(
        config.alignment.dropout_min, 
        config.alignment.dropout_max, 
        config.alignment.dropout_steps
    ).tolist()
    
    pruning_mode = getattr(config.extra, "dropout_pruning_mode", "global_joint")
    dropout_mode = getattr(config.extra, "dropout_mode", "scaled")
    metric = experiment.metric
    
    # Initialize timing results
    sequential_times = []
    multi_strategy_times = []
    
    # Run benchmarks multiple times
    for run in range(num_runs):
        logger.info(f"\nRun {run+1}/{num_runs}")
        
        # Make copies of networks to ensure fair comparison
        networks_sequential = [net.clone() if hasattr(net, 'clone') else net for net in networks]
        networks_multi = [net.clone() if hasattr(net, 'clone') else net for net in networks]
        
        # Process strategies sequentially (original approach)
        logger.info("Running sequential approach...")
        sequential_start = time.time()
        
        strategies = ["high_rq", "low_rq", "random"]
        for strategy in tqdm(strategies, desc="Strategies"):
            # Clone networks for this strategy to avoid interference
            strategy_networks = [net.clone() if hasattr(net, 'clone') else net for net in networks_sequential]
            
            # Run progressive dropout with this strategy
            network_accuracies, network_losses = progressive_dropout(
                strategy_networks,
                dataset,
                dropout_fractions,
                metric,
                device,
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode,
                strategy=strategy,
                show_progress=False  # Disable progress bars for cleaner output
            )
        
        sequential_time = time.time() - sequential_start
        sequential_times.append(sequential_time)
        logger.info(f"Sequential approach took {sequential_time:.2f} seconds")
        
        # Process all strategies at once (new approach)
        logger.info("Running multi-strategy approach...")
        multi_start = time.time()
        
        # Run with multi-strategy mode
        network_accuracies, network_losses = progressive_dropout(
            networks_multi,
            dataset,
            dropout_fractions,
            metric,
            device,
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode,
            show_progress=False,  # Disable progress bars for cleaner output
            use_multi_strategy=True
        )
        
        multi_time = time.time() - multi_start
        multi_strategy_times.append(multi_time)
        logger.info(f"Multi-strategy approach took {multi_time:.2f} seconds")
        
        # Calculate speedup
        speedup = sequential_time / multi_time if multi_time > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x")
    
    # Calculate average times and speedup
    avg_sequential = np.mean(sequential_times)
    avg_multi = np.mean(multi_strategy_times)
    avg_speedup = avg_sequential / avg_multi if avg_multi > 0 else float('inf')
    
    std_sequential = np.std(sequential_times)
    std_multi = np.std(multi_strategy_times)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*50)
    logger.info(f"Network count: {len(networks)}")
    logger.info(f"Dropout steps: {len(dropout_fractions)}")
    logger.info(f"Pruning mode: {pruning_mode}")
    logger.info(f"Dropout mode: {dropout_mode}")
    logger.info(f"Device: {device}")
    logger.info("-"*50)
    logger.info(f"Sequential approach: {avg_sequential:.2f} ± {std_sequential:.2f} seconds")
    logger.info(f"Multi-strategy approach: {avg_multi:.2f} ± {std_multi:.2f} seconds")
    logger.info(f"Average speedup: {avg_speedup:.2f}x")
    logger.info("="*50)
    
    return {
        "sequential": sequential_times,
        "multi_strategy": multi_strategy_times,
        "speedup": avg_speedup
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark dropout strategies")
    parser.add_argument("--config", type=str, default="configs/config_alignment_experiment.yaml", 
                      help="Path to config file")
    parser.add_argument("--runs", type=int, default=1, 
                      help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Run the benchmark
    results = run_benchmark(args.config, args.runs)
    
    # Print the recommendation
    if results["speedup"] > 1.5:
        print("\nRECOMMENDATION: Use multi-strategy approach for significant speedup.")
    else:
        print("\nRECOMMENDATION: Both approaches have similar performance in this configuration.")

if __name__ == "__main__":
    main() 