"""
Benchmark script for comparing different dropout implementation approaches.

This script compares the performance of three approaches to progressive dropout:
1. Original (sequential): Processing networks one-by-one
2. Batched: Processing networks in small batches
3. Tensorized: Using tensor operations to process networks at once
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from alignment_refac1.config import ExperimentConfig
from alignment_refac1.models.registry import create_model
from alignment_refac1.metrics import get_metric
from alignment_refac1.datasets import load_dataset
from alignment_refac1.dropout import progressive_dropout
from alignment_refac1 import utils

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def create_networks(config, num_networks=20):
    """Create multiple neural networks for benchmarking."""
    networks = []
    device = config.device
    
    for i in range(num_networks):
        # Set different seed for each network
        if hasattr(config, 'seed') and config.seed is not None:
            torch.manual_seed(config.seed + i)
            torch.cuda.manual_seed_all(config.seed + i)
            np.random.seed(config.seed + i)
        
        # Create model
        model = create_model(config.model)
        model.to(device)
        networks.append(model)
    
    logger.info(f"Created {len(networks)} models with independent initializations")
    return networks

def run_benchmark(config_path, approaches=["original", "batched", "tensorized"], 
                 num_networks=20, num_runs=3, num_batches=5, detailed_timing=True):
    """
    Run benchmarks for different dropout implementation approaches.
    
    Args:
        config_path: Path to configuration file
        approaches: List of approaches to benchmark
        num_networks: Number of networks to use
        num_runs: Number of times to run each benchmark for averaging
        num_batches: Number of data batches to process
        detailed_timing: Whether to time individual steps
    """
    # Load configuration
    config = ExperimentConfig.load(config_path)
    
    # Create networks
    networks = create_networks(config, num_networks)
    
    # Prepare dataset
    dataset = load_dataset(config.dataset)
    
    # Get metric
    metric = get_metric(config.alignment.metric)
    
    # Benchmark results
    results = {}
    detailed_results = defaultdict(lambda: defaultdict(list))
    
    # Run benchmarks
    for approach in approaches:
        logger.info(f"Benchmarking {approach} approach")
        
        # Set approach-specific parameters
        if approach == "original":
            # Original sequential approach
            kwargs = {"network_batch_size": 1, "use_tensorized": False}
        elif approach == "batched":
            # Batched approach
            kwargs = {"network_batch_size": 4, "use_tensorized": False}
        elif approach == "tensorized":
            # Tensorized approach
            kwargs = {"use_tensorized": True}
        else:
            logger.error(f"Unknown approach: {approach}")
            continue
        
        # Run multiple times and average
        timings = []
        for run in range(num_runs):
            logger.info(f"Run {run+1}/{num_runs}")
            
            # Time the execution of full dropout
            start_time = time.time()
            
            # Run progressive dropout with the specified approach
            dropout_results = progressive_dropout(
                networks,
                dataset,
                dropout_fractions=np.linspace(0.1, 0.9, config.alignment.dropout_steps),
                metric=metric,
                device=config.device,
                pruning_mode=config.extra.dropout_pruning_mode,
                dropout_mode=config.extra.dropout_mode,
                num_batches=num_batches,  
                detailed_timing=detailed_timing,
                **kwargs
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            timings.append(elapsed_time)
            
            # Collect detailed timing information if available
            if detailed_timing and hasattr(dropout_results, "timing_info"):
                for key, value in dropout_results.timing_info.items():
                    detailed_results[approach][key].append(value)
                
                # Log detailed timing information
                for key, values in dropout_results.timing_info.items():
                    if isinstance(values, (int, float)):
                        logger.info(f"  {key}: {values:.2f} seconds")
            
            logger.info(f"Completed run in {elapsed_time:.2f} seconds")
        
        # Calculate average and standard deviation
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        
        # Store results
        results[approach] = {
            "average_time": avg_time,
            "std_time": std_time,
            "timings": timings
        }
        
        logger.info(f"{approach}: {avg_time:.2f} ± {std_time:.2f} seconds")
    
    # Print comparison
    logger.info("\nBenchmarking Results:")
    logger.info("-" * 50)
    logger.info(f"{'Approach':<15} {'Average Time (s)':<20} {'Speedup':<10}")
    logger.info("-" * 50)
    
    # Calculate speedup relative to original approach
    original_time = results["original"]["average_time"] if "original" in results else None
    
    for approach in approaches:
        if approach not in results:
            continue
            
        avg_time = results[approach]["average_time"]
        speedup = original_time / avg_time if original_time else 1.0
        
        logger.info(f"{approach:<15} {avg_time:.2f} ± {results[approach]['std_time']:.2f} {'':>5} {speedup:.2f}x")
    
    logger.info("-" * 50)
    
    # Print detailed timing if available
    if detailed_timing:
        logger.info("\nDetailed Timing Breakdown:")
        logger.info("-" * 80)
        
        # Find all unique timing keys
        all_keys = set()
        for approach in approaches:
            if approach in detailed_results:
                all_keys.update(detailed_results[approach].keys())
        
        # Print header
        logger.info(f"{'Operation':<25} " + " ".join([f"{approach:<15}" for approach in approaches]))
        logger.info("-" * 80)
        
        # Print timings for each key
        for key in sorted(all_keys):
            line = f"{key:<25} "
            for approach in approaches:
                if approach in detailed_results and key in detailed_results[approach]:
                    values = detailed_results[approach][key]
                    if len(values) > 0:
                        avg = np.mean(values)
                        line += f"{avg:.2f}s{' ' * 10}"
                    else:
                        line += f"{'N/A':<15}"
                else:
                    line += f"{'N/A':<15}"
            logger.info(line)
        
        logger.info("-" * 80)
    
    return results, detailed_results if detailed_timing else None

def main():
    """Main function for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark dropout implementation approaches")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--approaches", type=str, nargs="+", default=["original", "batched", "tensorized"],
                      help="Approaches to benchmark")
    parser.add_argument("--num-networks", type=int, default=20, help="Number of networks to use")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for each benchmark")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of data batches to process")
    parser.add_argument("--no-detailed-timing", action="store_true", help="Disable detailed timing")
    
    args = parser.parse_args()
    
    # Run benchmarks
    run_benchmark(
        args.config, 
        args.approaches, 
        args.num_networks, 
        args.num_runs,
        args.num_batches,
        not args.no_detailed_timing
    )

if __name__ == "__main__":
    main() 