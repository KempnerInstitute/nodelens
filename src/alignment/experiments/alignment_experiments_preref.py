"""
Alignment experiment implementations based on the preref design pattern.

This module contains a streamlined version of the alignment experiment class,
focusing on orchestration rather than implementation details.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import numpy as np

from alignment.config import ExperimentConfig
from alignment.experiments.experiment import Experiment
from alignment.models.registry import create_model
from alignment.metrics import get_metric
from alignment.datasets import load_dataset
from alignment.dropout import progressive_dropout, eigenvector_dropout
from alignment import plotting

logger = logging.getLogger(__name__)


class AlignmentExperimentPreref(Experiment):
    """
    Alignment experiment class following the preref design pattern.
    
    This class orchestrates experiments for neural network alignment studies,
    delegating implementation details to specialized modules.
    """
    
    def __init__(self, config):
        """Initialize the experiment with the given config."""
        super().__init__(config)
        self.setup_paths()
        self.metric = get_metric(config.alignment.metric)
        logger.info(f"Initialized preref-style alignment experiment")
    
    def get_basename(self):
        """Get the base name for the experiment."""
        return f"alignment_{self.config.model.model_name}_{self.config.dataset.dataset_name}"
    
    def prepare_path(self):
        """Prepare the experiment path components."""
        return [
            "alignment_preref",
            self.config.model.model_name,
            self.config.dataset.dataset_name,
            f"metric_{self.config.alignment.metric}"
        ]
    
    def create_networks(self):
        """Create multiple neural networks with different initializations."""
        # Get replicates count
        num_replicates = getattr(self.config.training, "replicates", 5)
        
        # Create networks
        networks = []
        for i in range(num_replicates):
            # Set different seed for each network
            if hasattr(self.config, 'seed') and self.config.seed is not None:
                torch.manual_seed(self.config.seed + i)
                torch.cuda.manual_seed_all(self.config.seed + i)
                np.random.seed(self.config.seed + i)
            
            # Create model
            model = create_model(self.config.model)
            model.to(self.device)
            networks.append(model)
        
        logger.info(f"Created {len(networks)} models with independent initializations")
        return networks
    
    def main(self):
        """Main experiment execution method."""
        # Setup
        self.setup_paths()
        logger.info(f"Set up paths. Results will be saved to {self.results_path}")
        
        # Create networks
        networks = self.create_networks()
        
        # Prepare dataset
        dataset = load_dataset(self.config.dataset)
        
        # Run based on experiment type
        experiment_type = getattr(self.config, 'experiment_type', 'alignment_analysis')
        results = {}
        
        if experiment_type == "progressive_dropout":
            # Run progressive dropout
            results = self._run_progressive_dropout(networks, dataset)
            
        elif experiment_type == "eigenvector_dropout":
            # Run eigenvector dropout
            results = self._run_eigenvector_dropout(networks[0], dataset)
            
        elif experiment_type == "alignment_analysis" or experiment_type == "alignment":
            # Run combined analysis
            results = self._run_alignment_analysis(networks, dataset)
            
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        
        logger.info(f"Completed {experiment_type} experiment")
        return results, networks
    
    def _run_progressive_dropout(self, networks, dataset):
        """Run progressive dropout experiment."""
        logger.info("Running progressive dropout experiment")
        
        # Generate dropout fractions
        dropout_fractions = np.linspace(
            self.config.alignment.dropout_min,
            self.config.alignment.dropout_max,
            self.config.alignment.dropout_steps
        )
        
        # Map old pruning mode names to new ones for backward compatibility
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
        if pruning_mode == "global":
            logger.warning("Pruning mode 'global' is deprecated, use 'global_joint' instead")
            pruning_mode = "global_joint"
        elif pruning_mode == "per_layer_combined":
            logger.warning("Pruning mode 'per_layer_combined' is deprecated, use 'layer_wise' instead")
            pruning_mode = "layer_wise"
        elif pruning_mode == "per_layer_independent":
            logger.warning("Pruning mode 'per_layer_independent' is deprecated, use 'layer_isolated' instead")
            pruning_mode = "layer_isolated"
        
        # Log pruning mode
        logger.info(f"Using pruning mode: {pruning_mode}")
        
        # Run progressive dropout
        results = progressive_dropout(
            networks,
            dataset,
            dropout_fractions=dropout_fractions,
            metric=self.metric,
            device=self.device,
            pruning_mode=pruning_mode,
            dropout_mode=getattr(self.config.extra, "dropout_mode", "global"),
            use_tensorized=True  # Use tensorized approach by default for better performance
        )
        
        # Generate plots
        plot_files = plotting.plot_dropout_results(
            results,
            self.figure_path,
            pruning_mode=pruning_mode,
            dropout_mode=getattr(self.config.extra, "dropout_mode", "global"),
            title_prefix=f"Progressive Dropout ({pruning_mode})"
        )
        
        return {"progressive_dropout": results, "plot_files": plot_files}
    
    def _run_eigenvector_dropout(self, network, dataset):
        """Run eigenvector dropout experiment."""
        logger.info("Running eigenvector dropout experiment")
        
        # Generate dropout fractions
        dropout_fractions = np.linspace(
            self.config.alignment.dropout_min,
            self.config.alignment.dropout_max,
            self.config.alignment.dropout_steps
        )
        
        # Run eigenvector dropout
        results = eigenvector_dropout(
            network,
            dataset,
            dropout_fraction=dropout_fractions,
            metric=self.metric,
            device=self.device,
            dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled"),
            dropout_pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global")
        )
        
        # Generate plots
        plot_files = plotting.plot_dropout_results(
            results,
            self.figure_path,
            pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global"),
            dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled"),
            title_prefix="Eigenvector Dropout"
        )
        
        return {"eigenvector_dropout": results, "plot_files": plot_files}
    
    def _run_alignment_analysis(self, networks, dataset):
        """Run comprehensive alignment analysis."""
        logger.info("Running alignment analysis experiment")
        results = {"config": self.config}
        
        # Run progressive dropout if configured
        if getattr(self.config.alignment, "run_progressive", True):
            prog_results = self._run_progressive_dropout(networks, dataset)
            results["progressive_dropout"] = prog_results["progressive_dropout"]
        
        # Run eigenvector dropout if configured
        if getattr(self.config.alignment, "run_eigenvector", True):
            eig_results = self._run_eigenvector_dropout(networks[0], dataset)
            results["eigenvector_dropout"] = eig_results["eigenvector_dropout"]
        
        # Generate combined summary visualization
        summary_file = plotting.plot_experiment_summary(results, self.figure_path)
        results["summary_file"] = summary_file
        
        return results
    
    def plot(self, results):
        """Generate plots from experiment results."""
        logger.info("Generating experiment visualizations")
        
        # Create experiment summary
        summary_file = plotting.plot_experiment_summary(results, self.figure_path)
        
        # Check for progressive dropout results
        if "progressive_dropout" in results:
            plotting.plot_dropout_results(
                results["progressive_dropout"],
                self.figure_path,
                pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global"),
                dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled"),
                title_prefix="Progressive Dropout"
            )
        
        # Check for eigenvector dropout results
        if "eigenvector_dropout" in results:
            plotting.plot_dropout_results(
                results["eigenvector_dropout"],
                self.figure_path,
                pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global"),
                dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled"),
                title_prefix="Eigenvector Dropout"
            )
        
        logger.info("Completed experiment visualization generation")
    
    def run(self):
        """Run the experiment."""
        # Set random seed if configured
        if hasattr(self.config, 'seed') and self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info(f"Set random seed to {self.config.seed}")
        
        # Run main experiment
        results, networks = self.main()
        
        # Plot results
        self.plot(results)
        
        # Save experiment state
        self.save()
        
        return results, networks


def set_logging_level(level=logging.INFO):
    """Set logging level for all loggers in the alignment package."""
    logging.getLogger().setLevel(level)
    logging.getLogger('alignment').setLevel(level)
    
    # Reduce verbosity for other loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)


def cli_main():
    """Command-line interface for running alignment experiments."""
    parser = argparse.ArgumentParser(description="Neural network alignment experiment (preref style)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = parser.parse_args()

    # Set logging level based on quiet flag
    if args.quiet:
        set_logging_level(logging.WARNING)
    else:
        set_logging_level(logging.INFO)

    # Load configuration
    config = ExperimentConfig.load(args.config)
    
    # Initialize and run experiment
    experiment = AlignmentExperimentPreref(config)
    results, networks = experiment.run()
    
    return results, networks


if __name__ == "__main__":
    cli_main() 