"""
Alignment experiment implementations.

This module contains experiment classes for neural network alignment studies,
focusing on alignment metrics, dropout impacts, and training analysis.
"""

import logging
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from alignment.config import ExperimentConfig
from alignment.experiments.experiment import Experiment
from alignment.metrics import AlignmentMetric, get_metric
from alignment.models.registry import create_model
from alignment.dropout import progressive_dropout, eigenvector_dropout
from alignment.training import train_model, evaluate_model
from alignment.utils import setup_logging

logger = logging.getLogger(__name__)


class AlignmentExperiment(Experiment):
    """
    Experiment class for studying neural network alignment properties.
    
    This class implements experiments that assess alignment between layers
    in neural networks, with support for different dropout strategies,
    multiple metrics, and visualization.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the alignment experiment.
        
        Args:
            config: Experiment configuration object
        """
        super().__init__(config)
        self.metric = get_metric(config.alignment.metric)
        
    def get_basename(self) -> str:
        """
        Get the base name for the experiment.
        
        Returns:
            Base name string
        """
        return f"alignment_{self.config.model.model_name}_{self.config.dataset.dataset_name}"
    
    def prepare_path(self) -> List[str]:
        """
        Prepare the experiment path components.
        
        Returns:
            List of path components
        """
        return [
            "alignment",
            self.config.model.model_name,
            self.config.dataset.dataset_name,
            f"metric_{self.config.alignment.metric}"
        ]
    
    def create_networks(self) -> List[nn.Module]:
        """
        Create the neural networks for the experiment.
        
        Returns:
            List containing the alignment network
        """
        # Create model based on configuration
        model = create_model(self.config.model)
        model.to(self.device)
        logger.info(f"Created model: {self.config.model.model_name}")
        
        return [model]
    
    def main(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Main experiment execution method.
        
        Returns:
            Tuple of (results dictionary, list of networks)
        """
        # Create networks
        networks = self.create_networks()
        net = networks[0]  # Main network
        
        # Training phase if enabled
        if self.config.training.epochs > 0:
            logger.info(f"Training for {self.config.training.epochs} epochs")
            train_model(
                net, 
                self.config.dataset, 
                self.config.training,
                device=self.device,
                checkpoint_path=self.checkpoint_path,
                extra_config=self.config.extra
            )
            
            # Evaluation to verify training results
            eval_results = evaluate_model(
                net,
                self.config.dataset,
                device=self.device,
                extra_config=self.config.extra
            )
            logger.info(f"Evaluation results: {eval_results}")
        
        # Run alignment analysis
        logger.info(f"Running alignment analysis with {self.config.alignment.metric}")
        
        # Prepare the results structure
        results = {
            "progressive_dropout": {},
            "eigenvector_dropout": {},
            "config": self.config,
        }
        
        # Progressive dropout experiment
        if self.config.alignment.run_progressive:
            logger.info("Running progressive dropout experiment")
            prog_results = self._run_progressive_dropout(net)
            results["progressive_dropout"] = prog_results
        
        # Eigenvector dropout experiment
        if self.config.alignment.run_eigenvector:
            logger.info("Running eigenvector dropout experiment")
            eig_results = self._run_eigenvector_dropout(net)
            results["eigenvector_dropout"] = eig_results
        
        return results, networks
    
    def _run_progressive_dropout(self, net: nn.Module) -> Dict:
        """
        Run progressive dropout experiments.
        
        Args:
            net: Neural network to analyze
            
        Returns:
            Dictionary of progressive dropout results
        """
        dropout_fractions = np.linspace(
            self.config.alignment.dropout_min,
            self.config.alignment.dropout_max,
            self.config.alignment.dropout_steps
        )
        
        results = {}
        
        # Use tqdm for a nice progress bar
        for dropout_fraction in tqdm(dropout_fractions, desc="Progressive Dropout"):
            # Run without verbose logging for each fraction
            accuracy, alignment_values = progressive_dropout(
                net,
                self.config.dataset,
                dropout_fraction=dropout_fraction,
                metric=self.metric,
                device=self.device,
                dropout_mode=self.config.extra.dropout_mode
            )
            
            # Store results
            results[float(dropout_fraction)] = {
                "accuracy": accuracy,
                "alignment": alignment_values
            }
            
        return results
    
    def _run_eigenvector_dropout(self, net: nn.Module) -> Dict:
        """
        Run eigenvector dropout experiments.
        
        Args:
            net: Neural network to analyze
            
        Returns:
            Dictionary of eigenvector dropout results
        """
        dropout_fractions = np.linspace(
            self.config.alignment.dropout_min,
            self.config.alignment.dropout_max,
            self.config.alignment.dropout_steps
        )
        
        results = {}
        
        # Use tqdm for a nice progress bar
        for dropout_fraction in tqdm(dropout_fractions, desc="Eigenvector Dropout"):
            # Run without verbose logging for each fraction
            accuracy, alignment_values = eigenvector_dropout(
                net,
                self.config.dataset,
                dropout_fraction=dropout_fraction,
                metric=self.metric,
                device=self.device,
                dropout_mode=self.config.extra.dropout_mode
            )
            
            # Store results
            results[float(dropout_fraction)] = {
                "accuracy": accuracy,
                "alignment": alignment_values
            }
            
        return results
    
    def plot(self, results: Dict) -> None:
        """
        Plot experiment results.
        
        Args:
            results: Dictionary containing experiment results
        """
        # Check if any results exist to plot
        if not results.get("progressive_dropout") and not results.get("eigenvector_dropout"):
            logger.warning("No results to plot")
            return
        
        if results.get("progressive_dropout"):
            self._plot_dropout_results(
                results["progressive_dropout"], 
                "Progressive Dropout",
                "progressive_dropout.png"
            )
            
        if results.get("eigenvector_dropout"):
            self._plot_dropout_results(
                results["eigenvector_dropout"], 
                "Eigenvector Dropout",
                "eigenvector_dropout.png"
            )
    
    def _plot_dropout_results(self, results: Dict, title: str, filename: str) -> None:
        """
        Plot results from a dropout experiment.
        
        Args:
            results: Dropout experiment results
            title: Plot title
            filename: Filename for saving the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            dropout_fractions = sorted(results.keys())
            accuracies = [results[k]["accuracy"] for k in dropout_fractions]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot accuracy vs dropout fraction
            ax1.plot(dropout_fractions, accuracies, 'o-', linewidth=2)
            ax1.set_xlabel("Dropout Fraction")
            ax1.set_ylabel("Accuracy")
            ax1.set_title(f"{title} - Accuracy")
            ax1.grid(True)
            
            # Plot alignment values for each layer
            if len(results[dropout_fractions[0]]["alignment"]) > 0:
                for layer_idx in range(len(results[dropout_fractions[0]]["alignment"])):
                    layer_alignments = [
                        results[frac]["alignment"][layer_idx] 
                        for frac in dropout_fractions
                    ]
                    ax2.plot(dropout_fractions, layer_alignments, 'o-', linewidth=2, 
                             label=f"Layer {layer_idx}")
                
                ax2.set_xlabel("Dropout Fraction")
                ax2.set_ylabel("Alignment")
                ax2.set_title(f"{title} - {self.config.alignment.metric} Alignment")
                ax2.grid(True)
                ax2.legend()
            
            plt.tight_layout()
            self.plot_ready(filename)
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}", exc_info=True)


def cli_main():
    """Command-line interface for running alignment experiments."""
    parser = argparse.ArgumentParser(description="Neural network alignment experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.load(args.config)
    
    # Initialize and run experiment
    experiment = AlignmentExperiment(config)
    results, networks = experiment.run()
    
    return results, networks


if __name__ == "__main__":
    cli_main()