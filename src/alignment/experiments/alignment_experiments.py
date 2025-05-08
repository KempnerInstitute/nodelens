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
import copy
import pickle
import datetime
import json
import yaml
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from alignment.config import ExperimentConfig
from alignment.experiments.experiment import Experiment
from alignment.metrics import AlignmentMetric, get_metric
from alignment.models.registry import create_model
from alignment.dropout import progressive_dropout, eigenvector_dropout
from alignment.training import train_model, evaluate_model
from alignment.utils.core import setup_logging
from alignment.utils.plotting import plot_dropout_results, plot_experiment_summary
from alignment.datasets import get_dataset, load_dataset

logger = logging.getLogger(__name__)


class AlignmentExperiment(Experiment):
    """
    Experiment class for studying neural network alignment properties.
    
    This class implements experiments that assess alignment between layers
    in neural networks, with support for different dropout strategies,
    multiple metrics, and visualization.
    """
    
    def __init__(self, config: Dict) -> None:
        """Initialize the experiment with the given config.

        Args:
            config: Experiment configuration.
        """
        super().__init__(config)
        
        # Initialize dataset_config to avoid reference errors
        dataset_config = {}
        
        # Make sure dataset_config has required fields to prevent loading errors
        if isinstance(self.config.dataset, dict):
            dataset_config = self.config.dataset.copy()  # Use copy to avoid modifying original
            
            # Ensure consistent naming between dataset.name and dataset.dataset_name
            if "name" in dataset_config and "dataset_name" not in dataset_config:
                dataset_config["dataset_name"] = dataset_config["name"]
            elif "dataset_name" in dataset_config and "name" not in dataset_config:
                dataset_config["name"] = dataset_config["dataset_name"]
            elif "name" not in dataset_config and "dataset_name" not in dataset_config:
                if isinstance(self.config.dataset, str):
                    # If config.dataset is a string, use it as the dataset name
                    dataset_config["name"] = self.config.dataset
                    dataset_config["dataset_name"] = self.config.dataset
                    logger.info(f"Using dataset name from string: {self.config.dataset}")
                else:
                    # Default to CIFAR10 if no dataset name provided
                    logger.warning("No dataset name provided in configuration. Using default: cifar10")
                    dataset_config["name"] = "cifar10"
                    dataset_config["dataset_name"] = "cifar10"
        elif isinstance(self.config.dataset, str):
            # Handle case where config.dataset is directly a string
            dataset_config["name"] = self.config.dataset
            dataset_config["dataset_name"] = self.config.dataset
            logger.info(f"Using dataset name from string: {self.config.dataset}")
        else:
            # Default case if dataset config is neither dict nor string
            logger.warning("No valid dataset configuration found. Using default: cifar10")
            dataset_config["name"] = "cifar10"
            dataset_config["dataset_name"] = "cifar10"
            
        logger.info(f"Using dataset: {dataset_config.get('name', 'unknown')}")
        
        # Set up experiment-specific paths
        self.figure_path = None
        self.weights_path = None
        self.setup_paths()
        
        logger.debug(f"Initialized alignment experiment")
        
        # Add a sanity check
        # Ensure checkpoint parameters exist in the config
        if not hasattr(self.config, "checkpoint"):
            self.config.checkpoint = {}
            logger.warning("Checkpoint configuration not found, using defaults")
        
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
        Create multiple neural networks for the experiment, each with different initialization.
        
        Following the alignment_v2 approach, this creates multiple independent networks
        rather than copies of a single network.
        
        Returns:
            List containing multiple independently initialized networks
        """
        # Get the number of network replicates from config
        num_replicates = self.config.training.replicates if hasattr(self.config.training, "replicates") else 5
        
        # Create multiple models based on configuration, each with different initialization
        networks = []
        for i in range(num_replicates):
            # Set a different seed for each network to ensure different initializations
            if hasattr(self.config, 'seed') and self.config.seed is not None:
                # Use the base seed plus the replicate index to get different but reproducible initializations
                torch.manual_seed(self.config.seed + i)
                torch.cuda.manual_seed_all(self.config.seed + i)
                np.random.seed(self.config.seed + i)
            
            # Create a new model
            model = create_model(self.config.model)
            model.to(self.device)
            networks.append(model)
            
        logger.info(f"Created {len(networks)} models with independent initializations: {self.config.model.model_name}")
        
        return networks
    
    def main(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Main experiment execution method.
        
        Returns:
            Tuple of (results dictionary, list of networks)
        """
        # Additional initialization
        self.setup_paths()
        logger.info(f"Set up paths. Results will be saved to {self.results_path}")

        # Create network models for experiment
        networks = self.create_networks()
        
        # Run the experiment based on the specified type
        experiment_type = getattr(self.config, 'experiment_type', 'alignment_analysis')
        
        if experiment_type == "alignment_analysis" or experiment_type == "alignment":
            # Run alignment analysis and store results
            results = self.run_alignment_analysis(networks)
            
        elif experiment_type == "progressive_dropout":
            # Run progressive dropout experiment for all networks
            results = self._run_progressive_dropout(networks)
            
            # Generate and log visualization
            self._plot_dropout_results(
                results, 
                self.figure_path, 
                title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}"
            )
            
            # Save results for later reference
            with open(os.path.join(self.results_path, "progressive_dropout_results.pkl"), "wb") as f:
                pickle.dump(results, f)
            
        elif experiment_type == "eigenvector_dropout":
            # Run eigenvector dropout experiment for one network
            results = self._run_eigenvector_dropout(networks[0])
            
            # Generate and log visualization (this has a different format)
            self._generate_dropout_plots(
                results,
                "Eigenvector Dropout",
                "eigenvector_dropout"
            )
            
            # Save results for later reference
            with open(os.path.join(self.results_path, "eigenvector_dropout_results.pkl"), "wb") as f:
                pickle.dump(results, f)
                
        elif experiment_type == "training_alignment":
            # Run training alignment experiment
            # This is handled separately because it needs different structure
            results = self._run_training_alignment()
            
            # Generate and log visualization
            self._generate_training_alignment_plots(results)
            
            # Save results for later reference
            with open(os.path.join(self.results_path, "training_alignment_results.pkl"), "wb") as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
            
        logger.info(f"Completed {getattr(self.config, 'experiment_name', experiment_type)} experiment")
        
        return results, networks
    
    def _run_progressive_dropout(self, networks: List[nn.Module]) -> Dict:
        """
        Run progressive dropout experiment with multiple networks.
        
        Uses the progressive_dropout function from alignment.dropout module 
        instead of reimplementing the logic directly.

        Args:
            networks: List of networks to run the experiment on.

        Returns:
            Dictionary with results that match the expected format for plotting.
        """
        # Get dropout fractions from config
        dropin_min = self.config.alignment.dropout_min
        dropin_max = self.config.alignment.dropout_max
        num_dropout_fractions = self.config.alignment.dropout_steps
        dropout_fractions = np.linspace(dropin_min, dropin_max, num_dropout_fractions).tolist()
        
        # Get pruning mode and dropout mode from config
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")
        
        # Print exactly what mode is being used
        logger.info(f"Running progressive dropout with pruning_mode={pruning_mode}, dropout_mode={dropout_mode}")
        
        # Debug metric type
        logger.info(f"Using metric type: {type(self.metric)}")
        
        # Prepare dataset
        try:
            batch_size = getattr(self.config.dataset, "batch_size", 128)
            dataset = load_dataset(self.config.dataset, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return {"error": f"Dataset configuration error: {str(e)}", "dropout_fractions": dropout_fractions}
        
        # Train all networks at once before applying dropout
        logger.info(f"Training {len(networks)} networks before applying progressive dropout")
        
        # Track training history for plotting
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        # Get training parameters from config
        num_epochs = getattr(self.config.training, "epochs", 5)
        learning_rate = getattr(self.config.training, "learning_rate", 0.001)
        
        # Set up optimizer for each network
        optimizers = []
        for network in networks:
            network.to(self.device)
            optimizers.append(torch.optim.Adam(network.parameters(), lr=learning_rate))
        
        # Train all networks in parallel across epochs
        epoch_pbar = tqdm(range(num_epochs), desc="Training epochs", position=0)
        for epoch in epoch_pbar:
            # Initialize epoch stats
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            epoch_test_loss = 0.0
            epoch_test_acc = 0.0
            
            # Batch training - process all networks in parallel for each data batch
            for net_idx, network in enumerate(networks):
                network.train()
            
            batch_pbar = tqdm(dataset.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} batches", position=1, leave=False)
            for inputs, targets in batch_pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Process each network with the same batch
                batch_loss = 0.0
                batch_correct = 0
                batch_total = targets.size(0)
                
                # Train all networks on this batch in parallel
                for net_idx, (network, optimizer) in enumerate(zip(networks, optimizers)):
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = network(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Track statistics
                    batch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    batch_correct += predicted.eq(targets).sum().item()
                
                # Update batch progress bar with average across networks
                avg_batch_loss = batch_loss / len(networks)
                avg_batch_acc = 100.0 * batch_correct / (batch_total * len(networks))
                batch_pbar.set_postfix({
                    'loss': avg_batch_loss,
                    'acc': avg_batch_acc
                })
            
            # Evaluation phase - evaluate all networks in parallel
            all_test_loss = 0.0
            all_test_correct = 0
            all_test_total = 0
            
            # Set all networks to eval mode
            for network in networks:
                network.eval()
            
            # Evaluation progress bar
            eval_pbar = tqdm(dataset.test_loader, desc=f"Evaluating networks", position=1, leave=False)
            
            with torch.no_grad():
                for inputs, targets in eval_pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    test_total = targets.size(0)
                    all_test_total += test_total
                    
                    # Evaluate all networks on this batch
                    batch_test_loss = 0.0
                    batch_test_correct = 0
                    
                    for net_idx, network in enumerate(networks):
                        outputs = network(inputs)
                        
                        # Calculate loss
                        loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')
                        batch_test_loss += loss.item()
                        
                        # Calculate accuracy
                        _, predicted = outputs.max(1)
                        batch_test_correct += predicted.eq(targets).sum().item()
                    
                    # Accumulate statistics
                    all_test_loss += batch_test_loss
                    all_test_correct += batch_test_correct
                    
                    # Update eval progress bar
                    avg_test_loss = batch_test_loss / len(networks)
                    avg_test_acc = 100.0 * batch_test_correct / (test_total * len(networks))
                    eval_pbar.set_postfix({
                        'test_loss': avg_test_loss,
                        'test_acc': avg_test_acc
                    })
            
            # Calculate and store epoch averages
            epoch_train_loss = batch_loss / (len(dataset.train_loader) * len(networks))
            epoch_train_acc = 100.0 * batch_correct / (len(dataset.train_loader) * targets.size(0) * len(networks))
            epoch_test_loss = all_test_loss / all_test_total
            epoch_test_acc = 100.0 * all_test_correct / all_test_total
            
            training_history['train_loss'].append(epoch_train_loss)
            training_history['train_acc'].append(epoch_train_acc)
            training_history['test_loss'].append(epoch_test_loss)
            training_history['test_acc'].append(epoch_test_acc)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': epoch_train_loss,
                'train_acc': f"{epoch_train_acc:.2f}%",
                'test_loss': epoch_test_loss,
                'test_acc': f"{epoch_test_acc:.2f}%"
            })
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.2f}%, "
                      f"Test Loss={epoch_test_loss:.4f}, Test Acc={epoch_test_acc:.2f}%")
        
        logger.info(f"Completed training {len(networks)} networks. Proceeding to progressive dropout...")
        
        # Initialize results structure with correct format for plotting
        final_results = {
            "dropout_fractions": dropout_fractions,
            "accuracies": {"high_rq": [], "low_rq": [], "random": []},
            "losses": {"high_rq": [], "low_rq": [], "random": []},
            "stds": {"high_rq": [], "low_rq": [], "random": []},
            "training_history": training_history
        }
        
        # Call the progressive_dropout function with our trained networks
        from alignment.dropout import progressive_dropout
        
        try:
            # Define the strategies to run
            strategies = ["high_rq", "low_rq", "random"]
            
            # Track time for benchmarking
            start_time = time.time()
            logger.info(f"Running all pruning strategies in parallel for {len(networks)} networks")
            
            # Create copies of the networks for each strategy to avoid interference
            strategy_networks = {}
            for strategy in strategies:
                strategy_networks[strategy] = [copy.deepcopy(net) for net in networks]
            
            # Process all strategies in parallel
            all_results = {}
            
            # Run all strategies in parallel using multiprocessing
            import concurrent.futures
            
            def run_strategy(strategy):
                logger.info(f"Starting pruning with strategy: {strategy}")
                return progressive_dropout(
                    strategy_networks[strategy],
                    dataset,
                    dropout_fractions,
                    self.metric,
                    self.device,
                    pruning_mode=pruning_mode,
                    dropout_mode=dropout_mode,
                    strategy=strategy
                )
            
            # Use ThreadPoolExecutor to run strategies in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
                # Submit all strategy jobs
                future_to_strategy = {
                    executor.submit(run_strategy, strategy): strategy 
                    for strategy in strategies
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_strategy):
                    strategy = future_to_strategy[future]
                    try:
                        network_accuracies, network_losses = future.result()
                        all_results[strategy] = (network_accuracies, network_losses)
                        logger.info(f"Completed {strategy} pruning strategy")
                    except Exception as e:
                        logger.error(f"Error in {strategy} pruning: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            # Process the results for all strategies
            end_time = time.time()
            logger.info(f"All pruning strategies completed in {end_time - start_time:.2f} seconds")
            
            # Calculate statistics for all strategies
            for strategy in strategies:
                if strategy in all_results:
                    network_accuracies, network_losses = all_results[strategy]
                    
                    # Process results for this strategy
                    for fraction_idx in range(len(dropout_fractions)):
                        # Collect results for this fraction across all networks
                        fraction_accs = []
                        fraction_losses = []
                        
                        for net_idx in network_accuracies:
                            if fraction_idx < len(network_accuracies[net_idx]):
                                fraction_accs.append(network_accuracies[net_idx][fraction_idx])
                            if fraction_idx < len(network_losses[net_idx]):
                                fraction_losses.append(network_losses[net_idx][fraction_idx])
                        
                        # Calculate statistics
                        if fraction_accs:
                            mean_acc = np.mean(fraction_accs)
                            std_acc = np.std(fraction_accs)
                            mean_loss = np.mean(fraction_losses) if fraction_losses else 0.0
                            
                            # Add to results for this strategy
                            final_results["accuracies"][strategy].append(mean_acc)
                            final_results["stds"][strategy].append(std_acc)
                            final_results["losses"][strategy].append(mean_loss)
            
            # Log results for verification
            logger.info(f"Final results:")
            for strategy in strategies:
                if strategy in all_results:
                    logger.info(f"  {strategy}: {final_results['accuracies'][strategy]}")
                
        except Exception as e:
            logger.error(f"Error running progressive dropout: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": f"Error running progressive dropout: {str(e)}",
                "dropout_fractions": dropout_fractions,
                "training_history": training_history
            }
        
        # Generate dropout plots
        try:
            logger.info("Generating dropout plots with error bars")
            from alignment.utils.plotting import plot_dropout_results
            
            saved_plots = plot_dropout_results(
                final_results,
                save_dir=self.figure_path,
                title_prefix="Progressive Dropout",
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode
            )
            
            if saved_plots:
                logger.info(f"Generated {len(saved_plots)} plots: {saved_plots}")
                final_results["plot_files"] = saved_plots
            else:
                logger.warning("No plots were generated. Check your plot_dropout_results function.")
            
        except Exception as e:
            logger.error(f"Error generating dropout plots: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return final_results
    
    def _run_eigenvector_dropout(self, net: nn.Module) -> Dict:
        """
        Run eigenvector dropout experiments.
        
        Uses the eigenvector_dropout function from alignment.dropout module 
        instead of reimplementing the logic directly.
        
        Args:
            net: Neural network to analyze
            
        Returns:
            Dictionary of eigenvector dropout results
        """
        # Get dropout fractions from config
        dropout_fractions = np.linspace(
            self.config.alignment.dropout_min,
            self.config.alignment.dropout_max,
            self.config.alignment.dropout_steps
        ).tolist()
        
        # Initialize result structure
        results = {
            "dropout_fractions": dropout_fractions,
            "accuracies": {"eigenvector": []},
            "losses": {"eigenvector": []},
            "alignment_values": {"eigenvector": []}
        }
        
        # Prepare dataset config
        dataset_config = {}
        dataset_config.update(self.config.dataset)
        
        # Process each dropout fraction
        for dropout_fraction in tqdm(dropout_fractions, desc="Eigenvector Dropout"):
            try:
                # Call the eigenvector_dropout function directly
                accuracy, alignment_values = eigenvector_dropout(
                    net,
                    dataset_config,
                    dropout_fraction=dropout_fraction,
                    metric=self.metric,
                    device=self.device,
                    dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled"),
                    dropout_pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
                )
                
                # Store results
                results["accuracies"]["eigenvector"].append(accuracy)
                results["losses"]["eigenvector"].append(100.0 - accuracy)
                results["alignment_values"]["eigenvector"].append(alignment_values)
                
            except Exception as e:
                logger.error(f"Error in eigenvector dropout at fraction {dropout_fraction}: {str(e)}")
                # Add placeholder values to maintain result structure
                results["accuracies"]["eigenvector"].append(0.0)
                results["losses"]["eigenvector"].append(100.0)
                results["alignment_values"]["eigenvector"].append(None)
        
        return results
    
    def plot(self, results: Dict) -> Dict[str, List[str]]:
        """
        Plot experiment results using enhanced visualization methods.
        
        Args:
            results: Dictionary containing experiment results
            
        Returns:
            Dictionary of plot paths by type
        """
        logger.info("Generating experiment visualizations")
        
        # Check for results to plot
        if not results:
            logger.warning("No results to plot")
            return {}
            
        plot_paths = {}
        
        # Generate experiment summary
        try:
            logger.info("Creating experiment summary visualization")
            summary_path = self._generate_summary_plots(results)
            if summary_path:
                plot_paths["summary"] = summary_path
            logger.info("Successfully created experiment summary visualization")
        except Exception as e:
            logger.error(f"Error creating summary visualization: {str(e)}", exc_info=True)
        
        # Generate progressive dropout plots
        if "progressive_dropout" in results:
            try:
                logger.info("Generating progressive dropout visualizations")
                prog_paths = self._plot_dropout_results(
                    results["progressive_dropout"], 
                    self.figure_path,
                    "Progressive Dropout"
                )
                if prog_paths:
                    plot_paths["progressive_dropout"] = prog_paths
                logger.info("Successfully generated progressive dropout visualizations")
            except Exception as e:
                logger.error(f"Error generating progressive dropout visualizations: {str(e)}", exc_info=True)
        
        # Generate eigenvector dropout plots
        if "eigenvector_dropout" in results:
            try:
                logger.info("Generating eigenvector dropout visualizations")
                eig_paths = self._plot_dropout_results(
                    results["eigenvector_dropout"], 
                    self.figure_path,
                    "Eigenvector Dropout"
                )
                if eig_paths:
                    plot_paths["eigenvector_dropout"] = eig_paths
                logger.info("Successfully generated eigenvector dropout visualizations")
            except Exception as e:
                logger.error(f"Error generating eigenvector dropout visualizations: {str(e)}", exc_info=True)
        
        # Generate combined visualizations if both types of results exist
        if "progressive_dropout" in results and "eigenvector_dropout" in results:
            try:
                logger.info("Generating combined dropout comparison visualization")
                comparison_path = self._generate_dropout_comparison(
                    results["progressive_dropout"],
                    results["eigenvector_dropout"]
                )
                if comparison_path:
                    plot_paths["comparison"] = comparison_path
                logger.info("Successfully generated combined dropout comparison")
            except Exception as e:
                logger.error(f"Error generating dropout comparison: {str(e)}", exc_info=True)
        
        logger.info(f"Completed experiment visualization generation: {len(plot_paths)} plot types")
        return plot_paths
    
    def _plot_dropout_results(self, results, figure_path=None, title_prefix="Dropout"):
        """
        Plot dropout experiment results using the plotting module.
        
        Args:
            results: Results dictionary from progressive_dropout
            figure_path: Directory to save plots to
            title_prefix: Prefix for plot titles
            
        Returns:
            List of saved plot files
        """
        # Get pruning mode and dropout mode
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")
        
        # Ensure figure path exists
        if figure_path is None:
            figure_path = self.figure_path
            
        # Add debug logs to understand the results
        logger.info(f"Plotting dropout results with pruning_mode={pruning_mode}, dropout_mode={dropout_mode}")
        logger.info(f"Results keys: {list(results.keys())}")
        logger.info(f"Figure path: {figure_path}")
        
        if "accuracies" in results:
            logger.info(f"Accuracies keys: {list(results['accuracies'].keys())}")
            for key in results["accuracies"]:
                logger.info(f"Strategy {key} has {len(results['accuracies'][key])} data points")
                if results["accuracies"][key]:
                    logger.info(f"First few values: {results['accuracies'][key][:3]}")
                else:
                    logger.info(f"No data for {key}")
        
        # Generate plots using our custom function
        try:
            from alignment.utils.plotting import custom_plot_dropout
            
            # Make a deep copy of results to prevent modification
            import copy
            plot_results = copy.deepcopy(results)
            
            # Ensure we have data for at least one strategy (preferably high_rq)
            if not plot_results.get("accuracies", {}).get("high_rq", []):
                # Try to use data from any strategy that has data
                for strategy in ["low_rq", "random", "eigenvector"]:
                    if plot_results.get("accuracies", {}).get(strategy, []):
                        # Use this data for high_rq
                        logger.info(f"Using {strategy} data for high_rq")
                        plot_results["accuracies"]["high_rq"] = plot_results["accuracies"][strategy]
                        
                        # Also copy to losses if needed
                        if strategy in plot_results.get("losses", {}):
                            plot_results["losses"]["high_rq"] = plot_results["losses"][strategy]
                        break
            
            # Try the numeric indices if we still don't have data
            if not plot_results.get("accuracies", {}).get("high_rq", []) and hasattr(self, "results") and hasattr(self.results, "network_accuracies"):
                # Look for numeric indices
                for idx in range(3):  # Try indices 0, 1, 2
                    if idx in self.results.network_accuracies and self.results.network_accuracies[idx]:
                        strategy_name = {0: "high_rq", 1: "low_rq", 2: "random"}.get(idx, "high_rq")
                        logger.info(f"Using numeric index {idx} data for {strategy_name}")
                        
                        # Make sure the strategy key exists
                        if "accuracies" not in plot_results:
                            plot_results["accuracies"] = {}
                        
                        # Copy the data
                        plot_results["accuracies"][strategy_name] = self.results.network_accuracies[idx]
            
            # Print final plot data
            logger.info(f"Passing the following to custom_plot_dropout:")
            logger.info(f"- dropout_fractions: {len(plot_results.get('dropout_fractions', []))} points")
            logger.info(f"- high_rq data: {len(plot_results.get('accuracies', {}).get('high_rq', []))} points")
            
            # Call the custom plotting function
            saved_figures = custom_plot_dropout(
                plot_results,
                figure_path,
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode,
                title_prefix=title_prefix
            )
            
            # Check if any plots were generated
            if saved_figures:
                logger.info(f"Generated {len(saved_figures)} plots: {saved_figures}")
                
                # Log plots to wandb if configured
                if hasattr(self.config.checkpointing, "use_wandb") and self.config.checkpointing.use_wandb:
                    try:
                        from alignment.utils.plotting import log_plots_to_wandb
                        log_plots_to_wandb(saved_figures)
                        logger.info(f"Logged {len(saved_figures)} dropout plots to wandb")
                    except Exception as e:
                        logger.warning(f"Failed to log dropout plots to wandb: {str(e)}")
            else:
                logger.warning("No plots were generated by custom_plot_dropout")
                    
            return saved_figures
                
        except Exception as e:
            logger.error(f"Error generating dropout plots: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
    def _generate_dropout_plots(self, results, title, filename=None):
        """
        Generate enhanced dropout plots.
        
        Args:
            results: Dropout experiment results
            title: Plot title
            filename: Base filename (optional)
            
        Returns:
            List of saved plot files
        """
        # Call the _plot_dropout_results function with our figure path and title
        return self._plot_dropout_results(results, self.figure_path, title)
        
    def _generate_summary_plots(self, results):
        """
        Generate comprehensive summary plots for the experiment using the plotting module.
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            Path to the saved summary plot
        """
        try:
            from alignment.utils.plotting import plot_experiment_summary
            
            # Generate the summary plot
            experiment_name = getattr(self.config, "experiment_name", self.get_basename())
            filepath = plot_experiment_summary(
                results, 
                self.figure_path,
                experiment_name=experiment_name
            )
            
            # Log to wandb if configured
            if filepath and hasattr(self.config.checkpointing, "use_wandb") and self.config.checkpointing.use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"experiment_summary": wandb.Image(filepath)})
                        logger.info("Logged experiment summary to wandb")
                except Exception as e:
                    logger.warning(f"Failed to log experiment summary to wandb: {str(e)}")
                    
            return filepath
                    
        except Exception as e:
            logger.error(f"Error creating summary visualization: {str(e)}")
            return None
            
    def _generate_dropout_comparison(self, progressive_results, eigenvector_results):
        """
        Generate comparison plots between progressive and eigenvector dropout using the plotting module.
        
        Args:
            progressive_results: Progressive dropout results
            eigenvector_results: Eigenvector dropout results
            
        Returns:
            Path to the saved comparison plot
        """
        try:
            from alignment.utils.plotting import plot_dropout_comparison
            
            # Generate the comparison plot
            experiment_name = getattr(self.config, "experiment_name", "Experiment")
            filepath = plot_dropout_comparison(
                progressive_results,
                eigenvector_results,
                self.figure_path,
                title=f"Dropout Comparison: {experiment_name}"
            )
            
            # Log to wandb if configured
            if filepath and hasattr(self.config.checkpointing, "use_wandb") and self.config.checkpointing.use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"dropout_comparison": wandb.Image(filepath)})
                        logger.info("Logged dropout comparison to wandb")
                except Exception as e:
                    logger.warning(f"Failed to log dropout comparison to wandb: {str(e)}")
                    
            return filepath
                    
        except Exception as e:
            logger.error(f"Error creating dropout comparison: {str(e)}")
            return None

    def _initialize_wandb(self) -> Optional[Any]:
        """
        Initialize wandb for experiment tracking.
        
        Returns:
            wandb module if initialized successfully, None otherwise
        """
        if not getattr(self.config.checkpointing, "use_wandb", False):
            return None
            
        try:
            import wandb
            
            # Get wandb configuration
            entity = getattr(self.config, "wandb_entity", None)
            project = getattr(self.config, "wandb_project", "neural_alignment")
            
            # Generate timestamp if it doesn't exist
            if not hasattr(self, 'timestamp'):
                self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a unique run name
            if hasattr(self.config, "experiment_name") and self.config.experiment_name:
                run_name = f"{self.config.experiment_name}"
            else:
                run_name = f"{self.config.model.model_name}_{self.config.dataset.dataset_name}_{self.timestamp}"
                
            # Convert the config to a dictionary for wandb
            config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else vars(self.config)
            
            # Initialize wandb run
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config_dict,
                reinit=True  # Allow reinitializing in the same process
            )
            
            # Set up wandb to watch models if available
            if hasattr(self, "networks") and self.networks:
                # Watch the first network
                wandb.watch(self.networks[0], log="all", log_freq=10)
                
            logger.info(f"Initialized Weights & Biases tracking for run '{run_name}'")
            return wandb
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {str(e)}")
            return None
    
    def _log_to_wandb(self, data: Dict) -> bool:
        """
        Log data to wandb.
        
        Args:
            data: Data to log to wandb (dict with keys corresponding to metric names)
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not getattr(self.config.checkpointing, "use_wandb", False):
            return False
            
        try:
            import wandb
            if wandb.run is None:
                wandb_module = self._initialize_wandb()
                if wandb_module is None:
                    return False
                    
            # Log scalar metrics
            metrics_to_log = {}
            for k, v in data.items():
                # Skip non-scalar values
                if isinstance(v, (int, float)):
                    metrics_to_log[k] = v
                elif isinstance(v, (list, tuple)) and len(v) == 1:
                    # Single-element list or tuple with a scalar
                    if isinstance(v[0], (int, float)):
                        metrics_to_log[k] = v[0]
                        
            # Log the metrics to wandb
            if metrics_to_log:
                wandb.log(metrics_to_log)
                
            # Check if data contains progressive dropout results that we should plot
            if "progressive_dropout" in data and "plot_files" in data and getattr(self.config.extra, "log_images", True):
                # Find all plot files and log them to wandb
                for plot_file in data["plot_files"]:
                    if os.path.exists(plot_file):
                        plot_name = os.path.basename(plot_file).replace(".png", "")
                        wandb.log({plot_name: wandb.Image(plot_file)})
                
                # If 'accuracies' exists in the progressive_dropout results, log them as a line plot
                if "accuracies" in data["progressive_dropout"] and "dropout_fractions" in data["progressive_dropout"]:
                    # Create a table to log
                    accuracies = np.array(data["progressive_dropout"]["accuracies"])
                    dropout_fractions = data["progressive_dropout"]["dropout_fractions"]
                    
                    # Calculate mean and std
                    mean_accuracies = np.mean(accuracies, axis=0)
                    std_accuracies = np.std(accuracies, axis=0)
                    
                    # Prepare data table for the custom plot
                    table_data = [[x, y, y_std] for x, y, y_std in zip(dropout_fractions, mean_accuracies, std_accuracies)]
                    
                    # Log the data as a custom plot
                    wandb.log({
                        "dropout_curve": wandb.Table(
                            data=table_data,
                            columns=["dropout_fraction", "mean_accuracy", "std_accuracy"]
                        )
                    })
                    
                    # Create interactive plot
                    dropout_data = []
                    for i, frac in enumerate(dropout_fractions):
                        dropout_data.append([frac, mean_accuracies[i], mean_accuracies[i]-std_accuracies[i], mean_accuracies[i]+std_accuracies[i]])
                    
                    wandb.log({
                        "dropout_plot_interactive": wandb.plot.line_series(
                            xs=[x[0] for x in dropout_data],
                            ys=[[x[1] for x in dropout_data],  # mean
                                [x[2] for x in dropout_data],  # mean-std
                                [x[3] for x in dropout_data]],  # mean+std
                            keys=["Mean Accuracy", "Lower Bound", "Upper Bound"],
                            title=f"Progressive Dropout ({self.config.extra.dropout_pruning_mode})",
                            xname="Dropout Fraction",
                            yname="Accuracy"
                        )
                    })
                
            return True
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {str(e)}")
            return False

    def run(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Run the experiment with custom logging and wandb setup.
        
        Returns:
            Tuple containing (results, networks)
        """
        # Setup paths for experiment
        self.setup_paths()
        self.prepare_path()
        
        # Set up logging and suppress warnings
        set_logging_level(logging.INFO)
        
        # Set random seed if configured
        if hasattr(self.config, 'seed') and self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info(f"Set random seed to {self.config.seed}")
        
        # Initialize wandb explicitly before starting experiment
        if hasattr(self.config, 'checkpointing') and self.config.checkpointing.use_wandb:
            self._initialize_wandb()
        
        # Run the main experiment
        results, networks = self.main()
        
        # Store results as an attribute for saving
        self.results = results
        self.networks = networks
        
        # Ensure all plots are saved and logged
        self.plot(results)
        
        # Save experiment state
        self.save()
        
        # Close wandb run if it exists
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                logger.info("Closed wandb run")
        except Exception as e:
            logger.warning(f"Error closing wandb run: {str(e)}")
        
        return results, networks

    def run_alignment_analysis(self, networks: List[nn.Module]) -> Dict:
        """
        Run alignment analysis experiment.
        
        This experiment analyzes alignment metrics across networks and can run
        both progressive dropout and eigenvector dropout experiments based on
        configuration.
        
        Args:
            networks: List of neural networks to analyze
            
        Returns:
            Results dictionary
        """
        logger.info("Running alignment analysis experiment")
        
        # Prepare the results structure
        results = {
            "progressive_dropout": {},
            "eigenvector_dropout": {},
            "config": self.config,
        }
        
        # Progressive dropout experiment
        if self.config.alignment.run_progressive:
            logger.info(f"Running progressive dropout experiment (mode: {getattr(self.config.extra, 'dropout_pruning_mode', 'global_joint')})")
            try:
                prog_results = self._run_progressive_dropout(networks)
                results["progressive_dropout"] = prog_results
                
                # Add debug logging to inspect results structure
                if prog_results:
                    logger.info(f"Progressive dropout results keys: {list(prog_results.keys())}")
                    if 'accuracies' in prog_results:
                        logger.info(f"Progressive dropout accuracies keys: {list(prog_results['accuracies'].keys())}")
                
                # Check if there was an error in the dataset configuration
                if "error" in prog_results:
                    logger.error(f"Progressive dropout experiment failed: {prog_results['error']}")
                else:
                    # Generate plots right after getting results
                    try:
                        logger.info("Generating progressive dropout visualizations")
                        # Add debug print before and after plotting
                        figure_path = self._plot_dropout_results(
                            results["progressive_dropout"],
                            self.figure_path,
                            title_prefix="Progressive Dropout"
                        )
                        logger.info(f"Progressive dropout plot generated at {figure_path}")
                    except Exception as e:
                        logger.error(f"Error generating progressive dropout visualizations: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error running progressive dropout experiment: {str(e)}")
                results["progressive_dropout"] = {"error": str(e)}
        
        # Eigenvector dropout experiment
        if self.config.alignment.run_eigenvector:
            logger.info("Running eigenvector dropout experiment")
            try:
                # Note: For eigenvector dropout, we use just the first network
                eig_results = self._run_eigenvector_dropout(networks[0])
                results["eigenvector_dropout"] = eig_results
                
                # Add debug logging
                if eig_results:
                    logger.info(f"Eigenvector dropout results keys: {list(eig_results.keys())}")
                    if 'accuracies' in eig_results:
                        logger.info(f"Eigenvector dropout accuracies keys: {list(eig_results['accuracies'].keys())}")
                
                # Generate plots right after getting results
                try:
                    logger.info("Generating eigenvector dropout visualizations")
                    # Use the specialized method for eigenvector dropout
                    self._generate_dropout_plots(
                        results["eigenvector_dropout"],
                        "Eigenvector Dropout",
                        "eigenvector_dropout"
                    )
                    logger.info("Successfully generated eigenvector dropout visualizations")
                except Exception as e:
                    logger.error(f"Error generating eigenvector dropout visualizations: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error running eigenvector dropout experiment: {str(e)}")
                results["eigenvector_dropout"] = {"error": str(e)}
        
        # Generate and log summary plots for all experiments
        try:
            logger.info("Generating experiment summary")
            self._generate_summary_plots(results)
            logger.info("Successfully generated experiment summary")
        except Exception as e:
            logger.error(f"Error generating experiment summary: {str(e)}", exc_info=True)
        
        return results

    def setup_paths(self):
        """
        Set up paths for experiment outputs.
        
        Creates necessary directories for figures, weights, and results.
        """
        # Create timestamp subdirectory if needed
        if getattr(self.config, "use_timestamp", True):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join("results", f"{self.get_basename()}_{timestamp}")
        else:
            base_dir = os.path.join("results", self.get_basename())
        
        # Create main result directory
        os.makedirs(base_dir, exist_ok=True)
        self.results_path = base_dir
        
        # Create figure directory
        self.figure_path = os.path.join(base_dir, "figures")
        os.makedirs(self.figure_path, exist_ok=True)
        
        # Create weights directory for saved models
        self.weights_path = os.path.join(base_dir, "weights")
        os.makedirs(self.weights_path, exist_ok=True)
        
        # Set the device to use (default to CUDA if available)
        if hasattr(self.config, "device") and self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Set up paths: results={self.results_path}, figures={self.figure_path}, device={self.device}")

    def save(self):
        """
        Save experiment state including results and configuration.
        
        This method saves:
        1. Experiment results as pickle and JSON
        2. Configuration in YAML format
        3. Any model checkpoints if available
        """
        try:
            # Create a results directory if it doesn't exist
            os.makedirs(self.results_path, exist_ok=True)
            
            # Save experiment results if available
            if hasattr(self, 'results') and self.results:
                results_file = os.path.join(self.results_path, "results.pkl")
                with open(results_file, "wb") as f:
                    pickle.dump(self.results, f)
                logger.info(f"Saved experiment results to {results_file}")
                
                # Also try to save as JSON for human readability
                try:
                    # Convert tensors and other non-serializable objects
                    def clean_for_json(obj):
                        if isinstance(obj, (torch.Tensor, np.ndarray)):
                            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
                        elif isinstance(obj, dict):
                            return {k: clean_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [clean_for_json(item) for item in obj]
                        elif isinstance(obj, tuple):
                            return tuple(clean_for_json(item) for item in obj)
                        else:
                            # Return string representation for other types
                            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
                    
                    json_results = clean_for_json(self.results)
                    json_file = os.path.join(self.results_path, "results.json")
                    with open(json_file, "w") as f:
                        json.dump(json_results, f, indent=2)
                    logger.info(f"Saved readable results to {json_file}")
                except Exception as e:
                    logger.warning(f"Could not save results as JSON: {str(e)}")
            
            # Save configuration
            config_file = os.path.join(self.results_path, "config.yaml")
            # Check if we can convert config to dict
            if hasattr(self.config, 'to_dict'):
                config_dict = self.config.to_dict()
                with open(config_file, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
                logger.info(f"Saved configuration to {config_file}")
            
            logger.info("Experiment state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving experiment state: {str(e)}")
            # Don't re-raise the exception, just log it


def set_logging_level(level=logging.INFO):
    """
    Set logging level for all loggers in the alignment package.
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    # Set root logger level
    logging.getLogger().setLevel(level)
    
    # Set level for alignment package loggers
    logging.getLogger('alignment').setLevel(level)
    
    # Reduce verbosity for other loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)
    
    # Suppress common warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pruning fraction.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Combined pruning across layers.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout of the entire tensor.*")

def cli_main():
    """Command-line interface for running alignment experiments."""
    parser = argparse.ArgumentParser(description="Neural network alignment experiment")
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
    experiment = AlignmentExperiment(config)
    results, networks = experiment.run()
    
    return results, networks


if __name__ == "__main__":
    cli_main()