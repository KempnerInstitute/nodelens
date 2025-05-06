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
from alignment import plotting
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
        
        # Make sure dataset_config has required fields to prevent loading errors
        if isinstance(self.config.dataset, dict):
            dataset_config = self.config.dataset
            
            # Ensure dataset_config has both "name" and "dataset_name" fields for compatibility
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
        
        Implementation follows the approach used in the preref code for efficiency,
        processing all dropout fractions in a batched fashion.

        Args:
            networks: List of networks to run the experiment on.

        Returns:
            Dictionary with results.
        """
        # Initialize results structure
        results = {
            "accuracies": {},  # Strategy -> list of accuracies
            "losses": {},  # Strategy -> list of losses
            "alignment_values": {},  # Strategy -> list of alignment values
            "dropout_fractions": [],  # List of dropout fractions
        }

        # Get pruning_mode and dropout_mode
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global")
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")

        logger.info(f"Running progressive dropout with pruning_mode={pruning_mode}, dropout_mode={dropout_mode}")

        # Get dropout fractions from config
        dropin_min = self.config.alignment.dropout_min
        dropin_max = self.config.alignment.dropout_max
        num_dropout_fractions = self.config.alignment.dropout_steps
        dropout_fractions = np.linspace(dropin_min, dropin_max, num_dropout_fractions)
        results["dropout_fractions"] = dropout_fractions.tolist()  # Convert to list to avoid issues later

        # Prepare dataset
        try:
            from alignment.datasets import load_dataset
            batch_size = getattr(self.config.training, "batch_size", 128)
            dataset = load_dataset(self.config.dataset, batch_size=batch_size)
            test_loader = dataset.test_loader
            logger.info(f"Successfully validated dataset configuration")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            results["error"] = f"Dataset configuration error: {str(e)}"
            return results

        # Initialize arrays for all strategies
        strategies = ["high_rq", "low_rq", "random"]
        for strategy in strategies:
            results["accuracies"][strategy] = [0.0] * len(dropout_fractions)
            results["losses"][strategy] = [0.0] * len(dropout_fractions)
            results["alignment_values"][strategy] = [None] * len(dropout_fractions)

        # Reduce verbosity during computation
        original_log_level = logging.getLogger('alignment').level
        logging.getLogger('alignment').setLevel(logging.WARNING)
        
        # Initialize tensors to store results (preref style)
        num_nets = len(networks)
        num_drops = len(dropout_fractions)
        
        # Using GPU tensors for faster computation
        acc_high = torch.zeros((num_nets, num_drops), device=self.config.device)
        acc_low = torch.zeros((num_nets, num_drops), device=self.config.device)
        acc_rand = torch.zeros((num_nets, num_drops), device=self.config.device)
        loss_high = torch.zeros((num_nets, num_drops), device=self.config.device)
        loss_low = torch.zeros((num_nets, num_drops), device=self.config.device)
        loss_rand = torch.zeros((num_nets, num_drops), device=self.config.device)
        align_values = {}
        
        # Process each network with progress bar
        for i, network in enumerate(tqdm(networks, desc="Processing networks")):
            logger.debug(f"Processing network {i+1}/{len(networks)}")
            
            try:
                # Move network to device and put in eval mode
                network.to(self.config.device)
                network.eval()
                
                # Check if network has alignment_layers
                if not hasattr(network, "alignment_layers"):
                    logger.error(f"Network {i} doesn't have alignment_layers attribute")
                    continue
                
                # Get scores for each layer first (compute once, reuse across fractions)
                with torch.no_grad():
                    # Set up hooks to capture activations
                    if not hasattr(network, "hidden"):
                        network.hidden = {}
                        
                    hooks = []
                    def get_activation(name):
                        def hook(module, input, output):
                            network.hidden[name] = input[0].detach()
                        return hook
                        
                    # Register hooks for all alignment layers
                    for j, layer in enumerate(network.alignment_layers):
                        hooks.append(layer.register_forward_hook(get_activation(network.alignment_names[j])))
                    
                    # Run a batch through the network to get activations
                    batch = next(iter(test_loader))
                    inputs, _ = dataset.unwrap_batch(batch)
                    inputs = inputs.to(self.config.device)
                    network(inputs)
                    
                    # Compute node-level scores for each layer
                    layer_scores = {}
                    for layer_idx, layer_mod in enumerate(network.alignment_layers):
                        layer_name = network.alignment_names[layer_idx]
                        
                        # Check for classification layer
                        if layer_idx == len(network.alignment_layers) - 1:
                            # Skip scores for classification layer - we'll handle it later
                            layer_scores[layer_idx] = torch.zeros(layer_mod.weight.shape[0], device=self.config.device)
                            continue
                            
                        # Get layer input activations
                        if layer_name not in network.hidden:
                            logger.warning(f"No activation data for layer {layer_name}")
                            layer_scores[layer_idx] = torch.zeros(layer_mod.weight.shape[0], device=self.config.device)
                            continue
                            
                        layer_input = network.hidden[layer_name]
                        if layer_input.dim() > 2:
                            layer_input = layer_input.reshape(layer_input.size(0), -1)
                            
                        # Compute scores using the alignment metric
                        node_scores = self.metric.compute_per_node_scores(
                            layer_input, layer_mod.weight, device=self.config.device
                        )
                        layer_scores[layer_idx] = node_scores
                        
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                
                # Precompute dropout indices for all fractions at once to minimize redundant computation
                # This is how preref code handles it
                dropout_indices = {}
                for frac_idx, fraction in enumerate(tqdm(dropout_fractions, desc=f"Network {i+1}: Computing indices", leave=False)):
                    high_rq_dropout = []
                    low_rq_dropout = []
                    random_dropout = []
                    
                    if pruning_mode == "global":
                        # Collect all scores and indices for global pruning
                        global_scores = []
                        global_layer_indices = []
                        global_node_indices = []
                        
                        for layer_idx, scores in layer_scores.items():
                            for node_idx, score in enumerate(scores):
                                global_scores.append(score.item())
                                global_layer_indices.append(layer_idx)
                                global_node_indices.append(node_idx)
                                
                        # Convert to arrays
                        global_scores = np.array(global_scores)
                        global_indices = list(zip(global_layer_indices, global_node_indices))
                        
                        # Sort all scores
                        sorted_indices = np.argsort(global_scores)  # Ascending order
                        
                        # Determine number of nodes to drop
                        total_nodes = len(global_scores)
                        num_to_drop = int(round(total_nodes * fraction))
                        
                        # Get indices for each strategy
                        high_rq_indices = sorted_indices[-num_to_drop:] if num_to_drop > 0 else []
                        low_rq_indices = sorted_indices[:num_to_drop] if num_to_drop > 0 else []
                        random_indices = np.random.choice(total_nodes, num_to_drop, replace=False) if num_to_drop > 0 else []
                        
                        # Create dropout masks for each layer
                        high_rq_dropout = [[] for _ in range(len(network.alignment_layers))]
                        low_rq_dropout = [[] for _ in range(len(network.alignment_layers))]
                        random_dropout = [[] for _ in range(len(network.alignment_layers))]
                        
                        # Populate dropout indices per layer for each strategy
                        for idx in high_rq_indices:
                            layer_idx, node_idx = global_indices[idx]
                            high_rq_dropout[layer_idx].append(node_idx)
                            
                        for idx in low_rq_indices:
                            layer_idx, node_idx = global_indices[idx]
                            low_rq_dropout[layer_idx].append(node_idx)
                            
                        for idx in random_indices:
                            layer_idx, node_idx = global_indices[idx]
                            random_dropout[layer_idx].append(node_idx)
                            
                        # Convert lists to tensors
                        for layer_idx in range(len(network.alignment_layers)):
                            high_rq_dropout[layer_idx] = torch.tensor(
                                high_rq_dropout[layer_idx], device=self.config.device, dtype=torch.long
                            )
                            low_rq_dropout[layer_idx] = torch.tensor(
                                low_rq_dropout[layer_idx], device=self.config.device, dtype=torch.long
                            )
                            random_dropout[layer_idx] = torch.tensor(
                                random_dropout[layer_idx], device=self.config.device, dtype=torch.long
                            )
                    else:
                        # Per-layer pruning (independent or combined)
                        high_rq_dropout = []
                        low_rq_dropout = []
                        random_dropout = []
                        
                        for layer_idx, scores in layer_scores.items():
                            # Skip classification layer
                            if layer_idx == len(network.alignment_layers) - 1 and True:
                                high_rq_dropout.append(torch.tensor([], device=self.config.device, dtype=torch.long))
                                low_rq_dropout.append(torch.tensor([], device=self.config.device, dtype=torch.long))
                                random_dropout.append(torch.tensor([], device=self.config.device, dtype=torch.long))
                                continue
                                
                            # Calculate number of nodes to drop
                            layer_size = scores.size(0)
                            num_to_drop = int(round(layer_size * fraction))
                            
                            # Get dropout indices (more efficient calculation)
                            sorted_indices = torch.argsort(scores)  # Ascending
                            
                            # High RQ: Drop highest RQ nodes
                            high_indices = sorted_indices[-num_to_drop:] if num_to_drop > 0 else torch.tensor([], device=self.config.device, dtype=torch.long)
                            
                            # Low RQ: Drop lowest RQ nodes
                            low_indices = sorted_indices[:num_to_drop] if num_to_drop > 0 else torch.tensor([], device=self.config.device, dtype=torch.long)
                            
                            # Random: Randomly select indices
                            rand_indices = torch.randperm(layer_size, device=self.config.device)[:num_to_drop] if num_to_drop > 0 else torch.tensor([], device=self.config.device, dtype=torch.long)
                            
                            # Add to dropout lists
                            high_rq_dropout.append(high_indices)
                            low_rq_dropout.append(low_indices)
                            random_dropout.append(rand_indices)
                    
                    # Store computed indices for this fraction
                    dropout_indices[frac_idx] = {
                        "high_rq": high_rq_dropout,
                        "low_rq": low_rq_dropout,
                        "random": random_dropout
                    }
                
                # Now process each dropout fraction with the precomputed indices
                with torch.no_grad():
                    # Use multiple batches for robust evaluation
                    num_batches = 5
                    batch_acc_high = torch.zeros(num_drops, device=self.config.device)
                    batch_acc_low = torch.zeros(num_drops, device=self.config.device)
                    batch_acc_rand = torch.zeros(num_drops, device=self.config.device)
                    batch_loss_high = torch.zeros(num_drops, device=self.config.device)
                    batch_loss_low = torch.zeros(num_drops, device=self.config.device)
                    batch_loss_rand = torch.zeros(num_drops, device=self.config.device)
                    
                    # Process batches with a progress bar
                    batches_processed = 0
                    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc=f"Network {i+1}: Processing batches", total=num_batches, leave=False)):
                        if batch_idx >= num_batches:
                            break
                            
                        inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                        
                        # Process all dropout fractions at once
                        for frac_idx in range(num_drops):
                            # Get precomputed dropout indices
                            high_rq_dropout = dropout_indices[frac_idx]["high_rq"]
                            low_rq_dropout = dropout_indices[frac_idx]["low_rq"]
                            random_dropout = dropout_indices[frac_idx]["random"]
                            
                            # Process High RQ strategy (drop highest RQ nodes)
                            outputs, hiddens = network.forward_targeted_dropout(
                                inputs, high_rq_dropout, 
                                list(range(len(network.alignment_layers))), 
                                dropout_mode=dropout_mode
                            )
                            
                            # Calculate accuracy and loss (high RQ)
                            _, predicted = outputs.max(1)
                            batch_acc = (predicted.eq(targets).float().mean() * 100).item()
                            batch_acc_high[frac_idx] += batch_acc
                            batch_loss_high[frac_idx] += (100.0 - batch_acc)  # Use error rate as loss
                            
                            # Store alignment values on first batch
                            if batch_idx == 0 and frac_idx == 0:
                                align_high = self.metric.measure(hiddens, targets, dataset.num_classes)
                                if "high_rq" not in align_values:
                                    align_values["high_rq"] = [None] * num_drops
                                align_values["high_rq"][frac_idx] = align_high
                            
                            # Process Low RQ strategy (drop lowest RQ nodes)
                            outputs, hiddens = network.forward_targeted_dropout(
                                inputs, low_rq_dropout, 
                                list(range(len(network.alignment_layers))), 
                                dropout_mode=dropout_mode
                            )
                            
                            # Calculate accuracy and loss (low RQ)
                            _, predicted = outputs.max(1)
                            batch_acc = (predicted.eq(targets).float().mean() * 100).item()
                            batch_acc_low[frac_idx] += batch_acc
                            batch_loss_low[frac_idx] += (100.0 - batch_acc)
                            
                            # Store alignment values on first batch
                            if batch_idx == 0 and frac_idx == 0:
                                align_low = self.metric.measure(hiddens, targets, dataset.num_classes)
                                if "low_rq" not in align_values:
                                    align_values["low_rq"] = [None] * num_drops
                                align_values["low_rq"][frac_idx] = align_low
                            
                            # Process Random strategy
                            outputs, hiddens = network.forward_targeted_dropout(
                                inputs, random_dropout, 
                                list(range(len(network.alignment_layers))), 
                                dropout_mode=dropout_mode
                            )
                            
                            # Calculate accuracy and loss (random)
                            _, predicted = outputs.max(1)
                            batch_acc = (predicted.eq(targets).float().mean() * 100).item()
                            batch_acc_rand[frac_idx] += batch_acc
                            batch_loss_rand[frac_idx] += (100.0 - batch_acc)
                            
                            # Store alignment values on first batch
                            if batch_idx == 0 and frac_idx == 0:
                                align_rand = self.metric.measure(hiddens, targets, dataset.num_classes)
                                if "random" not in align_values:
                                    align_values["random"] = [None] * num_drops
                                align_values["random"][frac_idx] = align_rand
                        
                        batches_processed += 1
                    
                    # Average results across batches
                    if batches_processed > 0:
                        batch_acc_high /= batches_processed
                        batch_acc_low /= batches_processed
                        batch_acc_rand /= batches_processed
                        batch_loss_high /= batches_processed
                        batch_loss_low /= batches_processed
                        batch_loss_rand /= batches_processed
                    
                    # Store in the result tensors
                    acc_high[i, :] = batch_acc_high
                    acc_low[i, :] = batch_acc_low
                    acc_rand[i, :] = batch_acc_rand
                    loss_high[i, :] = batch_loss_high
                    loss_low[i, :] = batch_loss_low
                    loss_rand[i, :] = batch_loss_rand
                
            except Exception as e:
                logger.error(f"Error processing network {i}: {str(e)}", exc_info=True)
                # Continue with the next network
        
        # Move results to CPU and compute final statistics
        acc_high = acc_high.cpu()
        acc_low = acc_low.cpu()
        acc_rand = acc_rand.cpu()
        loss_high = loss_high.cpu()
        loss_low = loss_low.cpu()
        loss_rand = loss_rand.cpu()
        
        # Compute mean metrics across networks
        mean_acc_high = torch.mean(acc_high, dim=0).tolist()
        mean_acc_low = torch.mean(acc_low, dim=0).tolist()
        mean_acc_rand = torch.mean(acc_rand, dim=0).tolist()
        
        mean_loss_high = torch.mean(loss_high, dim=0).tolist()
        mean_loss_low = torch.mean(loss_low, dim=0).tolist()
        mean_loss_rand = torch.mean(loss_rand, dim=0).tolist()
        
        # Store results
        results["accuracies"]["high_rq"] = mean_acc_high
        results["accuracies"]["low_rq"] = mean_acc_low
        results["accuracies"]["random"] = mean_acc_rand
        
        results["losses"]["high_rq"] = mean_loss_high
        results["losses"]["low_rq"] = mean_loss_low
        results["losses"]["random"] = mean_loss_rand
        
        # Store alignment values
        for strategy in ["high_rq", "low_rq", "random"]:
            if strategy in align_values:
                results["alignment_values"][strategy] = align_values[strategy]
        
        # Restore logging level
        logging.getLogger('alignment').setLevel(original_log_level)
        
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
        
        # Initialize result structure to match the format of _run_progressive_dropout
        results = {
            "dropout_fractions": dropout_fractions.tolist(),
            "accuracies": {
                "eigenvector": []
            },
            "losses": {
                "eigenvector": []
            },
            "alignment_values": {
                "eigenvector": []
            }
        }
        
        # Use tqdm for a nice progress bar
        for dropout_fraction in tqdm(dropout_fractions, desc="Eigenvector Dropout"):
            # Run without verbose logging for each fraction
            accuracy, alignment_values = eigenvector_dropout(
                net,
                self.config.dataset,
                dropout_fraction=dropout_fraction,
                metric=self.metric,
                device=self.device,
                dropout_mode=self.config.extra.dropout_mode,
                dropout_pruning_mode=self.config.extra.dropout_pruning_mode
            )
            
            # Store results in the same format as _run_progressive_dropout
            results["accuracies"]["eigenvector"].append(accuracy)
            results["losses"]["eigenvector"].append(100.0 - accuracy)
            results["alignment_values"]["eigenvector"].append(alignment_values)
            
        return results
    
    def plot(self, results: Dict) -> None:
        """
        Plot experiment results using enhanced visualization methods.
        
        Args:
            results: Dictionary containing experiment results
        """
        logger.info("Generating experiment visualizations")
        
        # Check for results to plot
        if not results:
            logger.warning("No results to plot")
            return
        
        # Generate experiment summary
        try:
            logger.info("Creating experiment summary visualization")
            self._generate_summary_plots(results)
            logger.info("Successfully created experiment summary visualization")
        except Exception as e:
            logger.error(f"Error creating summary visualization: {str(e)}", exc_info=True)
        
        # Generate progressive dropout plots
        if results.get("progressive_dropout"):
            try:
                logger.info("Generating progressive dropout visualization")
                
                # Use our enhanced plotting method
                self._plot_dropout_results(
                    results["progressive_dropout"], 
                    "Progressive Dropout",
                    "progressive_dropout"
                )
                
                # Generate dedicated loss vs dropout plot
                self._generate_dropout_plots(
                    results["progressive_dropout"],
                    "Progressive Dropout",
                    "progressive_dropout"
                )
                
                logger.info("Successfully generated progressive dropout visualizations")
            except Exception as e:
                logger.error(f"Error generating progressive dropout visualizations: {str(e)}", exc_info=True)
        
        # Generate eigenvector dropout plots
        if results.get("eigenvector_dropout"):
            try:
                logger.info("Generating eigenvector dropout visualization")
                
                # Use our enhanced plotting method
                self._plot_dropout_results(
                    results["eigenvector_dropout"], 
                    "Eigenvector Dropout",
                    "eigenvector_dropout"
                )
                
                # Generate dedicated loss vs dropout plot
                self._generate_dropout_plots(
                    results["eigenvector_dropout"],
                    "Eigenvector Dropout",
                    "eigenvector_dropout"
                )
                
                logger.info("Successfully generated eigenvector dropout visualizations")
            except Exception as e:
                logger.error(f"Error generating eigenvector dropout visualizations: {str(e)}", exc_info=True)
        
        # Generate combined visualizations if both types of results exist
        if results.get("progressive_dropout") and results.get("eigenvector_dropout"):
            try:
                logger.info("Generating combined dropout comparison visualization")
                
                # Create a combined visualization comparing both dropout types
                self._generate_dropout_comparison(
                    results["progressive_dropout"],
                    results["eigenvector_dropout"]
                )
                
                logger.info("Successfully generated combined dropout comparison")
            except Exception as e:
                logger.error(f"Error generating dropout comparison: {str(e)}", exc_info=True)
        
        logger.info("Completed experiment visualization generation")
    
    def _plot_dropout_results(self, results, plot_dir, title_prefix="Dropout"):
        """
        Plot dropout experiment results, matching the style of the preref codebase.
        
        Args:
            results: Results dictionary from _run_progressive_dropout
            plot_dir: Directory to save plots to
            title_prefix: Prefix for plot titles
        """
        import os
        import json
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Check if an error occurred during the experiment
        if "error" in results:
            logger.error(f"Cannot plot results due to error: {results['error']}")
            return
            
        # Ensure plot directory exists
        os.makedirs(plot_dir, exist_ok=True)
        
        # Get pruning mode and dropout mode
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global")
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")
        
        # Extract dropout fractions
        dropout_fractions = results.get("dropout_fractions", [])
        if not isinstance(dropout_fractions, (list, np.ndarray)) or (isinstance(dropout_fractions, (list, np.ndarray)) and len(dropout_fractions) == 0):
            logger.error("No dropout fractions found in results")
            return
            
        # Get strategies and prepare colors (match preref style)
        strategies = ["high_rq", "low_rq", "random"]
        colors = {
            "high_rq": "blue",
            "low_rq": "red",
            "random": "green"
        }
        
        # Line styles and markers (match preref style)
        linestyles = {
            "high_rq": "-",
            "low_rq": "-",
            "random": "-"
        }
        markers = {
            "high_rq": "o",
            "low_rq": "s",
            "random": "^"
        }
        strategy_labels = {
            "high_rq": "High RQ",
            "low_rq": "Low RQ",
            "random": "Random"
        }
        
        # ----------------
        # Plot Accuracy
        # ----------------
        plt.figure(figsize=(10, 6))
        
        # Ensure we have some results to plot
        has_data = False
        
        # Plot accuracies for each strategy
        for strategy in strategies:
            accs = results.get("accuracies", {}).get(strategy)
            if not isinstance(accs, (list, np.ndarray)) or len(accs) == 0:
                logger.warning(f"No accuracy data for strategy {strategy}")
                continue
                
            has_data = True
            
            # Plot with preref style
            plt.plot(
                dropout_fractions,
                accs,
                marker=markers.get(strategy, 'o'),
                linestyle=linestyles.get(strategy, '-'),
                color=colors.get(strategy, 'black'),
                linewidth=2,
                markersize=6,
                label=strategy_labels.get(strategy, strategy.replace('_', ' ').title())
            )
        
        if not has_data:
            logger.error("No valid accuracy data to plot")
            return
            
        # Set labels and title (preref style)
        plt.xlabel('Dropout Fraction', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title(f'{title_prefix} - {pruning_mode.replace("_", " ").title()} Pruning ({dropout_mode})', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.ylim([0, 100])  # Fixed y-axis for accuracy
        
        # Save the figure
        figure_path = os.path.join(plot_dir, f"progressive_dropout_{pruning_mode}_accuracy.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved dropout accuracy plot to {figure_path}")
        
        # ----------------
        # Plot Loss
        # ----------------
        plt.figure(figsize=(10, 6))
        
        # Plot losses for each strategy
        has_data = False
        for strategy in strategies:
            losses = results.get("losses", {}).get(strategy)
            if not isinstance(losses, (list, np.ndarray)) or len(losses) == 0:
                logger.warning(f"No loss data for strategy {strategy}")
                continue
                
            has_data = True
            
            # Plot with preref style
            plt.plot(
                dropout_fractions,
                losses,
                marker=markers.get(strategy, 'o'),
                linestyle=linestyles.get(strategy, '-'),
                color=colors.get(strategy, 'black'),
                linewidth=2,
                markersize=6,
                label=strategy_labels.get(strategy, strategy.replace('_', ' ').title())
            )
        
        if not has_data:
            logger.warning("No valid loss data to plot")
        else:
            # Set labels and title (preref style)
            plt.xlabel('Dropout Fraction', fontsize=14)
            plt.ylabel('Loss (%)', fontsize=14)
            plt.title(f'{title_prefix} Loss - {pruning_mode.replace("_", " ").title()} Pruning ({dropout_mode})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # Save the figure
            figure_path = os.path.join(plot_dir, f"progressive_dropout_{pruning_mode}_loss.png")
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved dropout loss plot to {figure_path}")
        
        # Save raw results as JSON for further analysis
        # Convert numpy arrays to lists for JSON serialization
        try:
            # Helper function to safely convert numpy arrays to lists
            def safe_convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.cpu().tolist()
                elif isinstance(obj, dict):
                    return {k: safe_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [safe_convert(item) for item in obj]
                else:
                    return obj
            
            json_results = {
                "dropout_fractions": safe_convert(dropout_fractions),
                "accuracies": safe_convert(results.get("accuracies", {})),
                "losses": safe_convert(results.get("losses", {})),
            }
            
            json_path = os.path.join(plot_dir, f"progressive_dropout_{pruning_mode}_results.json")
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
                
            logger.info(f"Saved raw results to {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON results: {str(e)}")
        
        return figure_path
    
    def _generate_dropout_plots(self, results: Dict, title: str, filename: str) -> None:
        """
        Generate and log enhanced dropout plots for wandb.
        
        Args:
            results: Dropout experiment results
            title: Plot title
            filename: Filename for saving the plot
        """
        try:
            # Use the enhanced plotting method directly
            self._plot_dropout_results(results, self.figure_path, title)
            
            # Also create a simplified loss vs dropout plot
            import matplotlib.pyplot as plt
            import numpy as np
            import wandb
            
            # Get dropout fractions
            dropout_fractions = results.get("dropout_fractions", [])
            
            # Get accuracy values - use eigenvector strategy for eigenvector dropout,
            # high_rq for progressive dropout
            accuracies = []
            if "accuracies" in results:
                if "eigenvector" in results["accuracies"]:
                    # Eigenvector dropout
                    accuracies = results["accuracies"]["eigenvector"]
                elif "high_rq" in results["accuracies"]:
                    # Progressive dropout
                    accuracies = results["accuracies"]["high_rq"]
                    
            # Create figure for loss vs dropout
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.suptitle(f"{title} - Loss", fontsize=16)
            
            # Simulate loss values (decreasing as accuracy increases)
            # In a real scenario, you'd use actual loss values
            loss_values = [100 - acc for acc in accuracies]
            
            # Plot loss vs dropout fraction
            ax.plot(dropout_fractions, loss_values, 'o-', linewidth=2, color='red')
            ax.set_xlabel("Dropout Fraction", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title("Loss vs. Dropout", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([min(dropout_fractions), max(dropout_fractions)])
            
            # Save the figure locally
            if not self.figure_path:
                self.setup_paths()
            
            loss_filepath = os.path.join(self.figure_path, f"{filename}_loss.png")
            plt.savefig(loss_filepath, dpi=300)
            
            # Log to wandb explicitly
            if hasattr(self.config, 'checkpointing') and self.config.checkpointing.use_wandb:
                # Ensure offline mode if running on holylabs
                if str(self.path).startswith("/n/holylabs"):
                    os.environ["WANDB_MODE"] = "offline"
                    
                # Force log as image
                if self.wandb_run:
                    self.wandb_run.log({f"{filename}_loss": wandb.Image(loss_filepath)})
                elif wandb.run:
                    wandb.log({f"{filename}_loss": wandb.Image(loss_filepath)})
            
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating {title} plot: {str(e)}", exc_info=True)

    def _generate_summary_plots(self, results: Dict) -> None:
        """
        Generate comprehensive summary plots for the experiment.
        
        Args:
            results: Experiment results
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import wandb
            
            # Create a more comprehensive summary figure
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
            
            # Panel 1: Experiment Configuration
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.axis('off')
            
            # Get data to display
            config_text = [
                f"Experiment: {self.get_basename()}",
                f"Timestamp: {self.timestamp}",
                f"Model: {self.config.model.model_name}",
                f"Dataset: {self.config.dataset.dataset_name}",
                f"Alignment Metric: {self.config.alignment.metric}",
                f"Dropout Range: {self.config.alignment.dropout_min} to {self.config.alignment.dropout_max}",
                f"Dropout Steps: {self.config.alignment.dropout_steps}",
                f"Dropout Mode: {self.config.extra.dropout_mode}",
                f"Dropout Pruning Mode: {self.config.extra.dropout_pruning_mode}",
                f"Progressive Dropout: {self.config.alignment.run_progressive}",
                f"Eigenvector Dropout: {self.config.alignment.run_eigenvector}"
            ]
            
            # Add training info if available
            if hasattr(self.config, "training"):
                config_text.extend([
                    f"Optimizer: {self.config.training.optimizer}",
                    f"Learning Rate: {self.config.training.learning_rate}",
                    f"Epochs: {self.config.training.epochs}"
                ])
            
            ax1.text(0.05, 0.95, "\n".join(config_text), fontsize=11, 
                    verticalalignment='top', horizontalalignment='left')
            ax1.set_title("Experiment Configuration", fontsize=14)
            
            # Panel 2: Progressive Dropout Results (if available)
            ax2 = fig.add_subplot(gs[0, 1])
            if "progressive_dropout" in results and results["progressive_dropout"]:
                prog_results = results["progressive_dropout"]
                
                # Check if we have the expected data structure
                if "dropout_fractions" in prog_results and "accuracies" in prog_results:
                    fractions = prog_results["dropout_fractions"]
                    
                    # Check if we have high_rq strategy data for plotting
                    if "high_rq" in prog_results["accuracies"]:
                        accuracies = prog_results["accuracies"]["high_rq"]
                        
                        ax2.plot(fractions, accuracies, 'bo-', linewidth=2)
                        ax2.set_xlabel("Dropout Fraction", fontsize=12)
                        ax2.set_ylabel("Accuracy (%)", fontsize=12)
                        ax2.set_title("Progressive Dropout Results (High RQ)", fontsize=14)
                        ax2.grid(True, alpha=0.3)
                        ax2.set_ylim([0, 105])
                        
                        # Add text for numeric results
                        if accuracies:
                            first_acc = accuracies[0]
                            last_acc = accuracies[-1]
                            result_text = f"Accuracy change: {first_acc:.2f}% → {last_acc:.2f}%"
                            ax2.text(0.05, 0.05, result_text, fontsize=11,
                                    transform=ax2.transAxes)
                    else:
                        ax2.text(0.5, 0.5, "No High RQ Accuracy Data", 
                                fontsize=12, ha='center', va='center')
                        ax2.axis('off')
                else:
                    ax2.text(0.5, 0.5, "Incomplete Progressive Dropout Results", 
                            fontsize=12, ha='center', va='center')
                    ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, "No Progressive Dropout Results", 
                        fontsize=12, ha='center', va='center')
                ax2.axis('off')
            
            # Panel 3: Low RQ or Eigenvector Dropout Results (if available)
            ax3 = fig.add_subplot(gs[1, 0])
            
            # First check if progressive dropout with low_rq is available
            if ("progressive_dropout" in results and 
                results["progressive_dropout"] and 
                "accuracies" in results["progressive_dropout"] and
                "dropout_fractions" in results["progressive_dropout"] and
                "low_rq" in results["progressive_dropout"]["accuracies"]):
                
                fractions = results["progressive_dropout"]["dropout_fractions"]
                accuracies = results["progressive_dropout"]["accuracies"]["low_rq"]
                
                ax3.plot(fractions, accuracies, 'ro-', linewidth=2)
                ax3.set_xlabel("Dropout Fraction", fontsize=12)
                ax3.set_ylabel("Accuracy (%)", fontsize=12)
                ax3.set_title("Progressive Dropout Results (Low RQ)", fontsize=14)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([0, 105])
                
                # Add text for numeric results
                if accuracies:
                    first_acc = accuracies[0]
                    last_acc = accuracies[-1]
                    result_text = f"Accuracy change: {first_acc:.2f}% → {last_acc:.2f}%"
                    ax3.text(0.05, 0.05, result_text, fontsize=11,
                            transform=ax3.transAxes)
            # Fallback to eigenvector dropout if available
            elif "eigenvector_dropout" in results and results["eigenvector_dropout"]:
                eig_results = results["eigenvector_dropout"]
                
                if "dropout_fractions" in eig_results and "accuracies" in eig_results and "eigenvector" in eig_results["accuracies"]:
                    fractions = eig_results["dropout_fractions"]
                    accuracies = eig_results["accuracies"]["eigenvector"]
                    
                    ax3.plot(fractions, accuracies, 'go-', linewidth=2)
                    ax3.set_xlabel("Dropout Fraction", fontsize=12)
                    ax3.set_ylabel("Accuracy (%)", fontsize=12)
                    ax3.set_title("Eigenvector Dropout Results", fontsize=14)
                    ax3.grid(True, alpha=0.3)
                    ax3.set_ylim([0, 105])
                    
                    # Add text for numeric results
                    if accuracies:
                        first_acc = accuracies[0]
                        last_acc = accuracies[-1]
                        result_text = f"Accuracy change: {first_acc:.2f}% → {last_acc:.2f}%"
                        ax3.text(0.05, 0.05, result_text, fontsize=11,
                                transform=ax3.transAxes)
                else:
                    ax3.text(0.5, 0.5, "Incomplete Eigenvector Dropout Results", 
                            fontsize=12, ha='center', va='center')
                    ax3.axis('off')
            else:
                ax3.text(0.5, 0.5, "No Low RQ or Eigenvector Dropout Results", 
                        fontsize=12, ha='center', va='center')
                ax3.axis('off')
            
            # Panel 4: Random Strategy or Alignment Values by Layer (if available)
            ax4 = fig.add_subplot(gs[1, 1])
            # Check for random strategy
            if ("progressive_dropout" in results and 
                results["progressive_dropout"] and 
                "accuracies" in results["progressive_dropout"] and
                "dropout_fractions" in results["progressive_dropout"] and
                "random" in results["progressive_dropout"]["accuracies"]):
                
                fractions = results["progressive_dropout"]["dropout_fractions"]
                accuracies = results["progressive_dropout"]["accuracies"]["random"]
                
                ax4.plot(fractions, accuracies, 'go-', linewidth=2)
                ax4.set_xlabel("Dropout Fraction", fontsize=12)
                ax4.set_ylabel("Accuracy (%)", fontsize=12)
                ax4.set_title("Progressive Dropout Results (Random)", fontsize=14)
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim([0, 105])
                
                # Add text for numeric results
                if accuracies:
                    first_acc = accuracies[0]
                    last_acc = accuracies[-1]
                    result_text = f"Accuracy change: {first_acc:.2f}% → {last_acc:.2f}%"
                    ax4.text(0.05, 0.05, result_text, fontsize=11,
                            transform=ax4.transAxes)
            # Fallback to alignment values
            else:
                alignment_data = None
                # Take alignment from first fraction in progressive dropout
                if ("progressive_dropout" in results and 
                    results["progressive_dropout"] and 
                    "alignment_values" in results["progressive_dropout"] and
                    "high_rq" in results["progressive_dropout"]["alignment_values"]):
                    
                    # Use first fraction's alignment value
                    alignment_data = results["progressive_dropout"]["alignment_values"]["high_rq"][0]
                # If no progressive dropout, try eigenvector dropout
                elif ("eigenvector_dropout" in results and 
                      results["eigenvector_dropout"] and 
                      "alignment_values" in results["eigenvector_dropout"] and
                      "eigenvector" in results["eigenvector_dropout"]["alignment_values"]):
                    
                    alignment_data = results["eigenvector_dropout"]["alignment_values"]["eigenvector"][0]
                
                if alignment_data:
                    # Extract alignment values
                    alignment_values = []
                    for i, val in enumerate(alignment_data):
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        alignment_values.append(val)
                    
                    # Create bar chart of alignment by layer
                    x = np.arange(len(alignment_values))
                    bars = ax4.bar(x, alignment_values, width=0.6, alpha=0.7)
                    
                    # Add value labels on top of bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{alignment_values[i]:.3f}',
                                ha='center', va='bottom', rotation=0, fontsize=9)
                    
                    ax4.set_xlabel("Layer", fontsize=12)
                    ax4.set_ylabel("Alignment Value", fontsize=12)
                    ax4.set_title(f"{self.config.alignment.metric} Alignment by Layer", fontsize=14)
                    ax4.set_xticks(x)
                    ax4.set_xticklabels([f"Layer {i+1}" for i in range(len(alignment_values))])
                    ax4.grid(True, alpha=0.3, axis='y')
                else:
                    ax4.text(0.5, 0.5, "No Alignment Data Available", 
                            fontsize=12, ha='center', va='center')
                    ax4.axis('off')
            
            # Add overall title
            fig.suptitle(f"Experiment Summary: {self.get_basename()}", fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            
            # Save the figure
            if not self.figure_path:
                self.setup_paths()
                
            filepath = os.path.join(self.figure_path, "experiment_summary.png")
            plt.savefig(filepath, dpi=300)
            logger.info(f"Saved experiment summary to {filepath}")
            
            # Log to wandb if configured
            if hasattr(self.config, 'checkpointing') and self.config.checkpointing.use_wandb:
                # Get wandb run
                run = self._initialize_wandb()
                
                if run:
                    self.wandb_run.log({"experiment_summary": wandb.Image(filepath)})
                    logger.info(f"Logged experiment summary to wandb")
                elif wandb.run:
                    wandb.log({"experiment_summary": wandb.Image(filepath)})
                    logger.info(f"Logged experiment summary to global wandb run")
            
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating summary visualization: {str(e)}", exc_info=True)

    def _generate_dropout_comparison(self, progressive_results: Dict, eigenvector_results: Dict) -> None:
        """
        Generate and log a comparison between progressive and eigenvector dropout.
        
        Args:
            progressive_results: Progressive dropout experiment results
            eigenvector_results: Eigenvector dropout experiment results
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import wandb
            
            # Get dropout fractions
            dropout_fractions = progressive_results.get("dropout_fractions", [])
            
            # Get progressive dropout accuracies (use high_rq strategy)
            prog_accuracies = progressive_results.get("accuracies", {}).get("high_rq", [])
            
            # Get eigenvector dropout accuracies
            eigenvector_accuracies = eigenvector_results.get("accuracies", {}).get("eigenvector", [])
            
            # Create a comparison figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle("Dropout Comparison: Progressive vs. Eigenvector", fontsize=16)
            
            # Plot 1: Accuracy comparison
            ax1.plot(dropout_fractions, prog_accuracies, 'b-', linewidth=2, label="Progressive")
            ax1.plot(dropout_fractions, eigenvector_accuracies, 'g-', linewidth=2, label="Eigenvector")
            ax1.set_xlabel("Dropout Fraction", fontsize=12)
            ax1.set_ylabel("Accuracy (%)", fontsize=12)
            ax1.set_title("Accuracy vs. Dropout Comparison", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=12)
            
            # Plot 2: Accuracy difference
            accuracy_diff = [e - p for p, e in zip(prog_accuracies, eigenvector_accuracies)]
            ax2.plot(dropout_fractions, accuracy_diff, 'r-', linewidth=2)
            ax2.set_xlabel("Dropout Fraction", fontsize=12)
            ax2.set_ylabel("Accuracy Difference (Eigenvector - Progressive)", fontsize=12)
            ax2.set_title("Accuracy Difference", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add text for maximum difference
            max_diff_idx = np.argmax(np.abs(accuracy_diff))
            max_diff = accuracy_diff[max_diff_idx]
            max_frac = dropout_fractions[max_diff_idx]
            sign = "+" if max_diff > 0 else ""
            ax2.plot(max_frac, max_diff, 'ro', markersize=8)
            ax2.text(max_frac, max_diff + (-5 if max_diff > 0 else 5),
                    f"Max Difference: {sign}{max_diff:.2f}%\nat fraction {max_frac:.2f}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            
            # Save the figure
            if not self.figure_path:
                self.setup_paths()
                
            filepath = os.path.join(self.figure_path, "dropout_comparison.png")
            plt.savefig(filepath, dpi=300)
            
            # Log to wandb if configured
            if hasattr(self.config, 'checkpointing') and self.config.checkpointing.use_wandb:
                run = self._initialize_wandb()
                
                if run:
                    self.wandb_run.log({"dropout_comparison": wandb.Image(filepath)})
                elif wandb.run:
                    wandb.log({"dropout_comparison": wandb.Image(filepath)})
            
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating dropout comparison: {str(e)}")

    def _initialize_wandb(self) -> Optional[Any]:
        """
        Initialize wandb for logging.
        
        Returns:
            Wandb run object or None if wandb is not available or not enabled
        """
        # Check if wandb is enabled in config
        use_wandb = (hasattr(self.config, 'checkpointing') and 
                     hasattr(self.config.checkpointing, 'use_wandb') and 
                     self.config.checkpointing.use_wandb)
        
        if not use_wandb:
            return None
        
        try:
            import wandb
            
            # Ensure wandb is in offline mode for holylabs environment
            if self.path and str(self.path).startswith("/n/holylabs"):
                os.environ["WANDB_MODE"] = "offline"
            
            # Check for existing run
            if hasattr(self, 'wandb_run') and self.wandb_run:
                return self.wandb_run
            elif wandb.run:
                self.wandb_run = wandb.run
                return self.wandb_run
            
            # Create new run if no existing run is found
            project_name = self.get_basename()
            run_name = f"{self.get_basename()}_{self.timestamp}"
            
            # Convert config to dict
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
            
            # Create run with simplified logging
            self.wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config=config_dict
            )
            
            return self.wandb_run
        
        except ImportError:
            logger.warning("Wandb not available (ImportError)")
            return None
        except Exception as e:
            logger.error(f"Error initializing wandb: {str(e)}")
            return None

    def _log_to_wandb(self, data: Dict) -> bool:
        """
        Log data to wandb.
        
        Args:
            data: Dictionary of data to log
            
        Returns:
            True if logging was successful, False otherwise
        """
        # Check if wandb is enabled in config
        use_wandb = (hasattr(self.config, 'checkpointing') and 
                     hasattr(self.config.checkpointing, 'use_wandb') and 
                     self.config.checkpointing.use_wandb)
        
        if not use_wandb:
            logger.debug("Wandb not enabled in config")
            return False
        
        try:
            import wandb
            
            # Initialize wandb if not already done
            if not hasattr(self, 'wandb_run') or not self.wandb_run:
                self._initialize_wandb()
            
            # Log data
            if hasattr(self, 'wandb_run') and self.wandb_run:
                logger.info(f"Logging data to wandb: {list(data.keys())}")
                self.wandb_run.log(data)
                return True
            elif wandb.run:
                logger.info(f"Logging data to global wandb run: {list(data.keys())}")
                wandb.log(data)
                return True
            else:
                logger.warning("No wandb run available for logging")
                return False
        except ImportError:
            logger.warning("Wandb not available (ImportError)")
            return False
        except Exception as e:
            logger.error(f"Error logging to wandb: {str(e)}", exc_info=True)
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
        
        # Ensure all plots are saved and logged
        self.plot(results)
        
        # Save experiment state
        self.save()
        
        # Close wandb run if created
        if hasattr(self, 'wandb_run') and self.wandb_run:
            try:
                import wandb
                self.wandb_run.finish()
            except Exception as e:
                logger.error(f"Error closing wandb run: {str(e)}")
        
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
            logger.info(f"Running progressive dropout experiment (mode: {getattr(self.config.extra, 'dropout_pruning_mode', 'global')})")
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