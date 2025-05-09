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

from alignment.config import ExperimentConfig, DatasetConfig, CheckpointingConfig, AlignmentConfig
from alignment.experiments.experiment import Experiment
from alignment.metrics import AlignmentMetric, get_metric
from alignment.models.registry import create_model
from alignment.dropout import progressive_dropout, eigenvector_dropout, _normalize_device, _ensure_model_on_device, _compute_metric_for_all_nodes, _apply_pruning_to_single_net
from alignment.training import train_networks
from alignment.utils.core import setup_logging
from alignment.utils.plotting import (
    plot_dropout_results, 
    plot_experiment_summary, 
    plot_mean_rq_of_pruned_nodes, 
    plot_per_layer_pruning_percentage,
    plot_per_layer_contribution_to_pruning,
    plot_rq_stats_per_layer
)
from alignment.datasets import get_dataset, load_dataset

logger = logging.getLogger(__name__)


class AlignmentExperiment(Experiment):
    """
    Experiment class for studying neural network alignment properties.
    
    This class implements experiments that assess alignment between layers
    in neural networks, with support for different dropout strategies,
    multiple metrics, and visualization.
    """
    
    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the experiment with the given config.

        Args:
            config: Experiment configuration object (instance of ExperimentConfig).
        """
        super().__init__(config)
        
        # Ensure self.device is normalized at initialization
        if hasattr(self.config, "device") and self.config.device:
            device_str = self.config.device
            if device_str == "cuda": # Normalize "cuda" to "cuda:0"
                device_str = "cuda:0"
            self.device = _normalize_device(torch.device(device_str))
        else:
            self.device = _normalize_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Directly use self.config.debug_mode as it's guaranteed by ExperimentConfig dataclass
        # This makes self.debug_mode an instance attribute of AlignmentExperiment.
        self.debug_mode = self.config.debug_mode 

        current_dataset_name = "unknown"
        if hasattr(self.config, 'dataset'):
            if hasattr(self.config.dataset, 'dataset_name') and self.config.dataset.dataset_name:
                current_dataset_name = self.config.dataset.dataset_name
            elif isinstance(self.config.dataset, str): # Fallback if dataset is just a string name
                current_dataset_name = self.config.dataset
        
        if current_dataset_name == "unknown":
            logger.warning("Dataset name not found. Attempting to use default from DatasetConfig.")
            try:
                default_dataset_conf = DatasetConfig()
                current_dataset_name = default_dataset_conf.dataset_name
                logger.info(f"Using default dataset name: {current_dataset_name}")
            except Exception as e:
                logger.error(f"Could not determine dataset name from defaults: {e}")
                current_dataset_name = "MNIST" # Last resort fallback
                logger.warning(f"Falling back to hardcoded default dataset: {current_dataset_name}")

        logger.info(f"Using dataset: {current_dataset_name} on device: {self.device}")
        
        self.figure_path = None
        self.weights_path = None
        self.setup_paths()
        
        logger.debug(f"Initialized alignment experiment on device {self.device} with debug_mode={self.debug_mode}")
        
        if not hasattr(self.config, "checkpointing"):
            self.config.checkpointing = CheckpointingConfig()
            logger.warning("Checkpointing configuration attribute not found, initializing with defaults.")
        
        if hasattr(self.config, 'alignment') and hasattr(self.config.alignment, 'metric'):
            self.metric = get_metric(self.config.alignment.metric)
        else:
            logger.error("Alignment metric configuration not found! Please check config.alignment.metric.")
            default_alignment_conf = AlignmentConfig()
            self.metric = get_metric(default_alignment_conf.metric)
            logger.warning(f"Falling back to default alignment metric: {default_alignment_conf.metric}")
        
    def get_basename(self) -> str:
        """
        Get the base name for the experiment.
        
        Returns:
            Base name string
        """
        # Ensure dataset_name is correctly accessed
        dataset_name_for_path = "unknown_dataset"
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'dataset_name'):
            dataset_name_for_path = self.config.dataset.dataset_name
        elif hasattr(self.config, 'dataset') and isinstance(self.config.dataset, str):
            dataset_name_for_path = self.config.dataset
        
        return f"alignment_{self.config.model.model_name}_{dataset_name_for_path}"
    
    def prepare_path(self) -> List[str]:
        """
        Prepare the experiment path components.
        
        Returns:
            List of path components
        """
        dataset_name_for_path = "unknown_dataset"
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'dataset_name'):
            dataset_name_for_path = self.config.dataset.dataset_name
        elif hasattr(self.config, 'dataset') and isinstance(self.config.dataset, str):
            dataset_name_for_path = self.config.dataset

        return [
            "alignment",
            self.config.model.model_name,
            dataset_name_for_path,
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
        num_replicates = self.config.training.replicates if hasattr(self.config.training, "replicates") else 5
        networks = []
        for i in range(num_replicates):
            if hasattr(self.config, 'seed') and self.config.seed is not None:
                torch.manual_seed(self.config.seed + i)
                torch.cuda.manual_seed_all(self.config.seed + i)
                np.random.seed(self.config.seed + i)
            
            model = create_model(self.config.model)
            _ensure_model_on_device(model, self.device) # Use the normalized self.device
            networks.append(model)
            
        logger.info(f"Created {len(networks)} models on device {self.device}: {self.config.model.model_name}")
        return networks
    
    def evaluate_on_loader(self, model, data_loader, device=None, show_progress=True):
        eval_device = _normalize_device(device if device is not None else self.device)
        _ensure_model_on_device(model, eval_device)
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Control progress bar display with show_progress argument
        loader_iter = tqdm(data_loader, desc="Evaluating", leave=False) if show_progress else data_loader
        
        with torch.no_grad():
            for inputs, targets in loader_iter:
                inputs, targets = inputs.to(eval_device), targets.to(eval_device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                if show_progress and isinstance(loader_iter, tqdm):
                    loader_iter.set_postfix({
                        'loss': f"{total_loss / total if total > 0 else 0:.4f}",
                        'acc': f"{100.0 * correct / total if total > 0 else 0:.2f}%"
                    })
        
        metrics = {
            'loss': total_loss / total if total > 0 else 0.0,
            'accuracy': 100.0 * correct / total if total > 0 else 0.0
        }
        if show_progress: # Only log final if progress was shown for batches
            logger.info(f"Evaluation complete: Accuracy = {metrics['accuracy']:.2f}%, Loss = {metrics['loss']:.4f}")
        return metrics
        
    def train_networks(self, networks: List[nn.Module], dataset) -> Dict:
        """
        Train multiple networks on the given dataset.
        
        Args:
            networks: List of networks to train
            dataset: Dataset object for training

        Returns:
            Dictionary with training history
        """
        # Get training parameters from config
        num_epochs = getattr(self.config.training, "epochs", 5)
        learning_rate = getattr(self.config.training, "learning_rate", 0.001)
        optimizer_name = getattr(self.config.training, "optimizer", "Adam")
        weight_decay = getattr(self.config.training, "weight_decay", 0.0)
        # Get training_method from config.extra
        training_method = getattr(self.config.extra, "training_method", "auto")
        
        logger.info(f"Training {len(networks)} networks for {num_epochs} epochs using method: {training_method}.")
        
        return train_networks(
            networks=networks,
            dataset=dataset,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=self.device,
            show_progress=True,
            optimizer_class=getattr(torch.optim, optimizer_name, torch.optim.Adam),
            weight_decay=weight_decay,
            training_method=training_method # Pass the training_method to the dispatcher
        )
    
    def run_progressive_dropout(self, networks: List[nn.Module], dataset) -> Dict:
        """
        Run progressive dropout experiment on multiple networks.
        
        Args:
            networks: List of networks to evaluate
            dataset: Dataset object

        Returns:
            Dictionary with dropout experiment results
        """
        # Get dropout parameters from config
        dropout_min = self.config.alignment.dropout_min
        dropout_max = self.config.alignment.dropout_max
        num_dropout_steps = self.config.alignment.dropout_steps
        dropout_fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        
        # Get pruning and dropout modes from config
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")
        
        # IMPORTANT: Add a warning about scaled mode possibly hiding accuracy drops
        if dropout_mode == "scaled":
            logger.warning("Using 'scaled' dropout mode - this scales the remaining weights after pruning, "
                          "which can mask accuracy drops. Consider using 'unscaled' mode if you don't "
                          "see significant accuracy changes with pruning.")
            # Optionally, uncomment this to override the scaled mode for testing
            # dropout_mode = "unscaled"
            # logger.info("Overriding dropout_mode to 'unscaled' for clearer pruning effects")
        
        # DEBUGGING: Check model structures
        for net_idx, network in enumerate(networks[:1]):  # Just check first network
            logger.info(f"Network {net_idx} structure (before training):")
            total_params = 0
            for i, layer in enumerate(network.alignment_layers):
                if hasattr(layer, "weight") and layer.weight is not None:
                    weights = layer.weight.data
                    total_weights = weights.numel()
                    total_params += total_weights
                    
                    # Count zero weights
                    zero_weights = (weights == 0).sum().item()
                    zero_percent = 100.0 * zero_weights / total_weights if total_weights > 0 else 0.0
                    
                    logger.info(f"Layer {i}: Shape {weights.shape}, zeros: {zero_weights}/{total_weights} ({zero_percent:.2f}%)")
            logger.info(f"Total parameters: {total_params}")
        
        # Get multi-strategy option
        use_multi_strategy = getattr(self.config.extra, "use_multi_strategy_dropout", True)
        # Get exclude_classification_layer option
        effective_exclude_classification_layer = getattr(self.config.extra, "exclude_classification_layer", True)

        logger.info(f"Running progressive dropout with pruning_mode={pruning_mode}, dropout_mode={dropout_mode}, exclude_classification_layer={effective_exclude_classification_layer}")
        if use_multi_strategy:
            logger.info("Using optimized multi-strategy approach (processing all strategies simultaneously)")
        
        # Train networks if needed
        if getattr(self.config.training, "train_before_dropout", True):
            training_history = self.train_networks(networks, dataset)
        else:
            # Create empty training history
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': []
            }
        
        # DEBUGGING: Check model structures after training
        for net_idx, network in enumerate(networks[:1]):  # Just check first network
            logger.info(f"Network {net_idx} structure (after training):")
            for i, layer in enumerate(network.alignment_layers):
                if hasattr(layer, "weight") and layer.weight is not None:
                    weights = layer.weight.data
                    total_weights = weights.numel()
                    
                    # Count zero weights
                    zero_weights = (weights == 0).sum().item()
                    zero_percent = 100.0 * zero_weights / total_weights if total_weights > 0 else 0.0
                    
                    logger.info(f"Layer {i}: Shape {weights.shape}, zeros: {zero_weights}/{total_weights} ({zero_percent:.2f}%)")
        
        # Make deep copies of networks to preserve original for each strategy
        networks_copy = []
        for net_idx, net_to_copy in enumerate(networks): # Changed variable name to avoid confusion
            net_copy = copy.deepcopy(net_to_copy)
            _ensure_model_on_device(net_copy, self.device) # Ensure copy is on normalized device
            # Log device of copied network for debugging
            # if self.config.debug_mode:
            #     copied_param_device = next(net_copy.parameters()).device
            #     logger.info(f"Copied network {net_idx} for dropout placed on device: {copied_param_device}")
            networks_copy.append(net_copy)
        
        # Initialize results structure
        results = {
            "dropout_fractions": dropout_fractions,
            "accuracies": {"high_rq": [], "low_rq": [], "random": []},
            "losses": {"high_rq": [], "low_rq": [], "random": []},
            "stds": {"high_rq": [], "low_rq": [], "random": []},
            "training_history": training_history,
            "pruning_details": {},
            "pre_pruning_layer_stats": {} # New key for pre-pruning RQ stats
        }
        
        # DEBUGGING: Create a custom test function to directly verify pruning
        if self.debug_mode: 
            logger.info("DEBUG_MODE: Running detailed test_pruning function.")
            def test_pruning(strategy_name, network_idx=0, fraction_idx=5):
                """Verify pruning is applied correctly for a specific network and fraction"""
                fraction = dropout_fractions[fraction_idx]
                test_net_orig = networks[network_idx] # Original network for comparison of params
                test_net = copy.deepcopy(test_net_orig)
                _ensure_model_on_device(test_net, self.device) # Ensure copy is on normalized device
                
                # Count original non-zero weights
                orig_non_zero = {}
                total_params = 0
                for i, layer in enumerate(test_net.alignment_layers):
                    if hasattr(layer, "weight") and layer.weight is not None:
                        weights = layer.weight.data
                        non_zero = (weights != 0).sum().item()
                        orig_non_zero[i] = non_zero
                        total_params += weights.numel()
                
                logger.info(f"Testing {strategy_name} pruning at {fraction:.2f} fraction:")
                logger.info(f"Original non-zero weights: {sum(orig_non_zero.values())}/{total_params}")
                
                # Call progressive_dropout with debug_mode=True for its internal logging, 
                # but we primarily care about the pruning effect on test_net here.
                if strategy_name == "high_rq":
                    progressive_dropout(
                        [test_net], dataset, [0, fraction], self.metric, self.device,
                        pruning_mode=pruning_mode, dropout_mode=dropout_mode, 
                        strategy="high_rq", show_progress=False, debug_mode=True,
                        exclude_classification_layer_config=effective_exclude_classification_layer
                    )
                elif strategy_name == "low_rq":
                    progressive_dropout(
                        [test_net], dataset, [0, fraction], self.metric, self.device,
                        pruning_mode=pruning_mode, dropout_mode=dropout_mode, 
                        strategy="low_rq", show_progress=False, debug_mode=True,
                        exclude_classification_layer_config=effective_exclude_classification_layer
                    )
                else:  # random
                    progressive_dropout(
                        [test_net], dataset, [0, fraction], self.metric, self.device,
                        pruning_mode=pruning_mode, dropout_mode=dropout_mode, 
                        strategy="random", show_progress=False, debug_mode=True,
                        exclude_classification_layer_config=effective_exclude_classification_layer
                    )
                
                # Check if weights were pruned
                pruned_non_zero = {}
                for i, layer in enumerate(test_net.alignment_layers):
                    if hasattr(layer, "weight") and layer.weight is not None:
                        weights = layer.weight.data
                        non_zero = (weights != 0).sum().item()
                        pruned_non_zero[i] = non_zero
                        reduction = (orig_non_zero[i] - non_zero) / orig_non_zero[i] * 100 if orig_non_zero[i] > 0 else 0
                        logger.info(f"Layer {i}: Non-zero weights reduced from {orig_non_zero[i]} to {non_zero} ({reduction:.2f}% reduction)")
                
                total_before = sum(orig_non_zero.values())
                total_after = sum(pruned_non_zero.values())
                total_reduction = (total_before - total_after) / total_before * 100 if total_before > 0 else 0
                logger.info(f"Total non-zero weights reduced from {total_before} to {total_after} ({total_reduction:.2f}% reduction)")
                
                # Run evaluation to verify weights remain pruned during evaluation
                pre_eval_non_zero = sum((layer.weight.data != 0).sum().item() 
                                       for layer in test_net.alignment_layers 
                                       if hasattr(layer, "weight") and layer.weight is not None)
                
                # Double-check all tensors are on correct device before evaluation
                for name, param in test_net.named_parameters():
                    if param.device != self.device:
                        logger.warning(f"Parameter {name} on wrong device: {param.device} vs {self.device}")
                        param.data = param.data.to(self.device)
                
                # Evaluate the pruned test_net, passing show_progress=False to dataset.evaluate
                # to suppress its batch-level logging during this debug step.
                # The final accuracy/loss from test_pruning will still be logged.
                accuracy, loss = dataset.evaluate(test_net, self.device, show_progress=False)
                
                # Check post-evaluation
                post_eval_non_zero = sum((layer.weight.data != 0).sum().item() 
                                        for layer in test_net.alignment_layers 
                                        if hasattr(layer, "weight") and layer.weight is not None)
                
                logger.info(f"Non-zero weights before evaluation: {pre_eval_non_zero}")
                logger.info(f"Non-zero weights after evaluation: {post_eval_non_zero}")
                logger.info(f"Evaluation accuracy (after {strategy_name} pruning): {accuracy:.2f}%, loss: {loss:.4f}")
                
                return (pre_eval_non_zero == post_eval_non_zero)
            
            for strategy in ["high_rq", "low_rq", "random"]:
                test_pruning(strategy)
        else:
            logger.info("Skipping detailed test_pruning function (debug_mode is False).")
        
        try:
            if use_multi_strategy:
                start_time = time.time()
                try:
                    # progressive_dropout now returns a 4th item: aggregated_pre_pruning_stats
                    network_accuracies, network_losses, pruning_details_from_dropout, pre_pruning_stats = progressive_dropout(
                        networks_copy, 
                        dataset,
                        dropout_fractions,
                        self.metric,
                        self.device,
                        pruning_mode=pruning_mode,
                        dropout_mode=dropout_mode,
                        show_progress=True,
                        use_multi_strategy=True,
                        debug_mode=self.debug_mode,
                        exclude_classification_layer_config=effective_exclude_classification_layer
                    )
                    results["pruning_details"] = pruning_details_from_dropout
                    results["pre_pruning_layer_stats"] = pre_pruning_stats # Store it
                    
                    # CRITICAL FIX: Replace original networks with pruned copies
                    # This ensures any subsequent evaluation uses the pruned networks
                    networks = networks_copy
                    
                    logger.info("Successfully completed progressive_dropout with multi-strategy mode")
                    logger.info("Replaced original networks with pruned copies for accurate evaluation")
                    
                    # Add diagnostics for the returned structure
                    logger.info(f"Return value types: network_accuracies={type(network_accuracies)}, network_losses={type(network_losses)}")
                    
                    # Validate the returned structure
                    if not isinstance(network_accuracies, dict):
                        raise ValueError(f"Expected dictionary for network_accuracies but got {type(network_accuracies)}")
                    if not isinstance(network_losses, dict):
                        raise ValueError(f"Expected dictionary for network_losses but got {type(network_losses)}")
                        
                    # Debug the structure of returned results
                    logger.info(f"Network accuracies keys: {list(network_accuracies.keys())}")
                    
                    strategies = ["high_rq", "low_rq", "random"]
                    for strategy in strategies:
                        if strategy not in network_accuracies:
                            logger.warning(f"Strategy {strategy} not found in network_accuracies")
                            continue
                            
                        if not isinstance(network_accuracies[strategy], dict):
                            logger.warning(f"Expected dict for network_accuracies[{strategy}] but got {type(network_accuracies[strategy])}")
                            continue
                            
                        # Check a sample network's results
                        net_indices = list(network_accuracies[strategy].keys())
                        if not net_indices:
                            logger.warning(f"No network results found for strategy {strategy}")
                            continue
                            
                        sample_net_idx = net_indices[0]
                        sample_values = network_accuracies[strategy][sample_net_idx]
                        
                        if not isinstance(sample_values, list):
                            logger.warning(f"Expected list for network values but got {type(sample_values)}")
                            continue
                            
                        logger.info(f"Strategy {strategy}, sample network {sample_net_idx} has {len(sample_values)} accuracy values")
                        logger.info(f"Sample accuracy values: {sample_values[:3]}...")
                        
                except Exception as e:
                    logger.error(f"Error in progressive_dropout with multi-strategy: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                
                elapsed_time = time.time() - start_time
                logger.info(f"Completed multi-strategy progressive dropout in {elapsed_time:.2f} seconds")
                
                # Process results for each strategy
                for strategy in ["high_rq", "low_rq", "random"]:
                    if strategy not in network_accuracies or strategy not in network_losses:
                        logger.warning(f"Missing results for strategy {strategy}")
                        continue
                        
                    # Extract results for this strategy
                    strategy_accuracies = network_accuracies[strategy]
                    strategy_losses = network_losses[strategy]
                    
                    # Collect results for each fraction
                    fraction_accs = [[] for _ in range(len(dropout_fractions))]
                    fraction_losses = [[] for _ in range(len(dropout_fractions))]
                    
                    # Group results by fraction across all networks
                    for net_idx, acc_list in strategy_accuracies.items():
                        if not isinstance(acc_list, list):
                            logger.warning(f"Expected list for accuracies but got {type(acc_list)} for network {net_idx}")
                            continue
                            
                        for frac_idx, acc in enumerate(acc_list):
                            if frac_idx < len(fraction_accs):
                                fraction_accs[frac_idx].append(acc)
                                
                    for net_idx, loss_list in strategy_losses.items():
                        if not isinstance(loss_list, list):
                            logger.warning(f"Expected list for losses but got {type(loss_list)} for network {net_idx}")
                            continue
                            
                        for frac_idx, loss in enumerate(loss_list):
                            if frac_idx < len(fraction_losses):
                                fraction_losses[frac_idx].append(loss)
                    
                    # Calculate statistics for each fraction
                    for frac_idx in range(len(dropout_fractions)):
                        if fraction_accs[frac_idx]:
                            mean_acc = np.mean(fraction_accs[frac_idx])
                            std_acc = np.std(fraction_accs[frac_idx])
                            mean_loss = np.mean(fraction_losses[frac_idx]) if fraction_losses[frac_idx] else 0.0
                            
                            # Store in results
                            results["accuracies"][strategy].append(mean_acc)
                            results["stds"][strategy].append(std_acc)
                            results["losses"][strategy].append(mean_loss)
                
                    # Log the final result for this strategy
                    last_acc = results["accuracies"][strategy][-1] if results["accuracies"][strategy] else 0
                    logger.info(f"Strategy {strategy}: final accuracy = {last_acc:.2f}%")
                
                # Add timing information
                results["timing"] = {
                    "total_time": elapsed_time
                }
                
                # Explicitly verify pruning mechanism by re-pruning one network to max_fraction for each strategy
                logger.info("Performing explicit verification of pruning mechanism (re-pruning one net to max_fraction)...")
                if networks: # Ensure there's at least one network to test (networks_copy holds the originals for deepcopy)
                    original_network_for_test = networks[0] # Use the actual original network from the list passed to progressive_dropout
                    max_fraction = dropout_fractions[-1] if dropout_fractions else 0.9 # Default to 0.9 if list empty

                    for strategy_to_verify in ["high_rq", "low_rq", "random"]:
                        logger.info(f"Verifying pruning for strategy '{strategy_to_verify}' at {max_fraction*100:.1f}% pruning")
                        
                        net_for_verification = copy.deepcopy(original_network_for_test)
                        _ensure_model_on_device(net_for_verification, self.device)
                        net_for_verification.eval()

                        # Re-compute metrics for this specific clean network copy for verification
                        # This ensures clean scores for the verification independent of prior loops
                        current_scores = _compute_metric_for_all_nodes(net_for_verification, self.metric, self.device, dataset.test_loader, 
                                                                  num_batches=5, 
                                                                  debug_mode=self.debug_mode)
                        current_asc_indices = {l: torch.argsort(s, descending=False) for l,s in current_scores.items()}
                        current_desc_indices = {l: torch.argsort(s, descending=True) for l,s in current_scores.items()}
                        current_rand_indices = {l: torch.randperm(s.shape[0], device=self.device) for l,s in current_scores.items()}

                        _apply_pruning_to_single_net(
                            net_for_verification, max_fraction, strategy_to_verify, 
                            pruning_mode, dropout_mode, self.device,
                            current_scores,
                            current_asc_indices,
                            current_desc_indices,
                            current_rand_indices,
                            self.debug_mode,
                            effective_exclude_classification_layer
                        )
                        
                        total_weights_in_verified_net = 0
                        zero_weights_in_verified_net = 0
                        for i, layer in enumerate(net_for_verification.alignment_layers):
                            if hasattr(layer, "weight") and layer.weight is not None:
                                weights = layer.weight.data
                                layer_total = weights.numel()
                                layer_zeros = (weights == 0).sum().item()
                                total_weights_in_verified_net += layer_total
                                zero_weights_in_verified_net += layer_zeros
                                if self.debug_mode or layer_total > 0: # Log layer details if debug or if layer is not empty
                                    logger.info(f"Strategy '{strategy_to_verify}', Verified Layer {i}: {layer_zeros}/{layer_total} zeros ({100.0*layer_zeros/layer_total if layer_total > 0 else 0:.2f}% pruned)")
                        
                        total_pruned_percent_verified = 100.0*zero_weights_in_verified_net/total_weights_in_verified_net if total_weights_in_verified_net > 0 else 0
                        logger.info(f"Strategy '{strategy_to_verify}', Verified Total: {zero_weights_in_verified_net}/{total_weights_in_verified_net} zeros ({total_pruned_percent_verified:.2f}% pruned)")
                        
                        # Optionally evaluate this specifically pruned network for verification
                        if self.debug_mode: # Only evaluate if in deep debug to save time
                            accuracy, loss = dataset.evaluate(net_for_verification, self.device, show_progress=False) # Suppress progress for this eval
                            logger.info(f"Strategy '{strategy_to_verify}', Verified Max Pruning Eval: Accuracy={accuracy:.2f}%, Loss={loss:.4f}")
            else:
                # Define strategies to run sequentially (original approach)
                strategies = ["high_rq", "low_rq", "random"]
                
                # Create a progress bar for strategies
                strategy_pbar = tqdm(strategies, desc="Pruning strategies", position=0)
                
                # Process each strategy
                for strategy in strategy_pbar:
                    strategy_pbar.set_description(f"Strategy: {strategy}")
                    
                    # Create copies of networks for this strategy
                    strategy_networks = []
                    for net_to_copy_seq in networks: # Use a different loop variable
                        net_copy_seq = copy.deepcopy(net_to_copy_seq)
                        _ensure_model_on_device(net_copy_seq, self.device)
                        strategy_networks.append(net_copy_seq)
                    
                    # progressive_dropout (single-strategy) also needs to return these stats
                    # For now, let's assume it will also return a 4th element, or None if not implemented for single-strategy path yet
                    _accs, _losses, _details, _stats = progressive_dropout(
                        strategy_networks,
                        dataset,
                        dropout_fractions,
                        self.metric,
                        self.device,
                        pruning_mode=pruning_mode,
                        dropout_mode=dropout_mode,
                        strategy=strategy,
                        show_progress=True,
                        use_multi_strategy=False, # Explicitly False for this path
                        debug_mode=self.debug_mode,
                        exclude_classification_layer_config=effective_exclude_classification_layer
                    )
                    # Store these details under the current strategy
                    if "pruning_details" not in results: results["pruning_details"] = {}
                    results["pruning_details"][strategy] = _details

                    # Collect results for each fraction
                    fraction_accs = [[] for _ in range(len(dropout_fractions))]
                    fraction_losses = [[] for _ in range(len(dropout_fractions))]
                    
                    # Group results by fraction across all networks
                    for net_idx in _accs:
                        for frac_idx, acc in enumerate(_accs[net_idx]):
                            if frac_idx < len(fraction_accs):
                                fraction_accs[frac_idx].append(acc)
                        for frac_idx, loss in enumerate(_losses[net_idx]):
                            if frac_idx < len(fraction_losses):
                                fraction_losses[frac_idx].append(loss)
                    
                    # Calculate statistics for each fraction
                    for frac_idx in range(len(dropout_fractions)):
                        if fraction_accs[frac_idx]:
                            mean_acc = np.mean(fraction_accs[frac_idx])
                            std_acc = np.std(fraction_accs[frac_idx])
                            mean_loss = np.mean(fraction_losses[frac_idx]) if fraction_losses[frac_idx] else 0.0
                            
                            # Store in results
                            results["accuracies"][strategy].append(mean_acc)
                            results["stds"][strategy].append(std_acc)
                            results["losses"][strategy].append(mean_loss)
                    
                    # Update progress bar
                    last_acc = results["accuracies"][strategy][-1] if results["accuracies"][strategy] else 0
                    strategy_pbar.set_postfix({"final_acc": f"{last_acc:.2f}%"})

                    # CRITICAL FIX: Replace original networks with the pruned copies
                    # from the final strategy to ensure subsequent evaluation uses pruned networks
                    if strategy == strategies[-1]:  # After the last strategy
                        networks = strategy_networks
                        logger.info("Replaced original networks with pruned copies from final strategy")

                    # If single-strategy path provides these stats
                    if _stats:
                        if "pre_pruning_layer_stats_by_strategy" not in results: results["pre_pruning_layer_stats_by_strategy"] = {}
                        results["pre_pruning_layer_stats_by_strategy"][strategy] = _stats
                    elif "pre_pruning_layer_stats" not in results: # If only multi-strategy provides global stats
                        # This part is tricky if only multi-strategy calculates it globally. 
                        # For simplicity, the new plot will primarily use results["pre_pruning_layer_stats"]
                        # from the multi-strategy path, which averages over all initial networks.
                        pass
        
        except Exception as e:
            logger.error(f"Error running progressive dropout: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results["error"] = str(e)
        
        # Add training history to results (already there)
        # results["training_history"] = training_history # Ensure this is correctly populated
        
        # Existing plots for accuracy/loss
        accuracy_plot_files = plot_dropout_results(
            results, 
            save_dir=self.figure_path,
            title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')} Accuracy",
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode
        )
        if accuracy_plot_files:
            results.setdefault("plot_files", []).extend(accuracy_plot_files)

        # New plot for mean RQ of pruned nodes
        if "pruning_details" in results: # Check if data is available
            mean_rq_plot_file = plot_mean_rq_of_pruned_nodes(
                results, 
                save_dir=self.figure_path,
                title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}",
                show_plots=getattr(self.config, 'show_all', False)
            )
            if mean_rq_plot_file:
                results.setdefault("plot_files", []).append(mean_rq_plot_file)

            # New plots for per-layer pruning percentage
            per_layer_plots = plot_per_layer_pruning_percentage(
                results, 
                save_dir=self.figure_path,
                title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}",
                show_plots=getattr(self.config, 'show_all', False)
            )
            if per_layer_plots:
                results.setdefault("plot_files", []).extend(per_layer_plots)

            # New plot for layer contribution to pruning
            layer_contribution_plot_file = plot_per_layer_contribution_to_pruning(
                results, 
                save_dir=self.figure_path,
                title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}",
                show_plots=getattr(self.config, 'show_all', False)
            )
            if layer_contribution_plot_file:
                results.setdefault("plot_files", []).append(layer_contribution_plot_file)

            # New plot for pre-pruning layer RQ stats
            if "pre_pruning_layer_stats" in results and results["pre_pruning_layer_stats"]:
                rq_stats_plot_file = plot_rq_stats_per_layer(
                    results, 
                    save_dir=self.figure_path,
                    title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}",
                    show_plots=getattr(self.config, 'show_all', False)
                )
                if rq_stats_plot_file:
                    results.setdefault("plot_files", []).append(rq_stats_plot_file)
            else:
                logger.warning("Pre-pruning layer RQ stats not found, skipping RQ stats plot.")
        else:
            logger.warning("Pruning details not found in results, skipping RQ and per-layer pruning plots.")

        # Save results to file (already there)
        # self.save_results("progressive_dropout_results.pkl", results)
        
        return results
    
    def run_eigenvector_dropout(self, network: nn.Module, dataset) -> Dict:
        """
        Run eigenvector dropout experiment on a network.
        
        Args:
            network: Network to evaluate
            dataset: Dataset object

        Returns:
            Dictionary with eigenvector dropout results
        """
        # Get dropout parameters from config
        dropout_min = self.config.alignment.dropout_min
        dropout_max = self.config.alignment.dropout_max
        num_dropout_steps = self.config.alignment.dropout_steps
        dropout_fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        
        # Get dropout mode from config
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
        
        # Initialize results
        results = {
            "dropout_fractions": dropout_fractions,
            "accuracies": {"eigenvector": []},
            "losses": {"eigenvector": []},
            "alignment_values": {"eigenvector": []}
        }
        
        # Ensure network is on the correct, normalized device before passing to eigenvector_dropout
        _ensure_model_on_device(network, self.device)
        
        # Process each dropout fraction
        fraction_pbar = tqdm(dropout_fractions, desc="Eigenvector Dropout", position=0)
        for dropout_fraction in fraction_pbar:
            try:
                # Call the eigenvector_dropout function
                accuracy, alignment_values = eigenvector_dropout(
                    network,
                    self.config.dataset,
                    dropout_fraction=dropout_fraction,
                    metric=self.metric,
                    device=self.device,
                    dropout_mode=dropout_mode,
                    dropout_pruning_mode=pruning_mode
                )
                
                # Store results
                results["accuracies"]["eigenvector"].append(accuracy)
                results["losses"]["eigenvector"].append(100.0 - accuracy)
                results["alignment_values"]["eigenvector"].append(alignment_values)
                
                # Update progress bar
                fraction_pbar.set_postfix({"acc": f"{accuracy:.2f}%"})
                
            except Exception as e:
                logger.error(f"Error in eigenvector dropout at fraction {dropout_fraction}: {str(e)}")
                # Add placeholder values to maintain result structure
                results["accuracies"]["eigenvector"].append(0.0)
                results["losses"]["eigenvector"].append(100.0)
                results["alignment_values"]["eigenvector"].append(None)
        
        return results
    
    def main(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Main experiment execution method.
        
        Returns:
            Tuple of (results dictionary, list of networks)
        """
        # Setup paths for results
        self.setup_paths()
        logger.info(f"Set up paths. Results will be saved to {self.results_path}")

        # Create network models for experiment
        networks = self.create_networks()
        
        # Prepare dataset
        batch_size = getattr(self.config.dataset, "batch_size", 128)
        dataset = load_dataset(self.config.dataset, batch_size=batch_size)
        
        # Run the experiment based on the specified type
        experiment_type = getattr(self.config, 'experiment_type', 'alignment_analysis')
        
        if experiment_type == "alignment_analysis" or experiment_type == "alignment":
            # Run alignment analysis
            results = self.run_alignment_analysis(networks, dataset)
            
        elif experiment_type == "progressive_dropout":
            # Run progressive dropout experiment
            results = self.run_progressive_dropout(networks, dataset)
            
            # Generate plots
            plot_files = plot_dropout_results(
                results, 
                save_dir=self.figure_path,
                title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}",
                pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global_joint"),
                dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled")
            )
            
            # Save the plot files in results
            if plot_files:
                results["plot_files"] = plot_files
            
            # Save results to file
            self.save_results("progressive_dropout_results.pkl", results)
            
        elif experiment_type == "eigenvector_dropout":
            # Run eigenvector dropout experiment on first network
            results = self.run_eigenvector_dropout(networks[0], dataset)
            
            # Generate plots
            plot_files = plot_dropout_results(
                results, 
                save_dir=self.figure_path,
                title_prefix="Eigenvector Dropout",
                pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global_joint"),
                dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled")
            )
            
            # Save the plot files in results
            if plot_files:
                results["plot_files"] = plot_files
            
            # Save results to file
            self.save_results("eigenvector_dropout_results.pkl", results)
            
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
            
        logger.info(f"Completed {getattr(self.config, 'experiment_name', experiment_type)} experiment")
        
        return results, networks
    
    def run_alignment_analysis(self, networks: List[nn.Module], dataset) -> Dict:
        """
        Run a comprehensive alignment analysis with multiple experiment types.
        
        Args:
            networks: List of networks to analyze
            dataset: Dataset object

        Returns:
            Dictionary with all results
        """
        logger.info("Running alignment analysis experiment")
        
        # Prepare results structure
        results = {
            "config": self.config,
        }
        
        # Run progressive dropout if configured
        if self.config.alignment.run_progressive:
            logger.info(f"Running progressive dropout experiment as part of alignment analysis")
            # Results from run_progressive_dropout already include plots and pruning_details
            results["progressive_dropout"] = self.run_progressive_dropout(networks, dataset) 
            # No need to call plot_dropout_results again here for accuracies, it's done inside run_progressive_dropout
            # The new plots are also generated inside run_progressive_dropout
        
        # Run eigenvector dropout if configured
        if self.config.alignment.run_eigenvector:
            logger.info("Running eigenvector dropout experiment")
            results["eigenvector_dropout"] = self.run_eigenvector_dropout(networks[0], dataset)
            
            # Generate plots for eigenvector dropout
            plot_files = plot_dropout_results(
                results["eigenvector_dropout"], 
                save_dir=self.figure_path,
                title_prefix="Eigenvector Dropout",
                pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global_joint"),
                dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled")
            )
            
            # Add plot files to results
            if plot_files:
                results["eigenvector_dropout"]["plot_files"] = plot_files
        
        # Generate summary plots if both experiment types were run
        if "progressive_dropout" in results and "eigenvector_dropout" in results:
            try:
                summary_path = plot_experiment_summary(
                    results, 
                    self.figure_path,
                    experiment_name=getattr(self.config, "experiment_name", "Alignment Analysis")
                )
                results["summary_plot"] = summary_path
            except Exception as e:
                logger.error(f"Error generating summary plot: {str(e)}")
        
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
        
        # The self.device is already normalized in __init__
        logger.info(f"Set up paths: results={self.results_path}, figures={self.figure_path}, device={self.device}")
    
    def save_results(self, filename: str, results: Dict):
        """
        Save results to file.
        
        Args:
            filename: Name of the file to save to
            results: Results dictionary to save
        """
        # Create a results directory if it doesn't exist
        os.makedirs(self.results_path, exist_ok=True)
        
        # Save as pickle
        results_file = os.path.join(self.results_path, filename)
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved experiment results to {results_file}")
        
        # Try to save as JSON for readability
        try:
            # Helper to convert non-serializable objects
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
            
            # Convert to JSON-serializable format
            json_results = clean_for_json(results)
            
            # Save as JSON
            json_file = os.path.join(self.results_path, filename.replace(".pkl", ".json"))
            with open(json_file, "w") as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved readable results to {json_file}")
        except Exception as e:
            logger.warning(f"Could not save results as JSON: {str(e)}")
    
    def run(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Run the experiment.
        
        Returns:
            Tuple of (results, networks)
        """
        # Set up logging
        setup_logging(log_level="INFO")
        
        # Set random seed if configured
        if hasattr(self.config, 'seed') and self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info(f"Set random seed to {self.config.seed}")
        
        # Run the main experiment
        results, networks = self.main()
        
        # Store results and networks for later use
        self.results = results
        self.networks = networks
        
        # Save configuration
        config_file = os.path.join(self.results_path, "config.yaml")
        if hasattr(self.config, 'to_dict'):
            config_dict = self.config.to_dict()
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Saved configuration to {config_file}")
        
        return results, networks


def cli_main():
    """Command-line interface for running alignment experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural network alignment experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = parser.parse_args()

    # Set logging level
    log_level = "WARNING" if args.quiet else "INFO"
    setup_logging(log_level=log_level)

    # Load configuration
    config = ExperimentConfig.load(args.config)
    
    # Initialize and run experiment
    experiment = AlignmentExperiment(config)
    results, networks = experiment.run()
    
    return results, networks


if __name__ == "__main__":
    cli_main()