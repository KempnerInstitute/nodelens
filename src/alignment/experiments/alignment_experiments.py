"""
Alignment experiment implementations.

This module contains experiment classes for neural network alignment studies,
focusing on alignment metrics, dropout impacts, and training analysis.
"""

import logging
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
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
import wandb

from alignment.config import ExperimentConfig, DatasetConfig, CheckpointingConfig, AlignmentConfig
from alignment.experiments.experiment import Experiment
from alignment.metrics import AlignmentMetric, get_metric, compute_all_node_scores
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
    plot_rq_stats_per_layer,
    plot_layer_isolated_dropout_results,
    log_plots_to_wandb
)
from alignment.datasets import get_dataset, load_dataset
from alignment.dropout_manager import run_layer_isolated_dropout_experiment, run_progressive_dropout_experiment

# Import for callbacks
from torch.utils.data import DataLoader # Ensure DataLoader is imported
from alignment.callbacks import AlignmentMetricTracker

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
        # Call Experiment.__init__ which now handles config loading, paths, seeds, and W&B init
        super().__init__(config)
        
        # Ensure self.device is initialized for AlignmentExperiment
        if hasattr(self.config, "device") and self.config.device:
            device_str = self.config.device
            if device_str == "cuda": device_str = "cuda:0"
            self.device = _normalize_device(torch.device(device_str))
        else:
            self.device = _normalize_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Explicitly set self.debug_mode for the AlignmentExperiment instance
        # self.config is an ExperimentConfig instance and is guaranteed to have debug_mode
        self.debug_mode = self.config.debug_mode 

        # Dataset name determination logic
        current_dataset_name = "unknown"
        if hasattr(self.config, 'dataset'):
            if hasattr(self.config.dataset, 'dataset_name') and self.config.dataset.dataset_name:
                current_dataset_name = self.config.dataset.dataset_name
        elif isinstance(self.config.dataset, str):
                current_dataset_name = self.config.dataset
        
        if current_dataset_name == "unknown":
            logger.warning("Dataset name not found. Attempting to use default from DatasetConfig.")
            try:
                default_dataset_conf = DatasetConfig()
                current_dataset_name = default_dataset_conf.dataset_name
                logger.info(f"Using default dataset name: {current_dataset_name}")
            except Exception as e:
                logger.error(f"Could not determine dataset name from defaults: {e}")
                current_dataset_name = "MNIST" 
                logger.warning(f"Falling back to hardcoded default dataset: {current_dataset_name}")

        logger.info(f"AlignmentExperiment: Using dataset: {current_dataset_name} on device: {self.device}")

        # self.figure_path and self.weights_path are set by base Experiment.setup_paths via self.working_dir
        # (setup_paths is called by super().__init__())
        
        # self.wandb_run is initialized by super().__init__() calling _setup_wandb()
        logger.debug(f"AlignmentExperiment initialized. Debug mode: {self.debug_mode}. W&B run ID: {self.wandb_run.id if self.wandb_run else 'None'}")
        
        # Metric initialization
        if hasattr(self.config, 'alignment_settings') and self.config.alignment_settings is not None:
            self.metric = get_metric(self.config.alignment_settings.metric)
        else:
            logger.error("alignment_settings not found in config or metric not specified!")
            default_align_conf = AlignmentConfig()
            self.metric = get_metric(default_align_conf.metric)
            logger.warning(f"Falling back to default alignment metric: {default_align_conf.metric}")
        
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
            f"metric_{self.config.alignment_settings.metric}"
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
        
    def train_networks(self, networks: List[nn.Module], dataset, callbacks: Optional[List[Callable]] = None) -> Dict:
        """
        Train multiple networks on the given dataset.
        
        Args:
            networks: List of networks to train
            dataset: Dataset object for training
            callbacks: Optional list of callback functions for training.

        Returns:
            Dictionary with training history
        """
        # Get training parameters from config
        num_epochs = getattr(self.config.training, "epochs", 5)
        learning_rate = getattr(self.config.training, "learning_rate", 0.001)
        optimizer_name = getattr(self.config.training, "optimizer", "Adam")
        weight_decay = getattr(self.config.training, "weight_decay", 0.0)
        training_method = getattr(self.config.training, "training_method", "auto")
        
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
            training_method=training_method,
            callbacks=callbacks # Pass callbacks down
            )
    
    def run_progressive_dropout(self, networks: List[nn.Module], dataset) -> Dict:
        logger.info("Running Progressive Dropout Experiment (via DropoutManager)")
        
        # --- Parameter Fetching & Debug Logging --- 
        if not hasattr(self.config, 'pruning_settings'):
            logger.error("CRITICAL: self.config is missing 'pruning_settings' attribute!")
            # Fallback or raise error - for now, try to use alignment as a fallback for old structure
            if hasattr(self.config, 'alignment'):
                logger.warning("Falling back to self.config.alignment for pruning parameters.")
                pruning_config_source = self.config.alignment
            else:
                logger.error("CRITICAL: Cannot find pruning parameters in self.config.alignment either!")
                raise AttributeError("Missing pruning_settings and alignment in config for progressive dropout.")
        else:
            pruning_config_source = self.config.pruning_settings

        if not hasattr(self.config, 'alignment_settings'):
            logger.error("CRITICAL: self.config is missing 'alignment_settings' attribute!")
            if hasattr(self.config, 'alignment'): # Fallback check if old name still exists
                logger.warning("Falling back to self.config.alignment for alignment metric parameters.")
                alignment_config_source = self.config.alignment
            else:
                logger.error("CRITICAL: Cannot find alignment_settings or alignment in config.")
                raise AttributeError("Missing alignment_settings and alignment in config for progressive dropout.")
        else:
            alignment_config_source = self.config.alignment_settings

        dropout_min = getattr(pruning_config_source, "dropout_min", 0.0)
        dropout_max = getattr(pruning_config_source, "dropout_max", 0.9)
        num_dropout_steps = getattr(pruning_config_source, "dropout_steps", 10)
        
        logger.info(f"Fetched pruning params: min={dropout_min}, max={dropout_max}, steps={num_dropout_steps} from {type(pruning_config_source)}")

        # ... (dropout_fractions calculation as before) ...
        if num_dropout_steps <=0 : 
            _fractions = [0.0, dropout_max] 
        elif num_dropout_steps == 1:
             _fractions = [dropout_min, dropout_max]
        else:
            _fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in _fractions: 
            _fractions = [0.0] + _fractions 
        dropout_fractions = sorted(list(set(_fractions)))

        pruning_mode = getattr(pruning_config_source, "dropout_pruning_mode", "global_joint")
        dropout_mode = getattr(pruning_config_source, "dropout_mode", "scaled")
        effective_exclude_classification_layer = getattr(pruning_config_source, "exclude_classification_layer", True)
        
        # self.metric should be set in __init__ from alignment_settings.metric
        metric_to_use = self.metric 
        if metric_to_use is None: # Fallback if self.metric wasn't initialized
            logger.warning("self.metric not initialized, attempting to get from config.alignment_settings now.")
            metric_to_use = get_metric(getattr(alignment_config_source, "metric", "RQ"))

        logger.info(f"Using metric: {getattr(alignment_config_source, 'metric', 'RQ')}, PruningMode: {pruning_mode}, DropoutMode: {dropout_mode}, ExcludeCls: {effective_exclude_classification_layer}")

        training_history = None # Initialize
        if hasattr(self.config, 'training') and getattr(self.config.training, "epochs", 0) > 0 and \
           getattr(self.config.training, "train_before_dropout", True):
            logger.info(f"Starting training for {self.config.training.epochs} epochs...")
            training_history = self.train_networks(networks, dataset)
            if self.debug_mode and training_history:
                logger.info(f"Training completed. History keys: {training_history.keys()}")
                if 'test_acc' in training_history and training_history['test_acc']:
                    logger.info(f"Final test accuracy after training: {training_history['test_acc'][-1]:.2f}%")
            elif not training_history:
                logger.warning("self.train_networks was called but returned no history.")
        else:
            logger.info("Skipping training before dropout based on config (epochs=0 or train_before_dropout=false).")
            training_history = {
                'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []
            }
        
        results_from_manager = {}
        try:
            results_from_manager = run_progressive_dropout_experiment(
                networks, dataset, dropout_fractions, self.metric, self.device,
                    pruning_mode=pruning_mode, dropout_mode=dropout_mode, 
                show_progress=True, debug_mode=self.debug_mode,
                exclude_classification_layer_config=effective_exclude_classification_layer
            )
        except Exception as e:
            logger.error(f"Error during run_progressive_dropout_experiment call: {str(e)}")
            results_from_manager = {"error": str(e)}
            if self.debug_mode:
                import traceback
                logger.error(traceback.format_exc())
        
        # Combine with training history and return for main to handle plotting/saving
        final_results = results_from_manager
        final_results["training_history"] = training_history
        return final_results
    
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
        dropout_min = self.config.pruning_settings.dropout_min
        dropout_max = self.config.pruning_settings.dropout_max
        num_dropout_steps = self.config.pruning_settings.dropout_steps
        dropout_fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        
        # Get dropout mode from config
        dropout_mode = getattr(self.config.pruning_settings, "dropout_mode", "scaled")
        pruning_mode_for_eigen = getattr(self.config.pruning_settings, "dropout_pruning_mode", "global_joint")
        
        # Initialize results
        results_from_manager = {
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
                    dropout_pruning_mode=pruning_mode_for_eigen
                )
                
                # Store results
                results_from_manager["accuracies"]["eigenvector"].append(accuracy)
                results_from_manager["losses"]["eigenvector"].append(100.0 - accuracy)
                results_from_manager["alignment_values"]["eigenvector"].append(alignment_values)
                
                # Update progress bar
                fraction_pbar.set_postfix({"acc": f"{accuracy:.2f}%"})
                
            except Exception as e:
                logger.error(f"Error in eigenvector dropout at fraction {dropout_fraction}: {str(e)}")
                # Add placeholder values to maintain result structure
                results_from_manager["accuracies"]["eigenvector"].append(0.0)
                results_from_manager["losses"]["eigenvector"].append(100.0)
                results_from_manager["alignment_values"]["eigenvector"].append(None)
        
        return results_from_manager
    
    def run_layer_isolated_experiment(self, networks: List[nn.Module], dataset) -> Dict:
        logger.info("Running Layer Isolated Dropout Experiment")
        dropout_min = self.config.pruning_settings.dropout_min
        dropout_max = self.config.pruning_settings.dropout_max
        num_dropout_steps = self.config.pruning_settings.dropout_steps
        if num_dropout_steps <=1:
            _fractions = [0.0, dropout_max]
        else:
            _fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in _fractions: # Ensure 0.0 is present for baseline
            _fractions = [0.0] + _fractions
        dropout_fractions = sorted(list(set(_fractions)))

        exclude_cls_layer = getattr(self.config.pruning_settings, "exclude_classification_layer", True)
        dropout_mode = getattr(self.config.pruning_settings, "dropout_mode", "scaled")
        metric_instance_for_isolated = self.metric # from self.config.alignment_settings

        results_from_manager = run_layer_isolated_dropout_experiment(
            original_networks=networks,
            dataset=dataset,
            dropout_fractions=dropout_fractions,
            metric=metric_instance_for_isolated,
            device=self.device,
            dropout_mode=dropout_mode,
            show_progress=True, 
            debug_mode=self.debug_mode,
            exclude_classification_layer_config=exclude_cls_layer
        )

        # Call the specific plotting function for layer-isolated results
        if results_from_manager: # Check if results were successfully generated
            isolated_plot_files = plot_layer_isolated_dropout_results(
                results_from_manager, 
                save_dir=self.figure_path,
                title_prefix=f"{getattr(self.config, 'experiment_name', 'Layer Isolated Pruning')}",
                show_plots=getattr(self.config, 'show_all', False)
            )
            if isolated_plot_files:
                results_from_manager.setdefault("plot_files", []).extend(isolated_plot_files)
        
        # Potentially add training_history if relevant
        final_results = results_from_manager
        # If train_networks was called before this for the 'networks' list, get its history
        training_history_isolated = getattr(self, 'training_history', {})
        final_results["training_history"] = training_history_isolated
        return final_results
    
    def main(self) -> Tuple[Dict, List[nn.Module]]:
        self.setup_paths()
        logger.info(f"Set up paths. Results will be saved to {self.results_path}")
        networks = self.create_networks()
        batch_size = getattr(self.config.dataset, "batch_size", 128)
        dataset = load_dataset(self.config.dataset, batch_size=batch_size, device=self.device) # Pass device
        
        experiment_type = getattr(self.config, 'experiment_type', 'alignment_analysis')
        results = {} # Initialize results here

        # This training_history will be for the initial training of replicates
        # Initialize callbacks_list for the main training call
        callbacks_list = []
        self.metric_trackers_history = [] # Store tracker instances to retrieve history later

        if hasattr(self.config, 'alignment_settings') and \
           hasattr(self.config.alignment_settings, 'callbacks') and \
           hasattr(self.config.alignment_settings.callbacks, 'alignment_metrics') and \
           self.config.alignment_settings.callbacks.alignment_metrics:
            
            alignment_metrics_to_track = self.config.alignment_settings.callbacks.alignment_metrics
            logger.info(f"Configuring AlignmentMetricTracker callbacks for: {alignment_metrics_to_track}")

            # Ensure dataset has a test_loader for metrics, or use train_loader as fallback
            # Ideally, a dedicated, non-shuffled, non-augmented loader should be used for metric consistency.
            metric_dataloader = dataset.test_loader 
            if metric_dataloader is None and dataset.train_loader is not None:
                logger.warning("No test_loader in dataset for metric callback, using train_loader. This might be slow or inconsistent.")
                metric_dataloader = dataset.train_loader
            elif metric_dataloader is None:
                logger.error("No suitable dataloader (test_loader or train_loader) found in dataset for metric callback. Skipping tracker setup.")
                alignment_metrics_to_track = [] # Clear it so we don't try to use it

            if metric_dataloader: # Only proceed if we have a dataloader
                for metric_config in alignment_metrics_to_track:
                    metric_name = metric_config.get("name")
                    if not metric_name:
                        logger.warning("Metric config in callback missing 'name'. Skipping this tracker.")
                        continue
                    
                    num_batches_for_metric = metric_config.get("num_batches", 5)
                    
                    tracker = AlignmentMetricTracker(
                        metric_name=metric_name,
                        data_loader=metric_dataloader, 
                        device=self.device,
                        num_batches=num_batches_for_metric,
                        experiment_config=self.config # Pass the main experiment config for debug_mode etc.
                    )
                    callbacks_list.append(tracker)
                    self.metric_trackers_history.append(tracker) # Keep instance to get history later
                    logger.info(f"Added AlignmentMetricTracker for '{metric_name}'.")

        if hasattr(self.config, 'training') and getattr(self.config.training, "epochs", 0) > 0:
            logger.info(f"Starting initial training for {self.config.training.epochs} epochs before experiment type: {experiment_type}")
            # Pass the configured callbacks_list to train_networks
            self.training_history = self.train_networks(networks, dataset, callbacks=callbacks_list) 
            if self.debug_mode and self.training_history and self.training_history.get('test_acc'):
                logger.info(f"Initial training final test accuracy: {self.training_history['test_acc'][-1]:.2f}%")
        else:
            logger.info("Skipping initial training based on config.")
            self.training_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        if experiment_type == "alignment_analysis":
            results["alignment_analysis"] = True # Mark that this meta-experiment ran
            if self.config.alignment_settings.run_progressive:
                logger.info(f"Running progressive dropout as part of alignment analysis")
                results["progressive_dropout"] = self.run_progressive_dropout(networks, dataset)
            if self.config.alignment_settings.run_eigenvector:
                logger.info("Running eigenvector dropout as part of alignment analysis")
                # Eigenvector typically runs on one network
                results["eigenvector_dropout"] = self.run_eigenvector_dropout(networks[0], dataset) 
        elif experiment_type == "progressive_dropout":
            results = self.run_progressive_dropout(networks, dataset)
        elif experiment_type == "eigenvector_dropout":
            results = self.run_eigenvector_dropout(networks[0], dataset)
        elif experiment_type == "layer_isolated_pruning":
            results = self.run_layer_isolated_experiment(networks, dataset)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
            
        logger.info(f"Completed main computation for {experiment_type} experiment")

        # --- Centralized Plotting and Saving (after main computation) --- 
        plot_files_generated = []
        if results and "error" not in results:
            # Determine which plots to generate based on available data in results
            # For progressive_dropout or if it's a sub-result of alignment_analysis
            prog_results_data = results if experiment_type == "progressive_dropout" else results.get("progressive_dropout")
            if prog_results_data and "accuracies" in prog_results_data: # Basic check for progressive results
                pruning_mode_plot = getattr(self.config.pruning_settings, "dropout_pruning_mode", "global_joint")
                dropout_mode_plot = getattr(self.config.pruning_settings, "dropout_mode", "scaled")
                title_prefix_prog = f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}"

                acc_plots = plot_dropout_results(prog_results_data, self.figure_path, title_prefix_prog, pruning_mode_plot, dropout_mode_plot)
                if acc_plots: plot_files_generated.extend(acc_plots if isinstance(acc_plots, list) else [acc_plots])
                
                if "pruning_details" in prog_results_data and "pre_pruning_layer_stats" in prog_results_data:
                    plot_paths = [
                        plot_mean_rq_of_pruned_nodes(prog_results_data, self.figure_path, title_prefix_prog, getattr(self.config, 'show_all', False)),
                        plot_per_layer_pruning_percentage(prog_results_data, self.figure_path, title_prefix_prog, getattr(self.config, 'show_all', False)),
                        plot_per_layer_contribution_to_pruning(prog_results_data, self.figure_path, title_prefix_prog, getattr(self.config, 'show_all', False))
                    ]
                    plot_files_generated.extend([p for p in plot_paths if p])
                
                if prog_results_data.get("pre_pruning_layer_stats"):
                    rq_stats_plot = plot_rq_stats_per_layer(prog_results_data, self.figure_path, title_prefix_prog, getattr(self.config, 'show_all', False))
                    if rq_stats_plot: plot_files_generated.append(rq_stats_plot)

            # For layer_isolated_pruning
            iso_results_data = results if experiment_type == "layer_isolated_pruning" else results.get("layer_isolated_pruning") # Assuming results key matches exp_type
            if iso_results_data and "accuracies_isolated" in iso_results_data:
                title_prefix_iso = f"{getattr(self.config, 'experiment_name', 'Layer Isolated')}"
                iso_plots = plot_layer_isolated_dropout_results(iso_results_data, self.figure_path, title_prefix_iso, getattr(self.config, 'show_all', False))
                if iso_plots: plot_files_generated.extend(iso_plots)

            # For eigenvector_dropout (assuming its results structure is similar to progressive for plot_dropout_results)
            eig_results_data = results if experiment_type == "eigenvector_dropout" else results.get("eigenvector_dropout")
            if eig_results_data and "accuracies" in eig_results_data:
                pruning_mode_plot_eig = getattr(self.config.pruning_settings, "dropout_pruning_mode", "global_joint") 
                dropout_mode_plot_eig = getattr(self.config.pruning_settings, "dropout_mode", "scaled") 
                title_prefix_eig = f"{getattr(self.config, 'experiment_name', 'Eigenvector Dropout')}"
                eig_acc_plots = plot_dropout_results(eig_results_data, self.figure_path, title_prefix_eig, pruning_mode_plot_eig, dropout_mode_plot_eig)
                if eig_acc_plots: plot_files_generated.extend(eig_acc_plots if isinstance(eig_acc_plots, list) else [eig_acc_plots])
            
            # Summary plot if multiple types of results are present (e.g., from alignment_analysis)
            if experiment_type == "alignment_analysis" and "progressive_dropout" in results and "eigenvector_dropout" in results:
                summary_path = plot_experiment_summary(results, self.figure_path, getattr(self.config, "experiment_name", "Alignment Analysis"))
                if summary_path: plot_files_generated.append(summary_path)

            if plot_files_generated:
                results["plot_files"] = plot_files_generated # Store all generated plot files
                if self.wandb_run:
                    log_plots_to_wandb(plot_files_generated, tags={"experiment_type": experiment_type})
            
            # Save final comprehensive results dict once
            self.save_results(f"{experiment_type}_main_results.pkl", results)

        logger.info(f"Completed {self.config.experiment_name} (type: {experiment_type}) experiment run.")
        # After run, retrieve and store metric tracker histories
        if hasattr(self, 'metric_trackers_history') and self.metric_trackers_history:
            for tracker in self.metric_trackers_history:
                results[f"{tracker.metric_name}_evolution"] = tracker.metrics_history
            logger.info(f"Stored metric evolution data in results for: {[t.metric_name for t in self.metric_trackers_history]}")
            # Re-save results if metric history was added AFTER initial save
            # This depends on whether save_results is called multiple times or only at the very end.
            # Based on current structure, save_results is called once after plotting. So this new data should be included.
            # If results were saved earlier, need to call self.save_results again here.
            self.save_results(f"{experiment_type}_main_results_with_evolution.pkl", results) # Save again with new data

        return results, networks
    
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
        
        # Finish W&B run if it was initialized
        if self.wandb_run:
            wandb.finish()
            logger.info("Weights & Biases run finished.")
        
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