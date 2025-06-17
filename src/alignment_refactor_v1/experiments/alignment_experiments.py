"""
Alignment experiment implementations.

This module contains experiment classes for neural network alignment studies,
focusing on alignment metrics, dropout impacts, and training analysis.
"""

import logging
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type
import copy
import pickle
import datetime
import json
import yaml
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from alignment_refac1.config import ExperimentConfig, DatasetConfig, CheckpointingConfig, AlignmentConfig, MetricTrackerConfig
from alignment_refac1.experiments.experiment import Experiment
from alignment_refac1.metrics import AlignmentMetric, get_metric, compute_all_node_scores
from alignment_refac1.models.registry import create_model
from alignment_refac1.dropout import progressive_dropout, eigenvector_dropout, _normalize_device, _ensure_model_on_device, _apply_pruning_to_single_net
from alignment_refac1.training import train_networks
from alignment_refac1.utils.core import setup_logging
from alignment_refac1.utils.plotting import (
    plot_dropout_results,
    plot_experiment_summary,
    plot_mean_score_of_pruned_nodes,
    plot_per_layer_pruning_percentage,
    plot_per_layer_contribution_to_pruning,
    plot_score_stats_per_layer,
    plot_layer_isolated_dropout_results,
    log_plots_to_wandb,
    plot_metric_evolution,
)
from alignment_refac1.datasets import get_dataset, load_dataset
from alignment_refac1.dropout_manager import run_layer_isolated_dropout_experiment, run_progressive_dropout_experiment, run_eigenvector_dropout_experiment, run_cascading_layer_pruning_experiment

# Import for callbacks
from torch.utils.data import DataLoader  # Ensure DataLoader is imported
from alignment_refac1.callbacks import AlignmentMetricTracker

logger = logging.getLogger(__name__)


class AlignmentExperiment(Experiment):
    """
    Base experiment class for studying neural network alignment properties.

    Handles common setup like network creation, dataset loading, metric initialization,
    and plotting/saving results. Subclasses implement specific experiment logic in the `run` method.
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
            if device_str == "cuda":
                device_str = "cuda:0"
            self.device = _normalize_device(torch.device(device_str))
        else:
            self.device = _normalize_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Explicitly set self.debug_mode for the AlignmentExperiment instance
        self.debug_mode = self.config.debug_mode

        # Dataset name determination logic
        self._resolve_dataset_name()

        logger.info(f"AlignmentExperiment: Using dataset: {self.current_dataset_name} on device: {self.device}")

        # self.figure_path and self.weights_path are set by base Experiment.setup_paths via self.working_dir
        # (setup_paths is called by super().__init__())

        # self.wandb_run is initialized by super().__init__() calling _setup_wandb()
        logger.debug(f"AlignmentExperiment initialized. Debug mode: {self.debug_mode}. W&B run ID: {self.wandb_run.id if self.wandb_run else 'None'}")

        # Metric initialization
        self._initialize_metric()

        # --- NEW: Initialize Loss Criterion ---
        self._initialize_loss_criterion()
        # --- End NEW ---

        # Initialization for attributes used later
        self.networks: List[nn.Module] = []
        self.dataset = None
        self.training_history: Dict = {}
        self.metric_tracker_instance: Optional[AlignmentMetricTracker] = None
        self.results: Dict = {}  # Initialize results dict

    def _resolve_dataset_name(self):
        """Helper to determine the dataset name from config."""
        self.current_dataset_name = "unknown"
        if hasattr(self.config, "dataset"):
            if hasattr(self.config.dataset, "dataset_name") and self.config.dataset.dataset_name:
                self.current_dataset_name = self.config.dataset.dataset_name
            elif isinstance(self.config.dataset, str):
                self.current_dataset_name = self.config.dataset

        if self.current_dataset_name == "unknown":
            logger.warning("Dataset name not found in config. Attempting to use default from DatasetConfig.")
            try:
                default_dataset_conf = DatasetConfig()
                self.current_dataset_name = default_dataset_conf.dataset_name
                logger.info(f"Using default dataset name: {self.current_dataset_name}")
            except Exception as e:
                logger.error(f"Could not determine dataset name from defaults: {e}")
                self.current_dataset_name = "MNIST"
                logger.warning(f"Falling back to hardcoded default dataset: {self.current_dataset_name}")

    def _initialize_metric(self):
        """Helper to initialize the primary alignment metric and list of metric instances."""
        self.primary_metric: Optional[AlignmentMetric] = None
        self.pruning_metric_instances: List[AlignmentMetric] = []

        if hasattr(self.config, "alignment_settings") and self.config.alignment_settings is not None:
            # MODIFIED: Read directly from self.config.alignment_settings.metric (which is now a list)
            metric_names_from_config = self.config.alignment_settings.metric
            scale_by_norm_default = self.config.alignment_settings.scale_by_norm # Global default

            if not metric_names_from_config or not isinstance(metric_names_from_config, list):
                logger.warning(
                    "AlignmentConfig.metric is empty, not a list, or not specified. Defaulting to [\"RQ\"]."
                )
                metric_names_from_config = ["RQ"]

            for metric_name in metric_names_from_config:
                if not isinstance(metric_name, str):
                    logger.warning(f"Invalid metric name type found in config: {metric_name}. Skipping.")
                    continue
                try:
                    instance = get_metric(name=metric_name, scale_by_norm=scale_by_norm_default)
                    if instance:
                        self.pruning_metric_instances.append(instance)
                        if self.primary_metric is None: # Assign the first valid instance as primary
                            self.primary_metric = instance
                    # else: # get_metric raises ValueError if metric not found, so no explicit None check needed here
                    #    logger.error(f"Metric '{metric_name}' could not be initialized from registry (get_metric returned None).")
                except ValueError as e:
                    logger.error(f"Error initializing metric '{metric_name}': {e}")
            
            if not self.pruning_metric_instances:
                logger.error("No pruning metric instances could be initialized from config. Falling back to default RQ.")
                # Fallback to a hardcoded default if all else fails
                try:
                    self.primary_metric = get_metric(name="RQ", scale_by_norm=scale_by_norm_default)
                    self.pruning_metric_instances = [self.primary_metric]
                except ValueError as e:
                    logger.critical(f"CRITICAL: Default RQ metric could not be initialized: {e}. Pruning experiments may fail.")
                    # self.pruning_metric_instances will remain empty, and self.primary_metric None.

            if self.primary_metric:
                 logger.info(f"Initialized primary metric: {self.primary_metric.name} (scale_by_norm={self.primary_metric.scale_by_norm})")
            else:
                 logger.warning("No primary metric could be initialized.")
            logger.info(f"Initialized {len(self.pruning_metric_instances)} pruning metric instances: {[m.name for m in self.pruning_metric_instances]}")

        else:
            logger.error("alignment_settings not found in config!")
            # Fallback to a hardcoded default if alignment_settings is entirely missing
            try:
                self.primary_metric = get_metric(name="RQ", scale_by_norm=False) # Provide a default scale_by_norm
                self.pruning_metric_instances = [self.primary_metric]
                logger.warning(f"Falling back to default alignment metric: {self.primary_metric.name}")
                logger.info(f"Initialized pruning metric instances: {[m.name for m in self.pruning_metric_instances]}")
            except ValueError as e:
                logger.critical(f"CRITICAL: Default RQ metric could not be initialized on fallback: {e}. Pruning experiments may fail.")

    # --- NEW: Method to initialize loss criterion ---
    def _initialize_loss_criterion(self):
        """Helper to initialize the loss criterion based on config."""
        self.loss_criterion = None
        loss_name = "CrossEntropyLoss"  # Default
        if hasattr(self.config, "training") and hasattr(self.config.training, "loss") and self.config.training.loss:
            loss_name_from_config = self.config.training.loss
            # Simple mapping for common losses. Can be extended.
            if loss_name_from_config.lower() == "cross_entropy" or loss_name_from_config.lower() == "crossentropy":
                loss_name = "CrossEntropyLoss"
            elif loss_name_from_config.lower() == "mse" or loss_name_from_config.lower() == "mseloss":
                loss_name = "MSELoss"
            # Add other mappings as needed, e.g., NLLLoss, BCELoss, etc.
            else:
                # Try to use the name directly if it's a valid nn.Module loss
                if (
                    hasattr(nn, loss_name_from_config)
                    and isinstance(getattr(nn, loss_name_from_config), type)
                    and issubclass(getattr(nn, loss_name_from_config), nn.Module)
                ):
                    loss_name = loss_name_from_config
                else:
                    logger.warning(f"Unsupported loss function '{loss_name_from_config}' in config. Defaulting to CrossEntropyLoss.")
                    loss_name = "CrossEntropyLoss"
        else:
            logger.warning("Loss function not specified in training config. Defaulting to CrossEntropyLoss.")

        try:
            # Instantiate the loss function.
            # This assumes loss functions are classes in torch.nn (e.g., nn.CrossEntropyLoss())
            # For functional losses like F.cross_entropy, train_model would need different handling or a wrapper.
            # For now, supporting nn.Module based losses instantiated here.
            if hasattr(nn, loss_name):
                self.loss_criterion = getattr(nn, loss_name)()
                logger.info(f"Initialized loss criterion: {loss_name}")
            else:
                logger.error(f"Loss class '{loss_name}' not found in torch.nn. Defaulting to CrossEntropyLoss.")
                self.loss_criterion = nn.CrossEntropyLoss()
        except Exception as e:
            logger.error(f"Error instantiating loss '{loss_name}': {e}. Defaulting to CrossEntropyLoss.")
            self.loss_criterion = nn.CrossEntropyLoss()

    # --- End NEW ---

    def get_basename(self) -> str:
        """
        Get the base name for the experiment. Overrides base Experiment method if needed.
        Uses experiment_name from config if available, otherwise falls back to model/dataset
        and includes pruning mode if relevant.
        """
        if hasattr(self.config, "experiment_name") and self.config.experiment_name:
            base = self.config.experiment_name
        else:
            model_name = self.config.model.model_name if hasattr(self.config, "model") else "unknown_model"
            dataset_n = self.current_dataset_name
            base = f"{model_name}_{dataset_n}"

        # Append experiment type and pruning mode for clarity if not in a custom name
        if not (hasattr(self.config, "experiment_name") and self.config.experiment_name):
            exp_type = self.config.experiment_type
            pruning_mode_str = ""
            if exp_type == "progressive_dropout":
                if hasattr(self.config, "pruning_settings") and self.config.pruning_settings.dropout_pruning_mode:
                    pruning_mode_str = f"_{self.config.pruning_settings.dropout_pruning_mode}"
                base = f"{exp_type}{pruning_mode_str}_{base}"
            elif exp_type == "layer_isolated_pruning":
                base = f"{exp_type}_{base}" # dropout_pruning_mode is less critical here
            elif exp_type == "eigenvector_dropout":
                if hasattr(self.config, "pruning_settings") and self.config.pruning_settings.dropout_pruning_mode:
                    pruning_mode_str = f"_{self.config.pruning_settings.dropout_pruning_mode}"
                base = f"{exp_type}{pruning_mode_str}_{base}"
            elif exp_type == "alignment_analysis":
                 base = f"{exp_type}_{base}"
            # Else, just the model_dataset base name is used

        return base

    def create_networks(self) -> List[nn.Module]:
        """
        Create multiple neural networks for the experiment, each with different initialization.

        Following the alignment_v2 approach, this creates multiple independent networks
        rather than copies of a single network.

        Returns:
            List containing multiple independently initialized networks
        """
        num_replicates = self.config.training.replicates
        networks_list = []  # Renamed to avoid confusion with self.networks
        for i in range(num_replicates):
            if self.config.seed is not None:
                torch.manual_seed(self.config.seed + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed + i)
                np.random.seed(self.config.seed + i)

            current_model_config = copy.deepcopy(self.config.model)

            # Set cnn_mode in ModelConfig from AlignmentSettings
            if hasattr(self.config, "alignment_settings") and self.config.alignment_settings.cnn_mode:
                current_model_config.cnn_mode = self.config.alignment_settings.cnn_mode
            # Remove from extra_model_params if it was there as part of the old workaround
            if current_model_config.extra_model_params and "cnn_mode" in current_model_config.extra_model_params:
                del current_model_config.extra_model_params["cnn_mode"]

            alignment_model_instance = create_model(current_model_config)
            _ensure_model_on_device(alignment_model_instance, self.device)

            final_model_for_list = alignment_model_instance

            # Check DDP conditions
            can_ddp = False
            if self.config.use_ddp and hasattr(torch, "distributed") and torch.distributed.is_initialized():
                if self.config.ddp_world_size > 1:
                    can_ddp = True

            if can_ddp:
                if self.device.type == "cuda":
                    final_model_for_list = nn.parallel.DistributedDataParallel(
                        alignment_model_instance,
                        device_ids=[self.config.ddp_local_rank],
                        output_device=self.config.ddp_local_rank,
                        find_unused_parameters=False,
                    )
                    logger.info(f"Wrapped model replicate {i} with DDP (CUDA). Local rank: {self.config.ddp_local_rank}")
                elif self.device.type == "cpu":
                    final_model_for_list = nn.parallel.DistributedDataParallel(alignment_model_instance, find_unused_parameters=False)
                    logger.info(f"Wrapped model replicate {i} with DDP (CPU).")

            networks_list.append(final_model_for_list)

        logger.info(f"Created {len(networks_list)} model replicates. DDP active for wrapping: {can_ddp}")
        return networks_list

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
        num_epochs = self.config.training.epochs
        learning_rate = self.config.training.learning_rate
        optimizer_name = self.config.training.optimizer
        weight_decay = self.config.training.weight_decay
        training_method = self.config.training.training_method

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
            callbacks=callbacks,
            loss_criterion=self.loss_criterion,
        )

    def _setup_callbacks(self) -> List[Callable]:
        """Prepare callback functions based on config."""
        callbacks_list = []
        self.metric_tracker_instance = None  # Reset instance

        if (
            self.config.alignment_settings
            and hasattr(self.config.alignment_settings, "callbacks")
            and self.config.alignment_settings.callbacks
            and self.config.alignment_settings.callbacks.alignment_metrics
        ):
            callback_metric_configs_from_yaml: List[MetricTrackerConfig] = self.config.alignment_settings.callbacks.alignment_metrics

            # Extract just the names for AlignmentMetricTracker constructor
            metric_names_to_track: List[str] = [mc.name for mc in callback_metric_configs_from_yaml]
            if not metric_names_to_track:
                logger.warning("No metric names found in callback configurations. Skipping AlignmentMetricTracker setup.")
                return callbacks_list

            logger.info(f"Configuring AlignmentMetricTracker for metric names: {metric_names_to_track}")

            metric_dataloader = self.dataset.test_loader
            if metric_dataloader is None and self.dataset.train_loader is not None:
                logger.warning("No test_loader in dataset for metric callback, using train_loader.")
                metric_dataloader = self.dataset.train_loader
            elif metric_dataloader is None:
                logger.error("No suitable dataloader for metric callback. Skipping tracker setup.")
                return callbacks_list  # Return empty list if no dataloader

            # Extract num_batches from the first metric config in YAML for the tracker, or default.
            # Tracker applies this num_batches globally for its activation collection.
            tracker_num_batches = (
                callback_metric_configs_from_yaml[0].num_batches
                if callback_metric_configs_from_yaml and callback_metric_configs_from_yaml[0].num_batches is not None
                else 5
            )

            # metric_kwargs for AlignmentMetricTracker constructor: if specific metrics in callbacks need special args.
            # For now, assuming AlignmentMetricTracker will use global settings from experiment_config
            # or defaults when it calls compute_metrics_for_layers.
            # If MetricTrackerConfig had more fields (e.g. scale_by_norm_override), they could be mapped here.
            tracker_metric_kwargs = {}
            # Example: Populate tracker_metric_kwargs from callback_metric_configs_from_yaml if they had more fields
            # for mc_obj in callback_metric_configs_from_yaml:
            #     if mc_obj.name not in tracker_metric_kwargs: tracker_metric_kwargs[mc_obj.name] = {}
            #     # if hasattr(mc_obj, 'some_specific_kwarg'): tracker_metric_kwargs[mc_obj.name]['some_specific_kwarg'] = mc_obj.some_specific_kwarg

            tracker = AlignmentMetricTracker(
                metric_names=metric_names_to_track,
                data_loader=metric_dataloader,
                device=self.device,
                num_batches=tracker_num_batches,
                experiment_config=self.config,  # Pass the main experiment config
                metric_kwargs=tracker_metric_kwargs,  # Pass specific kwargs if any were defined
            )
            callbacks_list.append(tracker)  # Tracker instance is callable due to __call__ method
            self.metric_tracker_instance = tracker
            logger.info(f"Added AlignmentMetricTracker for metric names: {metric_names_to_track}.")

        return callbacks_list

    def _run_initial_training(self, callbacks_list: List[Callable]):
        """Runs the initial training phase if configured."""
        if self.config.training.epochs > 0:
            if self.config.training.train_before_dropout:
                logger.info(f"Starting initial training for {self.config.training.epochs} epochs...")
                self.training_history = self.train_networks(
                    self.networks, self.dataset, callbacks=callbacks_list
                )  # train_networks will use self.loss_criterion
                if self.debug_mode and self.training_history and self.training_history.get("test_acc") and self.training_history["test_acc"]:
                    logger.info(f"Initial training final test accuracy: {self.training_history['test_acc'][-1]:.2f}%")
            else:
                logger.info("Skipping initial training because train_before_dropout is false.")
                self.training_history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        else:
            logger.info("Skipping initial training based on config (epochs=0).")
            self.training_history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    def _run_plotting_and_saving(self):
        """Handles plotting and saving results after the main experiment logic."""
        logger.info(f"Completed main computation for {self.config.experiment_type} experiment")
        overall_plot_files_generated: List[str] = []

        training_history_data = self.results.get("training_history") # training_history is top-level

        results_to_process_by_metric: Dict[str, Dict[str, Any]] = {}

        if self.config.experiment_type in ["progressive_dropout", "layer_isolated_pruning", "cascading_layer_pruning", "eigenvector_dropout"]:
            # These experiments now produce results directly as {metric_name: metric_results_dict, ...}
            # (excluding training_history which is handled separately)
            for key, value in self.results.items():
                if key != "training_history" and isinstance(value, dict): # Ensure value is a dict (metric_results)
                    results_to_process_by_metric[key] = value
        elif self.config.experiment_type == "alignment_analysis":
            # AlignmentAnalysis nests results under "progressive_dropout" and "eigenvector_dropout"
            # Both of these will now be dicts keyed by metric name.
            if "progressive_dropout" in self.results and isinstance(self.results["progressive_dropout"], dict):
                for metric_name, metric_results in self.results["progressive_dropout"].items():
                    if metric_name != "training_history" and isinstance(metric_results, dict):
                         results_to_process_by_metric[f"prog_{metric_name}"] = metric_results
            
            if "eigenvector_dropout" in self.results and isinstance(self.results["eigenvector_dropout"], dict):
                for metric_name, metric_results in self.results["eigenvector_dropout"].items():
                    if metric_name != "training_history" and isinstance(metric_results, dict):
                         results_to_process_by_metric[f"eig_{metric_name}"] = metric_results 
        else:
            logger.warning(f"Plotting logic for experiment type {self.config.experiment_type} might not fully support multi-metric results structure or is not a recognized experiment type for standard plotting.")
            # Attempt to plot if self.results itself looks like a single metric's results (legacy or unknown type)
            if isinstance(self.results, dict) and "accuracies" in self.results and "dropout_fractions" in self.results:
                 logger.info("Attempting to plot self.results as a single metric's data.")
                 results_to_process_by_metric["unknown_metric_type_direct_results"] = self.results

        if not results_to_process_by_metric and not (self.config.experiment_type == "alignment_analysis" and not self.config.alignment_settings.run_progressive and not self.config.alignment_settings.run_eigenvector) :
            # Only warn if not an alignment_analysis run where both sub-experiments were disabled
            logger.warning("No metric-specific results found to plot. Skipping main plotting loop.")
            if "error" in self.results:
                 logger.error(f"Experiment run resulted in an error: {self.results['error']}")
            # Save raw results even if plotting is skipped (base self.results)
            results_filename = f"{self.config.experiment_name}_{self.config.experiment_type}_main_results.pkl"
            self.save_results(self.results, filename=results_filename)
            # Still attempt to plot metric evolution if tracker ran
            # This plotting part is moved after the main loop for results_to_process_by_metric
        
        # Loop through each metric's results and generate plots
        for metric_key_in_plot_dict, metric_specific_results_for_plot in results_to_process_by_metric.items():
            # metric_key_in_plot_dict could be "RQ", "MI_G", or "prog_RQ", "eig_RQ" etc.
            # We need the actual metric name for axis labels and consistent naming.
            actual_metric_name_for_plot = metric_key_in_plot_dict
            if metric_key_in_plot_dict.startswith("prog_"):
                actual_metric_name_for_plot = metric_key_in_plot_dict[len("prog_"):]
            elif metric_key_in_plot_dict.startswith("eig_"):
                actual_metric_name_for_plot = metric_key_in_plot_dict[len("eig_"):]
            # else, it's a direct metric name or "unknown_metric_type_direct_results"

            if not isinstance(metric_specific_results_for_plot, dict) or "error" in metric_specific_results_for_plot:
                logger.warning(f"Skipping plotting for metric key '{metric_key_in_plot_dict}' due to missing data or error: {metric_specific_results_for_plot.get('error')}")
                continue

            logger.info(f"[Plotting] Generating plots for metric key: {metric_key_in_plot_dict} (Actual Metric: {actual_metric_name_for_plot})")
            current_metric_plot_files: List[str] = [] # Renamed to avoid conflict with outer scope if this was nested differently before

            # Use metric_key_in_plot_dict for unique filenames if it includes prefixes like prog_ or eig_
            title_prefix_for_current_plot = f"{self.config.experiment_name}_{metric_key_in_plot_dict}"
            show_all_plots = self.config.show_all
            pruning_mode_plot = self.config.pruning_settings.dropout_pruning_mode # This is a global setting
            dropout_mode_plot = self.config.pruning_settings.dropout_mode # Global

            # Determine effective experiment type for this specific plot based on metric_key_in_plot_dict
            effective_exp_type_for_plot = self.config.experiment_type
            if metric_key_in_plot_dict.startswith("prog_"):
                effective_exp_type_for_plot = "progressive_dropout"
            elif metric_key_in_plot_dict.startswith("eig_"):
                effective_exp_type_for_plot = "eigenvector_dropout"
            # For direct metric keys, effective_exp_type_for_plot remains self.config.experiment_type

            if "accuracies" in metric_specific_results_for_plot: 
                effective_pruning_mode_title = pruning_mode_plot
                if effective_exp_type_for_plot == "cascading_layer_pruning":
                     effective_pruning_mode_title = "cascading_layer"
                
                acc_plots = plot_dropout_results(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, effective_pruning_mode_title, dropout_mode_plot, metric_name_for_plot=actual_metric_name_for_plot)
                if acc_plots: current_metric_plot_files.extend(acc_plots if isinstance(acc_plots, list) else [acc_plots])

                if metric_specific_results_for_plot.get("pruning_details") and metric_specific_results_for_plot.get("pre_pruning_layer_stats"):
                    mean_score_plot = plot_mean_score_of_pruned_nodes(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, show_all_plots, metric_name_for_axis=actual_metric_name_for_plot)
                    if mean_score_plot: current_metric_plot_files.append(mean_score_plot)

                    if effective_exp_type_for_plot == "progressive_dropout":
                        percent_plot = plot_per_layer_pruning_percentage(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, show_all_plots)
                        if percent_plot: current_metric_plot_files.append(percent_plot)
                        contrib_plot = plot_per_layer_contribution_to_pruning(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, show_all_plots)
                        if contrib_plot: current_metric_plot_files.append(contrib_plot)
                
                if metric_specific_results_for_plot.get("pre_pruning_layer_stats"):
                    score_stats_plot = plot_score_stats_per_layer(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, show_all_plots, metric_name_for_axis=actual_metric_name_for_plot)
                    if score_stats_plot: current_metric_plot_files.append(score_stats_plot)

            if effective_exp_type_for_plot == "layer_isolated_pruning" and "accuracies_isolated" in metric_specific_results_for_plot:
                iso_acc_plots = plot_layer_isolated_dropout_results(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, show_all_plots, metric_name_for_plot=actual_metric_name_for_plot)
                if iso_acc_plots: current_metric_plot_files.extend(iso_acc_plots)
                
                if metric_specific_results_for_plot.get("pre_pruning_layer_stats"):
                    iso_score_stats_plot = plot_score_stats_per_layer(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, show_all_plots, metric_name_for_axis=actual_metric_name_for_plot)
                    if iso_score_stats_plot: current_metric_plot_files.append(iso_score_stats_plot)
                
                if metric_specific_results_for_plot.get("pruning_details") and metric_specific_results_for_plot.get("pre_pruning_layer_stats"):
                    iso_mean_score_pruned_plot = plot_mean_score_of_pruned_nodes(metric_specific_results_for_plot, self.figure_path, title_prefix_for_current_plot, show_all_plots, metric_name_for_axis=actual_metric_name_for_plot)
                    if iso_mean_score_pruned_plot: current_metric_plot_files.append(iso_mean_score_pruned_plot)
            
            overall_plot_files_generated.extend(current_metric_plot_files)

        # Summary plot for alignment_analysis (compares progressive vs eigenvector for primary metric)
        if self.config.experiment_type == "alignment_analysis":
            primary_metric_name_for_summary = self.primary_metric.name if self.primary_metric else "RQ" # Fallback if primary_metric is somehow None
            prog_key_for_summary = f"prog_{primary_metric_name_for_summary}"
            eig_key_for_summary = f"eig_{primary_metric_name_for_summary}"

            prog_results_for_summary = results_to_process_by_metric.get(prog_key_for_summary)
            eig_results_for_summary = results_to_process_by_metric.get(eig_key_for_summary)
            
            if prog_results_for_summary and eig_results_for_summary:
                summary_data_for_plot = {
                    "progressive_dropout": prog_results_for_summary,
                    "eigenvector_dropout": eig_results_for_summary 
                }
                title_prefix_summary = f"{self.config.experiment_name}_{primary_metric_name_for_summary}_summary"
                summary_path = plot_experiment_summary(summary_data_for_plot, self.figure_path, title_prefix_summary)
                if summary_path:
                    overall_plot_files_generated.append(summary_path)
            else:
                logger.warning(f"Could not generate summary plot for alignment_analysis: missing progressive ('{prog_key_for_summary}') or eigenvector ('{eig_key_for_summary}') results for the primary metric.")

        # Plot metric evolution from tracker (metric-specific by design)
        if self.metric_tracker_instance and self.metric_tracker_instance.metrics_evolution:
            logger.info(f"Generating evolution plots from AlignmentMetricTracker...")
            tracked_metric_names = self.metric_tracker_instance.metric_names
            for metric_name_to_plot in tracked_metric_names:
                current_metric_evolution_for_plot: Dict[str, Dict[str, List[Any]]] = {}
                for epoch_data in self.metric_tracker_instance.metrics_evolution:
                    epoch_num = epoch_data["epoch"]
                    all_scores_this_epoch = epoch_data["all_scores_per_layer"]
                    for layer_name, metrics_in_layer in all_scores_this_epoch.items():
                        if metric_name_to_plot in metrics_in_layer:
                            scores_tensor = metrics_in_layer[metric_name_to_plot]
                            if not isinstance(scores_tensor, torch.Tensor):
                                mean_score = float(scores_tensor) if isinstance(scores_tensor, (float, int)) else np.nan
                                std_score = 0.0
                            elif scores_tensor.numel() == 0: continue
                            else:
                                mean_score = torch.mean(scores_tensor.float()).item()
                                std_score = torch.std(scores_tensor.float()).item() if scores_tensor.numel() > 1 else 0.0
                            if layer_name not in current_metric_evolution_for_plot:
                                current_metric_evolution_for_plot[layer_name] = {"epochs": [], "mean_scores": [], "std_scores": []}
                            current_metric_evolution_for_plot[layer_name]["epochs"].append(epoch_num)
                            current_metric_evolution_for_plot[layer_name]["mean_scores"].append(mean_score)
                            current_metric_evolution_for_plot[layer_name]["std_scores"].append(std_score)
                if current_metric_evolution_for_plot:
                    for layer_data_for_plot in current_metric_evolution_for_plot.values():
                        if layer_data_for_plot["epochs"]:
                            sorted_indices = np.argsort(layer_data_for_plot["epochs"])
                            layer_data_for_plot["epochs"] = [layer_data_for_plot["epochs"][i] for i in sorted_indices]
                            layer_data_for_plot["mean_scores"] = [layer_data_for_plot["mean_scores"][i] for i in sorted_indices]
                            layer_data_for_plot["std_scores"] = [layer_data_for_plot["std_scores"][i] for i in sorted_indices]
                    evolution_plot_path = plot_metric_evolution(
                        metric_evolution_data=current_metric_evolution_for_plot,
                        metric_name=metric_name_to_plot,
                        save_dir=self.figure_path,
                        title_prefix=f"{self.config.experiment_name}", # Title prefix already includes experiment name
                        show_plots=show_all_plots,
                    )
                    if evolution_plot_path: overall_plot_files_generated.append(evolution_plot_path)
                else: logger.info(f"No evolution data found to plot for metric: {metric_name_to_plot}")

        if overall_plot_files_generated:
            self.results["plot_files"] = overall_plot_files_generated # Store all generated plot paths
            if self.wandb_run:
                log_plots_to_wandb(overall_plot_files_generated, tags={"experiment_type": self.config.experiment_type})

        if self.metric_tracker_instance and self.metric_tracker_instance.metrics_evolution:
            # Store the raw evolution data if needed, perhaps already part of self.results if added by experiment logic
            if "all_metrics_evolution_data" not in self.results: # Avoid overwriting if already set by specific experiment
                 self.results["all_metrics_evolution_data"] = self.metric_tracker_instance.metrics_evolution
            logger.info(f"Stored consolidated metric evolution data in results.")

        # Save results using base class method
        # Filename reflects the overall experiment, results dict now contains per-metric sub-dicts.
        results_filename = f"{self.config.experiment_name}_{self.config.experiment_type}_main_results.pkl"
        self.save_results(self.results, filename=results_filename)

        logger.info(f"Completed {self.config.experiment_name} (type: {self.config.experiment_type}) experiment run.")

    # The main execution flow, common setup + call subclass 'run' + common teardown
    def execute_experiment(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Main execution flow: setup, run specific logic, teardown.
        This replaces the old 'run' method which called 'main'.
        The base class 'Experiment.run' will call this.
        """
        # 1. Setup (already done by base class __init__ and AlignmentExperiment __init__)
        # self.setup_paths() # Base class Experiment.__init__ handles this
        # logger.info(f"Set up paths. Results will be saved to {self.results_path}") # Base class logs this

        # 2. Create Networks & Load Dataset (Common Setup)
        self.networks = self.create_networks()
        batch_size = self.config.dataset.batch_size
        
        # Prepare transform parameters based on model type
        model_name = self.config.model.model_name
        dataset_name = self.config.dataset.dataset_name
        from alignment_refac1.models.models import get_transform_parameters
        
        # Normalize external model names to their base names for transform lookup
        normalized_model_name = model_name.lower()
        if normalized_model_name.startswith("torchvision_"):
            normalized_model_name = normalized_model_name.replace("torchvision_", "")
        elif normalized_model_name.startswith("hf_"):
            hf_name = normalized_model_name.replace("hf_", "")
            if "bert" in hf_name:
                normalized_model_name = "bert"
            elif "gpt" in hf_name:
                normalized_model_name = "gpt"
            else:
                normalized_model_name = hf_name.split("-")[0]
        
        try:
            transform_params = get_transform_parameters(normalized_model_name, dataset_name)
            logger.info(f"Using transform parameters for model '{normalized_model_name}' (from '{model_name}') with dataset '{dataset_name}': {transform_params}")
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not get transform parameters for model '{normalized_model_name}': {e}. Using default (no transforms).")
            transform_params = {}
        
        self.dataset = load_dataset(
            self.config.dataset,
            batch_size=batch_size,
            device=self.device,
            transform_params=transform_params,
            use_ddp=self.config.use_ddp,
            ddp_rank=self.config.ddp_rank,
            ddp_world_size=self.config.ddp_world_size,
        )

        # 3. Setup Callbacks (Common Setup)
        callbacks_list = self._setup_callbacks()

        # 4. Initial Training (Common Setup)
        self._run_initial_training(callbacks_list)

        # 5. Run the specific experiment logic (implemented by subclass)
        # The '_run_specific_logic' method implemented by subclasses will perform the core computations
        # and should store its primary output in self.results.
        self.results = self._run_specific_logic()  # Call the RENAMED abstract method

        # 6. Plotting and Saving (Common Teardown/Reporting)
        self._run_plotting_and_saving()

        # 7. Return results and networks
        # Base class Experiment.run() calls this method and returns its result.
        return self.results, self.networks

    # This is the method called by the base Experiment class's execution flow.
    def run(self) -> Tuple[Dict, List[nn.Module]]:
        """ 
        This method is called by the base Experiment class (or its equivalent execution trigger).
        It orchestrates the experiment by calling execute_experiment.
        """
        return self.execute_experiment()

    # Abstract run method to be implemented by subclasses
    def _run_specific_logic(self) -> Dict:
        """
        Abstract method for experiment-specific logic.
        Subclasses must implement this method to perform their computations.
        The results dictionary should be returned.
        """
        raise NotImplementedError("Subclasses must implement the '_run_specific_logic' method.")


# --- Subclasses for Specific Experiment Types ---


class ProgressiveDropoutExperiment(AlignmentExperiment):
    """Experiment for progressive dropout based on alignment metrics."""

    def _run_specific_logic(self) -> Dict:
        """Runs the progressive dropout experiment."""
        logger.info("Running Progressive Dropout Experiment specific logic")

        # Common setup: Create networks, load data, initial training (done in __init__ or called before run)
        if not self.networks or self.dataset is None:
            raise RuntimeError("Networks or dataset not initialized before run(). Ensure execute_experiment is called.")

        # Get necessary configs
        pruning_config = self.config.pruning_settings
        alignment_config = self.config.alignment_settings

        # Prepare dropout fractions
        dropout_min = pruning_config.dropout_min
        dropout_max = pruning_config.dropout_max
        num_dropout_steps = pruning_config.dropout_steps
        if num_dropout_steps <= 0:
            _fractions = [0.0, dropout_max]
        elif num_dropout_steps == 1:
            _fractions = [dropout_min, dropout_max]
        else:
            _fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in _fractions:
            _fractions = [0.0] + _fractions
        dropout_fractions = sorted(list(set(_fractions)))

        pruning_mode = pruning_config.dropout_pruning_mode
        dropout_mode_val = pruning_config.dropout_mode
        effective_exclude_classification_layer = pruning_config.exclude_classification_layer
        num_batches_for_metric_calc = pruning_config.num_batches_for_scores
        force_cpu_flag = alignment_config.force_cpu_for_large_metric_ops
        cnn_mode_for_pruning_scores = alignment_config.cnn_mode
        cnn_rq_op_for_pruning_scores = alignment_config.cnn_rq_aggregation_op
        
        # MODIFIED: Use the list of pruning_metric_instances
        metrics_to_use = self.pruning_metric_instances 

        if not metrics_to_use:
            logger.error("No metrics initialized properly for progressive dropout!")
            # Attempt to re-initialize metric here or raise error
            self._initialize_metric() # This re-initializes self.pruning_metric_instances
            metrics_to_use = self.pruning_metric_instances
            if not metrics_to_use:
                raise ValueError("Failed to initialize any metrics for progressive dropout.")

        logger.info(
            f"Using metrics: {[m.name for m in metrics_to_use]}, PruningMode: {pruning_mode}, DropoutMode: {dropout_mode_val}, ExcludeCls: {effective_exclude_classification_layer}"
        )

        # Call the manager function
        results_from_manager = {}
        try:
            results_from_manager = run_progressive_dropout_experiment(
                self.networks,
                self.dataset,
                dropout_fractions,
                metrics_to_use, # MODIFIED: Pass the list of metric instances
                self.device,
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode_val,
                show_progress=True,
                debug_mode=self.debug_mode,
                exclude_classification_layer_config=effective_exclude_classification_layer,
                num_batches_for_pre_scoring=num_batches_for_metric_calc,
                force_cpu_for_large_metric_ops=force_cpu_flag,
                configured_cnn_mode=cnn_mode_for_pruning_scores,
                configured_cnn_rq_op=cnn_rq_op_for_pruning_scores,
            )
        except Exception as e:
            logger.error(f"Error during run_progressive_dropout_experiment call: {str(e)}")
            results_from_manager = {"error": str(e)}
            if self.debug_mode:
                logger.error(traceback.format_exc())

        # Add training history to the results (now results_from_manager is a dict of dicts)
        # We can add training_history to each metric's sub-dictionary or at a common level.
        # For now, let's add it at the top level if results_from_manager is not just an error dict.
        if "error" not in results_from_manager:
            results_from_manager["training_history"] = self.training_history
        elif not results_from_manager: # If it's empty due to no metrics
            results_from_manager["training_history"] = self.training_history

        return results_from_manager


class EigenvectorDropoutExperiment(AlignmentExperiment):
    """Experiment for eigenvector-based dropout."""

    def _run_specific_logic(self) -> Dict:
        """Runs the eigenvector dropout experiment for each specified metric."""
        logger.info("Running Eigenvector Dropout Experiment specific logic")

        if not self.networks or self.dataset is None:
            raise RuntimeError("Networks or dataset not initialized before run(). Ensure execute_experiment is called.")

        if not self.networks:
            logger.error("No networks available for Eigenvector Dropout Experiment.")
            return {"error": "No networks found."}

        # MODIFIED: Iterate over all pruning_metric_instances
        metrics_to_process = self.pruning_metric_instances
        if not metrics_to_process:
            logger.error("No metrics initialized properly for Eigenvector Dropout!")
            # Attempt to re-initialize if necessary, though _initialize_metric should have handled fallbacks
            self._initialize_metric()
            metrics_to_process = self.pruning_metric_instances
            if not metrics_to_process:
                # If still no metrics after re-init, this is a critical failure.
                return {"error": "Failed to initialize any metrics for Eigenvector Dropout."}

        overall_results_by_metric: Dict[str, Dict] = {}
        network_to_use = self.networks[0]  # Eigenvector typically runs on one network replicate
        logger.info(f"Using the first network replicate (of {len(self.networks)}) for all eigenvector dropout runs.")

        pruning_config = self.config.pruning_settings
        # alignment_config = self.config.alignment_settings # Not directly used here, metric instance has scale_by_norm

        dropout_min = pruning_config.dropout_min
        dropout_max = pruning_config.dropout_max
        num_dropout_steps = pruning_config.dropout_steps
        if num_dropout_steps <= 0:
            _fractions = [0.0, dropout_max]
        elif num_dropout_steps == 1:
            _fractions = [dropout_min, dropout_max]
        else:
            _fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in _fractions:
            _fractions = [0.0] + _fractions
        dropout_fractions = sorted(list(set(_fractions)))

        dropout_mode_val = pruning_config.dropout_mode
        pruning_mode_for_eigen = pruning_config.dropout_pruning_mode

        for metric_to_use in metrics_to_process:
            current_metric_name = metric_to_use.name
            logger.info(f"Eigenvector Dropout: Processing for metric: {current_metric_name}")
            logger.info(f"  Using metric: {current_metric_name}, PruningMode: {pruning_mode_for_eigen}, DropoutMode: {dropout_mode_val}")

            single_metric_results = {}
            try:
                # run_eigenvector_dropout_experiment is the manager function
                single_metric_results = run_eigenvector_dropout_experiment(
                    network=network_to_use, # Pass a fresh copy if manager modifies it, though it shouldn't for just one frac
                    dataset=self.dataset,
                    dropout_fractions=dropout_fractions,
                    metric=metric_to_use, # Current metric in the loop
                    device=self.device,
                    dropout_mode=dropout_mode_val,
                    pruning_mode=pruning_mode_for_eigen,
                    show_progress=True, # Or self.config.show_progress
                    debug_mode=self.debug_mode,
                )
            except Exception as e:
                logger.error(f"Error during run_eigenvector_dropout_experiment call for metric {current_metric_name}: {str(e)}")
                single_metric_results = {"error": str(e)}
                if self.debug_mode:
                    logger.error(traceback.format_exc())
            
            overall_results_by_metric[current_metric_name] = single_metric_results

        overall_results_by_metric["training_history"] = self.training_history
        return overall_results_by_metric


class LayerIsolatedPruningExperiment(AlignmentExperiment):
    """Experiment for pruning layers in isolation."""

    def _run_specific_logic(self) -> Dict:
        """Runs the layer isolated pruning experiment for each specified metric."""
        logger.info("Running Layer Isolated Pruning Experiment specific logic")

        if not self.networks or self.dataset is None:
            raise RuntimeError("Networks or dataset not initialized before run(). Ensure execute_experiment is called.")

        pruning_config = self.config.pruning_settings
        alignment_config = self.config.alignment_settings

        dropout_min = pruning_config.dropout_min
        dropout_max = pruning_config.dropout_max
        num_dropout_steps = pruning_config.dropout_steps
        if num_dropout_steps <= 0:
            _fractions = [0.0, dropout_max]
        elif num_dropout_steps == 1:
            _fractions = [dropout_min, dropout_max]
        else:
            _fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in _fractions:  # Ensure 0.0 is present for baseline
            _fractions = [0.0] + _fractions
        dropout_fractions = sorted(list(set(_fractions)))

        exclude_cls_layer = pruning_config.exclude_classification_layer
        dropout_mode_val = pruning_config.dropout_mode
        num_batches_for_metric_calc_isolated = pruning_config.num_batches_for_scores
        force_cpu_flag_isolated = alignment_config.force_cpu_for_large_metric_ops
        cnn_mode_for_isolated_scores = alignment_config.cnn_mode
        cnn_rq_op_for_isolated_scores = alignment_config.cnn_rq_aggregation_op
        
        # MODIFIED: Iterate over all pruning_metric_instances
        metrics_to_process = self.pruning_metric_instances
        if not metrics_to_process:
            logger.error("No metrics initialized properly for layer isolated pruning!")
            self._initialize_metric() # Re-attempt initialization
            metrics_to_process = self.pruning_metric_instances
            if not metrics_to_process:
                raise ValueError("Failed to initialize any metrics for layer isolated pruning.")

        overall_results_by_metric: Dict[str, Dict] = {}

        for metric_instance_for_isolated in metrics_to_process:
            current_metric_name = metric_instance_for_isolated.name
            logger.info(f"Layer Isolated Pruning: Processing for metric: {current_metric_name}")
            logger.info(f"  Using metric: {current_metric_name}, DropoutMode: {dropout_mode_val}, ExcludeCls: {exclude_cls_layer}")

            # Call the manager function for the current metric
            # The manager function (run_layer_isolated_dropout_experiment) is assumed to return 
            # the standard results structure (accuracies_isolated, stds_isolated, etc.) for a single metric.
            single_metric_results = {}
            try:
                single_metric_results = run_layer_isolated_dropout_experiment(
                    original_networks=self.networks, # Pass fresh copies for each metric run might be safer if manager modifies networks
                    dataset=self.dataset,
                    dropout_fractions=dropout_fractions,
                    metric=metric_instance_for_isolated, # Pass the current metric
                    device=self.device,
                    dropout_mode=dropout_mode_val,
                    show_progress=True, # Or self.config.show_progress
                    debug_mode=self.debug_mode,
                    exclude_classification_layer_config=exclude_cls_layer,
                    num_batches_for_pre_scoring=num_batches_for_metric_calc_isolated,
                    force_cpu_for_large_metric_ops=force_cpu_flag_isolated,
                    configured_cnn_mode=cnn_mode_for_isolated_scores,
                    configured_cnn_rq_op=cnn_rq_op_for_isolated_scores,
                )
            except Exception as e:
                logger.error(f"Error during run_layer_isolated_dropout_experiment call for metric {current_metric_name}: {str(e)}")
                single_metric_results = {"error": str(e)}
                if self.debug_mode:
                    logger.error(traceback.format_exc())
            
            overall_results_by_metric[current_metric_name] = single_metric_results

        # Add training history at a common level
        overall_results_by_metric["training_history"] = self.training_history
        return overall_results_by_metric


class AlignmentAnalysisExperiment(AlignmentExperiment):
    """Meta-experiment that runs multiple alignment analyses (e.g., progressive and eigenvector)."""

    def _run_specific_logic(self) -> Dict:
        """Runs the alignment analysis experiment, iterating over metrics for both sub-experiments."""
        logger.info("Running Alignment Analysis Experiment specific logic")

        if not self.networks or self.dataset is None:
            raise RuntimeError("Networks or dataset not initialized before run(). Ensure execute_experiment is called.")

        analysis_results: Dict[str, Any] = {"alignment_analysis": True} 

        pruning_config = self.config.pruning_settings
        alignment_config = self.config.alignment_settings # For cnn_mode, force_cpu etc.

        # Prepare dropout fractions (common for both sub-experiments)
        dropout_min = pruning_config.dropout_min
        dropout_max = pruning_config.dropout_max
        num_dropout_steps = pruning_config.dropout_steps
        if num_dropout_steps <= 0: _fractions = [0.0, dropout_max]
        elif num_dropout_steps == 1: _fractions = [dropout_min, dropout_max]
        else: _fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in _fractions: _fractions = [0.0] + _fractions
        dropout_fractions = sorted(list(set(_fractions)))
        
        metrics_to_process = self.pruning_metric_instances
        if not metrics_to_process:
            logger.error("No metrics initialized for Alignment Analysis!")
            self._initialize_metric() # Re-attempt initialization
            metrics_to_process = self.pruning_metric_instances
            if not metrics_to_process:
                # If still no metrics, this is a critical failure.
                analysis_results["error"] = "Failed to initialize any metrics for Alignment Analysis."
                analysis_results["training_history"] = self.training_history # Still add history
                return analysis_results

        # Run Progressive Dropout if configured (will run for all metrics in metrics_to_process)
        if alignment_config.run_progressive:
            logger.info("Alignment Analysis: Running progressive dropout part for all configured metrics.")
            logger.info(f"  Metrics for progressive dropout: {[m.name for m in metrics_to_process]}")
            try:
                # run_progressive_dropout_experiment now handles the list of metrics internally
                # and returns results keyed by metric name.
                prog_results_by_metric = run_progressive_dropout_experiment(
                    self.networks,
                    self.dataset,
                    dropout_fractions,
                    metrics_to_process, # Pass the full list
                    self.device,
                    pruning_mode=pruning_config.dropout_pruning_mode,
                    dropout_mode=pruning_config.dropout_mode,
                    show_progress=True,
                    debug_mode=self.debug_mode,
                    exclude_classification_layer_config=pruning_config.exclude_classification_layer,
                    num_batches_for_pre_scoring=pruning_config.num_batches_for_scores,
                    force_cpu_for_large_metric_ops=alignment_config.force_cpu_for_large_metric_ops,
                    configured_cnn_mode=alignment_config.cnn_mode,
                    configured_cnn_rq_op=alignment_config.cnn_rq_aggregation_op,
                )
                analysis_results["progressive_dropout"] = prog_results_by_metric
            except Exception as e:
                logger.error(f"Error running progressive dropout within analysis: {e}")
                analysis_results["progressive_dropout"] = {"error": str(e)}
        else:
            logger.info("Alignment Analysis: Skipping progressive dropout part as per config.")

        # Run Eigenvector Dropout if configured (will also run for all metrics in metrics_to_process)
        if alignment_config.run_eigenvector:
            logger.info("Alignment Analysis: Running eigenvector dropout part for all configured metrics.")
            logger.info(f"  Metrics for eigenvector dropout: {[m.name for m in metrics_to_process]}")
            
            if not self.networks:
                logger.error("No networks available for Eigenvector Dropout in analysis.")
                analysis_results["eigenvector_dropout"] = {"error": "No networks found for eigenvector part."}
            else:
                network_to_use_for_eigen = self.networks[0] # Eigenvector typically uses one network replicate
                eigen_results_by_metric: Dict[str, Dict] = {}
                for metric_instance in metrics_to_process:
                    current_metric_name = metric_instance.name
                    logger.info(f"  Eigenvector Dropout for metric: {current_metric_name}")
                    try:
                        eig_single_metric_results = run_eigenvector_dropout_experiment(
                            network=network_to_use_for_eigen, # Consider fresh copy for each metric if needed
                            dataset=self.dataset,
                            dropout_fractions=dropout_fractions,
                            metric=metric_instance,
                            device=self.device,
                            dropout_mode=pruning_config.dropout_mode,
                            pruning_mode=pruning_config.dropout_pruning_mode,
                            show_progress=True,
                            debug_mode=self.debug_mode,
                        )
                        eigen_results_by_metric[current_metric_name] = eig_single_metric_results
                    except Exception as e:
                        logger.error(f"Error running eigenvector dropout for metric {current_metric_name} within analysis: {e}")
                        eigen_results_by_metric[current_metric_name] = {"error": str(e)}
                analysis_results["eigenvector_dropout"] = eigen_results_by_metric
        else:
            logger.info("Alignment Analysis: Skipping eigenvector dropout part as per config.")

        analysis_results["training_history"] = self.training_history
        return analysis_results


# --- NEW EXPERIMENT SUBCLASS FOR CASCADING LAYER PRUNING ---
class CascadingLayerPruningExperiment(AlignmentExperiment):
    """Experiment for cascading layer pruning."""

    def _run_specific_logic(self) -> Dict:
        """Runs the cascading layer pruning experiment for each specified metric."""
        logger.info("Running Cascading Layer Pruning Experiment specific logic")

        if not self.networks or self.dataset is None:
            raise RuntimeError("Networks or dataset not initialized. Ensure execute_experiment is called.")

        pruning_config = self.config.pruning_settings
        alignment_config = self.config.alignment_settings

        dropout_min = pruning_config.dropout_min
        dropout_max = pruning_config.dropout_max
        num_dropout_steps = pruning_config.dropout_steps
        _fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist() if num_dropout_steps > 0 else [0.0, dropout_max]
        if 0.0 not in _fractions: _fractions = [0.0] + _fractions
        dropout_fractions = sorted(list(set(_fractions)))

        # MODIFIED: Iterate over all pruning_metric_instances
        metrics_to_process = self.pruning_metric_instances
        if not metrics_to_process:
            logger.error("No metrics initialized properly for cascading layer pruning!")
            self._initialize_metric() # Re-attempt initialization
            metrics_to_process = self.pruning_metric_instances
            if not metrics_to_process:
                raise ValueError("Failed to initialize any metrics for cascading layer pruning.")

        overall_results_by_metric: Dict[str, Dict] = {}

        for metric_instance_for_cascading in metrics_to_process:
            current_metric_name = metric_instance_for_cascading.name
            logger.info(f"Cascading Layer Pruning: Processing for metric: {current_metric_name}")
            logger.info(
                f"  Using metric: {current_metric_name} for score calculation in cascading prune. "
                f"DropoutMode: {pruning_config.dropout_mode}, ExcludeCls: {pruning_config.exclude_classification_layer}"
            )
            
            single_metric_results = {}
            try:
                # Note: run_cascading_layer_pruning_experiment currently assumes a single metric internally for its 
                # 'strategies_to_run_cascade' (high_rq, low_rq, random). These strategies will be based on the 
                # *current* metric_instance_for_cascading.
                single_metric_results = run_cascading_layer_pruning_experiment(
                    networks=self.networks, # Consider passing deep copies if the manager modifies them in-place across metrics
                    dataset=self.dataset,
                    dropout_fractions=dropout_fractions,
                    metric_instance=metric_instance_for_cascading, # Pass current metric
                    device=self.device,
                    dropout_mode=pruning_config.dropout_mode,
                    show_progress=True, # Or self.config.show_progress
                    debug_mode=self.debug_mode,
                    exclude_classification_layer_config=pruning_config.exclude_classification_layer,
                    num_batches_for_pre_scoring=pruning_config.num_batches_for_scores,
                    force_cpu_for_large_metric_ops=alignment_config.force_cpu_for_large_metric_ops,
                    configured_cnn_mode=alignment_config.cnn_mode, 
                    configured_cnn_rq_op=alignment_config.cnn_rq_aggregation_op
                )
            except Exception as e:
                logger.error(f"Error during run_cascading_layer_pruning_experiment call for metric {current_metric_name}: {str(e)}")
                single_metric_results = {"error": str(e)}
                if self.debug_mode:
                    logger.error(traceback.format_exc())

            overall_results_by_metric[current_metric_name] = single_metric_results

        overall_results_by_metric["training_history"] = self.training_history
        return overall_results_by_metric


# --- Factory Function ---


def get_experiment_class(experiment_type: str) -> Type[AlignmentExperiment]:
    """Gets the appropriate experiment class based on the type string."""
    if experiment_type == "progressive_dropout":
        return ProgressiveDropoutExperiment
    elif experiment_type == "eigenvector_dropout":
        return EigenvectorDropoutExperiment
    elif experiment_type == "layer_isolated_pruning":
        return LayerIsolatedPruningExperiment
    elif experiment_type == "alignment_analysis":
        return AlignmentAnalysisExperiment
    elif experiment_type == "cascading_layer_pruning": # Added new type
        return CascadingLayerPruningExperiment
    else:
        raise ValueError(f"Unsupported experiment type: {experiment_type}")


# --- CLI Entry Point ---


def cli_main():
    """Command-line interface for running alignment experiments."""
    # import argparse # Moved to top

    parser = argparse.ArgumentParser(description="Neural network alignment experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP (usually set by launcher)")  # For DDP

    args = parser.parse_args()

    # Set logging level
    log_level = "WARNING" if args.quiet else "INFO"
    # Base Experiment __init__ handles logging setup if config.log_level is set, or defaults.
    # setup_logging(log_level=log_level) # REMOVE THIS LINE

    # Load configuration
    config = ExperimentConfig.load(args.config)

    # --- DDP Setup ---
    is_ddp_initialized = False
    if config.use_ddp:
        # import torch.distributed as dist # Moved to top

        # import os # Already imported at top of file

        if args.local_rank != -1:  # Launched with torch.distributed.launch or similar
            config.ddp_local_rank = args.local_rank
        elif "LOCAL_RANK" in os.environ:
            config.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        else:
            logger.warning(
                "DDP is enabled but local_rank is not provided via --local_rank or LOCAL_RANK env var. Assuming single-node, single-GPU or manual setup."
            )
            # If not set, it might be a single GPU run or user handles DDP init outside.
            # For this auto-setup, we proceed assuming it will be set if multi-GPU DDP is intended.

        # Ensure device is set using local_rank for DDP
        if torch.cuda.is_available() and config.ddp_local_rank >= 0:
            torch.cuda.set_device(config.ddp_local_rank)
            config.device = f"cuda:{config.ddp_local_rank}"
            logger.info(f"DDP: Set device to cuda:{config.ddp_local_rank}")
        else:
            config.device = "cpu"
            logger.info("DDP: CUDA not available or local_rank not set, DDP will use CPU or fail if backend needs CUDA.")

        # Initialize process group if not already initialized
        # Check WORLD_SIZE to determine if we are in a distributed environment
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            if not dist.is_initialized():
                try:
                    # RANK and WORLD_SIZE should be set by the launcher (e.g., torchrun)
                    config.ddp_rank = int(os.environ["RANK"])
                    config.ddp_world_size = int(os.environ["WORLD_SIZE"])

                    dist.init_process_group(
                        backend=config.ddp_backend,
                        init_method="env://",  # Assumes MASTER_ADDR and MASTER_PORT are set
                        world_size=config.ddp_world_size,
                        rank=config.ddp_rank,
                    )
                    is_ddp_initialized = True
                    logger.info(
                        f"DDP: Initialized process group with backend '{config.ddp_backend}'. Rank: {config.ddp_rank}, World Size: {config.ddp_world_size}."
                    )
                except Exception as e:
                    logger.error(f"DDP: Failed to initialize process group: {e}. Running in non-DDP mode despite use_ddp=True.")
                    config.use_ddp = False  # Fallback to non-DDP
            else:
                # Already initialized, likely by launcher or another part of the script
                config.ddp_rank = dist.get_rank()
                config.ddp_world_size = dist.get_world_size()
                is_ddp_initialized = True  # Mark as initialized from our perspective
                logger.info(f"DDP: Process group already initialized. Rank: {config.ddp_rank}, World Size: {config.ddp_world_size}.")
        else:
            # Not a distributed environment (e.g. WORLD_SIZE=1 or not set)
            logger.info("DDP: WORLD_SIZE not set or is 1. Assuming single process. DDP will not be fully activated.")
            config.use_ddp = False  # Effectively disable DDP logic if not truly distributed
            config.ddp_rank = 0
            config.ddp_world_size = 1
            config.ddp_local_rank = 0  # Ensure this is 0 for single process
            if config.device is None and torch.cuda.is_available():  # If device wasn't set by local_rank logic
                config.device = "cuda:0"
            elif config.device is None:
                config.device = "cpu"
    else:
        # Not using DDP, ensure ranks are default for non-DDP logic that might check them
        config.ddp_rank = 0
        config.ddp_world_size = 1
        config.ddp_local_rank = 0
        if config.device is None:  # Set default device if not specified and not DDP
            config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- End DDP Setup ---

    # Get the correct experiment class based on config
    try:
        ExperimentClass = get_experiment_class(config.experiment_type)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)  # Exit if experiment type is invalid

    # Initialize and run experiment
    # The ExperimentClass __init__ will call the base Experiment __init__
    # which now receives a config potentially updated with DDP rank/device info.
    try:
        experiment = ExperimentClass(config)

        # INSTEAD, CALL THE MAIN EXPERIMENT LIFECYCLE METHOD FROM THE BASE CLASS
        experiment.run() # Assuming the base Experiment class has a .run() method that calls self.execute_experiment()

        logger.info("Experiment finished successfully.")
        # Potentially print summary results here if not quiet
    except Exception as e:
        logger.error(f"Experiment execution failed: {e}", exc_info=True)
        # Ensure DDP cleanup on error before exit
        if is_ddp_initialized:
            dist.destroy_process_group()
            logger.info("DDP: Destroyed process group due to error.")
        sys.exit(1)

    # --- DDP Cleanup ---
    if is_ddp_initialized:
        dist.destroy_process_group()
        logger.info("DDP: Destroyed process group successfully.")
    # --- End DDP Cleanup ---

    # cleanup is handled by Experiment.__del__ (e.g. wandb.finish())


if __name__ == "__main__":
    cli_main()

# Removed old AlignmentExperiment methods that are now part of subclasses:
# - run_progressive_dropout
# - run_eigenvector_dropout
# - run_layer_isolated_experiment
# Modified main() -> execute_experiment() and added abstract run()
# Updated cli_main() to use the factory function get_experiment_class()
