"""
Experiment runner for alignment experiments.

This module provides functionality for running different types of experiments,
handling configuration, and managing results.
"""

import os
import logging
import datetime
import pickle
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import numpy as np

from alignment.models.registry import create_model
from alignment.datasets import load_dataset
from alignment.metrics import get_metric
from alignment.training import train_networks
from alignment.dropout_manager import run_progressive_dropout_experiment, run_eigenvector_dropout_experiment
from alignment.utils.plotting import plot_dropout_results, plot_experiment_summary
from alignment.utils.model_utils import enhance_cnn_configuration, _normalize_device, _ensure_model_on_device

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Handles experiment execution, results storage and visualization.
    """
    
    def __init__(self, config):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results_path = None
        self.figure_path = None
        self.weights_path = None
        
        # Get device with normalization
        if hasattr(self.config, "device") and self.config.device:
            self.device = _normalize_device(torch.device(self.config.device))
        else:
            self.device = _normalize_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
        self.metric = None
        self.networks = None
        self.results = None
        self.debug_mode = getattr(self.config, "debug_mode", False)
        
        # Log initialization info if in debug mode
        if self.debug_mode:
            logger.info(f"Initializing ExperimentRunner with debug_mode=True")
            logger.info(f"Device: {self.device}")
            logger.info(f"Config: {self.config}")
            
        # Set up paths and device
        self.setup_paths()
        
        # Set up metric based on configuration
        if hasattr(config, 'alignment_settings') and config.alignment_settings is not None and hasattr(config.alignment_settings, 'metric'): # Check for alignment_settings and then metric
            self.metric = get_metric(config.alignment_settings.metric)
            if self.debug_mode:
                logger.info(f"Using metric: {config.alignment_settings.metric}")
        elif hasattr(config, 'alignment') and hasattr(config.alignment, 'metric'): # Fallback for older config structure
            logger.warning("Accessing metric from config.alignment.metric. Please update config to use alignment_settings.metric.")
            self.metric = get_metric(config.alignment.metric)
            if self.debug_mode:
                logger.info(f"Using metric (from legacy config.alignment): {config.alignment.metric}")
    
    def setup_paths(self):
        """Set up paths for experiment outputs."""
        # Create base name for experiment
        base_name = f"alignment_{self.config.model.model_name}_{self.config.dataset.dataset_name}"
        
        # Create timestamp subdirectory if needed
        if getattr(self.config, "use_timestamp", True):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join("results", f"{base_name}_{timestamp}")
        else:
            base_dir = os.path.join("results", base_name)
        
        # Create main result directory
        os.makedirs(base_dir, exist_ok=True)
        self.results_path = base_dir
        
        # Create figure directory
        self.figure_path = os.path.join(base_dir, "figures")
        os.makedirs(self.figure_path, exist_ok=True)
        
        # Create weights directory for saved models
        self.weights_path = os.path.join(base_dir, "weights")
        os.makedirs(self.weights_path, exist_ok=True)
        
        logger.info(f"Set up paths: results={self.results_path}, figures={self.figure_path}, device={self.device}")

    def create_networks(self) -> List[nn.Module]:
        """
        Create multiple neural networks for the experiment.
        
        Returns:
            List of neural networks
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
            
            # Ensure model is on the normalized device
            _ensure_model_on_device(model, self.device)
            
            networks.append(model)
            
        logger.info(f"Created {len(networks)} models with independent initializations: {self.config.model.model_name}")
        
        return networks
    
    def run_experiment(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Run the experiment based on configuration.
        
        Returns:
            Tuple of (results, networks)
        """
        # Enhance CNN configuration if applicable
        self.config = enhance_cnn_configuration(self.config)
        
        # Create networks for the experiment
        networks = self.create_networks()
        self.networks = networks
        
        # Load dataset
        batch_size = getattr(self.config.dataset, "batch_size", 128)
        dataset = load_dataset(self.config.dataset, batch_size=batch_size)
        
        # Determine experiment type
        experiment_type = getattr(self.config, 'experiment_type', 'alignment_analysis')
        
        # Run the appropriate experiment
        if experiment_type == "alignment_analysis" or experiment_type == "alignment":
            results = self.run_alignment_analysis(networks, dataset)
            
        elif experiment_type == "progressive_dropout":
            results = self.run_progressive_dropout(networks, dataset)
            
        elif experiment_type == "eigenvector_dropout":
            results = self.run_eigenvector_dropout(networks[0], dataset)
            
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
            
        logger.info(f"Completed {getattr(self.config, 'experiment_name', experiment_type)} experiment")
        
        # Store results for later use
        self.results = results
        
        # Save configuration
        self.save_config()
        
        return results, networks
    
    def run_progressive_dropout(self, networks: List[nn.Module], dataset) -> Dict:
        """
        Run progressive dropout experiment.
        
        Args:
            networks: List of networks
            dataset: Dataset object
            
        Returns:
            Dictionary with results
        """
        # Get dropout parameters from config.pruning_settings
        pruning_config = self.config.pruning_settings
        dropout_min = pruning_config.dropout_min
        dropout_max = pruning_config.dropout_max
        num_dropout_steps = pruning_config.dropout_steps
        dropout_fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in dropout_fractions: # Ensure baseline is present
            dropout_fractions = sorted(list(set([0.0] + dropout_fractions)))
        elif len(dropout_fractions) == 1 and dropout_fractions[0] != 0.0: # handles case of single non-zero step
             dropout_fractions = [0.0] + dropout_fractions

        # Get pruning and dropout modes from config.pruning_settings
        pruning_mode = pruning_config.dropout_pruning_mode
        dropout_mode = pruning_config.dropout_mode
        
        logger.info(f"Running progressive dropout with pruning_mode={pruning_mode}, dropout_mode={dropout_mode}")
        
        # Train networks if needed
        if getattr(self.config.training, "train_before_dropout", True):
            training_history = train_networks(
                networks,
                dataset,
                num_epochs=getattr(self.config.training, "epochs", 5),
                learning_rate=getattr(self.config.training, "learning_rate", 0.001),
                device=self.device,
                show_progress=True
            )
        else:
            # Create empty training history
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': []
            }
        
        # Run the experiment
        results = run_progressive_dropout_experiment(
            networks,
            dataset,
            dropout_fractions,
            self.metric,
            self.device,
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode,
            show_progress=True,
            debug_mode=self.debug_mode
        )
        
        # Add training history to results
        results["training_history"] = training_history
        
        # Generate plots
        plot_files = plot_dropout_results(
            results, 
            save_dir=self.figure_path,
            title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}",
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode
        )
        
        # Save the plot files in results
        if plot_files:
            results["plot_files"] = plot_files
        
        # Save results to file
        self.save_results("progressive_dropout_results.pkl", results)
        
        return results
    
    def run_eigenvector_dropout(self, network: nn.Module, dataset) -> Dict:
        """
        Run eigenvector dropout experiment.
        
        Args:
            network: Network to evaluate
            dataset: Dataset object
            
        Returns:
            Dictionary with results
        """
        # Get dropout parameters from config.pruning_settings
        pruning_config = self.config.pruning_settings
        dropout_min = pruning_config.dropout_min
        dropout_max = pruning_config.dropout_max
        num_dropout_steps = pruning_config.dropout_steps
        dropout_fractions = np.linspace(dropout_min, dropout_max, num_dropout_steps).tolist()
        if 0.0 not in dropout_fractions: # Ensure baseline is present
            dropout_fractions = sorted(list(set([0.0] + dropout_fractions)))
        elif len(dropout_fractions) == 1 and dropout_fractions[0] != 0.0:
             dropout_fractions = [0.0] + dropout_fractions

        # Get dropout mode from config.pruning_settings
        dropout_mode = pruning_config.dropout_mode
        pruning_mode = pruning_config.dropout_pruning_mode # Also from pruning_settings
        
        # Run the experiment
        results = run_eigenvector_dropout_experiment(
            network,
            dataset,
            dropout_fractions,
            self.metric,
            self.device,
            dropout_mode=dropout_mode,
            pruning_mode=pruning_mode,
            show_progress=True,
            debug_mode=self.debug_mode
        )
        
        # Generate plots
        plot_files = plot_dropout_results(
            results, 
            save_dir=self.figure_path,
            title_prefix="Eigenvector Dropout",
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode
        )
        
        # Save the plot files in results
        if plot_files:
            results["plot_files"] = plot_files
        
        # Save results to file
        self.save_results("eigenvector_dropout_results.pkl", results)
        
        return results
    
    def run_alignment_analysis(self, networks: List[nn.Module], dataset) -> Dict:
        """
        Run alignment analysis with multiple experiment types.
        
        Args:
            networks: List of networks
            dataset: Dataset object
            
        Returns:
            Dictionary with results
        """
        logger.info("Running alignment analysis experiment")
        
        # Prepare results structure
        results = {
            "config": self.config,
        }
        
        # Run progressive dropout if configured
        run_prog = False
        if hasattr(self.config, 'alignment_settings') and self.config.alignment_settings is not None:
            run_prog = getattr(self.config.alignment_settings, 'run_progressive', True) # Default to True if attribute exists but not set
        elif hasattr(self.config, 'alignment') and hasattr(self.config.alignment, 'run_progressive'): # Fallback
            logger.warning("Accessing run_progressive from config.alignment. Please update config to use alignment_settings.run_progressive.")
            run_prog = self.config.alignment.run_progressive

        if run_prog:
            logger.info(f"Running progressive dropout experiment")
            results["progressive_dropout"] = self.run_progressive_dropout(networks, dataset)
        
        # Run eigenvector dropout if configured
        run_eig = False
        if hasattr(self.config, 'alignment_settings') and self.config.alignment_settings is not None:
            run_eig = getattr(self.config.alignment_settings, 'run_eigenvector', False) # Default to False
        elif hasattr(self.config, 'alignment') and hasattr(self.config.alignment, 'run_eigenvector'): # Fallback
            logger.warning("Accessing run_eigenvector from config.alignment. Please update config to use alignment_settings.run_eigenvector.")
            run_eig = self.config.alignment.run_eigenvector
            
        if run_eig:
            logger.info("Running eigenvector dropout experiment")
            results["eigenvector_dropout"] = self.run_eigenvector_dropout(networks[0], dataset)
        
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
    
    def save_results(self, filename: str, results: Dict):
        """
        Save results to file.
        
        Args:
            filename: Name of the file to save
            results: Results dictionary
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
    
    def save_config(self):
        """Save configuration to file."""
        config_file = os.path.join(self.results_path, "config.yaml")
        if hasattr(self.config, 'to_dict'):
            config_dict = self.config.to_dict()
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Saved configuration to {config_file}")


def run_experiment(config, seed=None):
    """
    Run an experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (results, networks)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logger.info(f"Set random seed to {seed}")
    
    # Create and run the experiment
    runner = ExperimentRunner(config)
    return runner.run_experiment() 