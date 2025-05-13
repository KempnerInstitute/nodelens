"""
Base experiment class for alignment experiments.

This module provides the foundation for all alignment experiments,
handling common operations like configuration loading, checkpointing,
and result management.
"""

import argparse
import datetime
import json
import logging
import os
import pickle
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from alignment.config import Config, ExperimentConfig, WandbConfig
from alignment.datasets import DataSet, load_dataset
from alignment.models import load_model, load_model_family
from alignment.metrics import get_metric

# Update imports to use the new module structure
from alignment.utils.core import (
    setup_logging,
    timer,
    debug,
    to_numpy,
    to_tensor,
    check_iterable,
    ensure_device,
    timed
)

from alignment.utils.plotting import (
    plot_dropout_results,
    plot_experiment_summary,
    plot_dropout_comparison,
    log_plots_to_wandb
)

logger = logging.getLogger(__name__)


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(self, config: Union[Dict, str, DictConfig, ExperimentConfig], 
                 working_dir: Optional[str] = None, 
                 setup_logger: bool = True):
        """
        Initialize an experiment.
        
        Args:
            config: Configuration in dict, filepath, omegaconf, or ExperimentConfig format
            working_dir: Directory to store results in
            setup_logger: Whether to set up logging
        """
        self.start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        if isinstance(config, str):
            if os.path.isfile(config):
                self.config = Config.load(config)
                self.config_path = os.path.basename(config)
                logger.info(f"Loaded config from file: {self.config_path}")
            else:
                raise ValueError(f"Config file {config} not found")
        elif isinstance(config, (dict, DictConfig)):
            logger.info(f"Creating config from dict-like object: {type(config)}")
            self.config = Config.from_dict(config)
            self.config_path = "config_dict"
        elif isinstance(config, ExperimentConfig):
            # Already an ExperimentConfig instance, just use it directly
            logger.info(f"Using provided ExperimentConfig directly: {type(config)}")
            self.config = config
            self.config_path = getattr(config, "config_path", "config_object")
        else:
            logger.error(f"Unsupported config type: {type(config)}")
            raise ValueError(f"Unsupported config type: {type(config)}")
            
        # Set up working directory
        if working_dir is None:
            # Default to a directory named with config and timestamp
            base_name = self.get_basename()
            timestamp = self.start_time
            working_dir = os.path.join(
                "results", f"{base_name}_{timestamp}"
            )
        
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_path = os.path.join(self.working_dir, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.output_path = os.path.join(self.working_dir, "outputs")
        os.makedirs(self.output_path, exist_ok=True)
        
        self.figure_path = os.path.join(self.working_dir, "figures")
        os.makedirs(self.figure_path, exist_ok=True)
        
        # Set up logging
        if setup_logger:
            # Create log directory if needed
            log_dir = os.path.join(self.working_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # Convert OmegaConf to dict for logging setup, handle if config is already dict (e.g. in tests)
            config_dict_for_logging = OmegaConf.to_container(self.config, resolve=True) if isinstance(self.config, (DictConfig, ListConfig)) else self.config
            
            # Ensure debug_mode is a boolean for force_debug
            debug_mode_val = False
            if isinstance(config_dict_for_logging, dict):
                debug_mode_val = config_dict_for_logging.get("debug_mode", False)
            elif hasattr(config_dict_for_logging, "debug_mode"): # For dataclass instances not yet dict
                debug_mode_val = config_dict_for_logging.debug_mode

            setup_logging(
                config=config_dict_for_logging, 
                log_file_path=os.path.join(log_dir, "experiment.log"),
                force_debug=bool(debug_mode_val) # Pass debug_mode to force_debug, ensure boolean
            )
            
            logger.info(f"Initialized experiment in {self.working_dir}")
        
        # Initialize random seeds
        self._set_random_seeds()
        
        # Save config
        self._save_config()
        
        # Unified W&B setup call
        self.wandb_run = None # Initialize attribute
        self._setup_wandb()      # Uses self.config.wandb and self.config.experiment_name
        
        logger.info(f"Initialized experiment in {self.working_dir}")
    
    def get_basename(self) -> str:
        """Get a base name for the experiment derived from the config."""
        if hasattr(self.config, "experiment_name"):
            return self.config.experiment_name
        elif hasattr(self, "config_path"):
            return os.path.splitext(os.path.basename(self.config_path))[0]
        else:
            return "experiment"
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility if a seed is provided in the config."""
        # Check if seed is present and is not None
        if hasattr(self.config, "seed") and self.config.seed is not None:
            seed = self.config.seed
            try:
                # Ensure seed can be converted to int, though Optional[int] should handle it
                seed_int = int(seed)
                random.seed(seed_int)
                np.random.seed(seed_int)
                torch.manual_seed(seed_int)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_int)
                logger.info(f"Set random seed to {seed_int}")
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid seed value: {seed}. Must be an integer. Seeds will not be set. Error: {e}")
        else:
            logger.info("No specific seed provided or seed is null. Using random initialization.")
    
    def _save_config(self) -> None:
        """Save the configuration to a file."""
        config_path = os.path.join(self.working_dir, "config.yaml")
        try:
            # Convert to dictionary if it's a Config object
            if hasattr(self.config, 'to_dict'):
                # Use the to_dict method if available
                config_dict = self.config.to_dict()
                with open(config_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                # Use OmegaConf for other types
                with open(config_path, "w") as f:
                    OmegaConf.save(self.config, f)
            
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.warning(f"Failed to save config to {config_path}: {str(e)}")
            # Try a simpler approach
            try:
                # Just save the basic attributes
                simple_config = {
                    k: v for k, v in vars(self.config).items()
                    if not k.startswith('_') and not callable(v)
                }
                with open(config_path, "w") as f:
                    yaml.dump(simple_config, f, default_flow_style=False)
                logger.info(f"Saved simplified config to {config_path}")
            except Exception as e2:
                logger.error(f"Failed to save even simplified config: {str(e2)}")
    
    def _setup_wandb(self) -> None:
        """Set up Weights & Biases for experiment tracking if configured."""
        if not (hasattr(self.config, "wandb") and self.config.wandb and isinstance(self.config.wandb, WandbConfig) and self.config.wandb.use_wandb):
            logger.info("W&B usage not specified or disabled in config.wandb.use_wandb")
            return
            
        if not WANDB_AVAILABLE:
            logger.warning("wandb library not installed/found, skipping wandb initialization.")
            return
        
        config_dict_for_wandb = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        experiment_name_for_wandb = getattr(self.config, "experiment_name", self.get_basename())
        project_name = self.config.wandb.wandb_project
        entity = self.config.wandb.wandb_entity
        
        if entity and entity.lower() in ["none", "null", "your_wandb_entity"]:
            entity = None # Let W&B client use default entity (requires user to be logged in)

        try:
            self.wandb_run = wandb.init(
                project=project_name,
                entity=entity,
                config=config_dict_for_wandb,
                name=experiment_name_for_wandb,
                dir=self.working_dir, # Save W&B files locally within experiment results directory
                reinit=True, 
                settings=wandb.Settings(start_method="thread")
            )
            if self.wandb_run:
                logger.info(f"W&B run initialized: {self.wandb_run.url}. Name: {self.wandb_run.name}, Project: {project_name}, Entity: {entity or self.wandb_run.entity}")
            else:
                logger.error("wandb.init() returned None, W&B run failed to initialize properly.")
        except Exception as e:
            logger.error(f"Failed to initialize Weights & Biases: {e}", exc_info=True)
            self.wandb_run = None # Ensure it's None on failure

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Returns:
            Dictionary of results
        """
        pass
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.pkl") -> str:
        """
        Save experiment results to a file.
        
        Args:
            results: Dictionary of results to save
            filename: Name of the file to save to
            
        Returns:
            Path to the saved file
        """
        filepath = os.path.join(self.output_path, filename)
        
        # Save as pickle for complete object serialization
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
        
        # Also save as JSON for human readability if possible
        try:
            # Convert numpy arrays and tensors to lists
            def convert_for_json(obj):
                if isinstance(obj, (np.ndarray, np.number)):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_for_json(item) for item in obj)
                else:
                    return obj
            
            json_results = convert_for_json(results)
            json_path = os.path.join(self.output_path, f"{os.path.splitext(filename)[0]}.json")
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save results as JSON: {str(e)}")
        
        logger.info(f"Saved results to {filepath}")
        return filepath
    
    def load_results(self, filename: str = "results.pkl") -> Dict[str, Any]:
        """
        Load experiment results from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Dictionary of results
        """
        filepath = os.path.join(self.output_path, filename)
        
        if not os.path.isfile(filepath):
            raise ValueError(f"Results file {filepath} not found")
        
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        
        logger.info(f"Loaded results from {filepath}")
        return results
    
    def cleanup(self) -> None:
        """Clean up resources used by the experiment."""
        if self.wandb_run:
            self.wandb_run.finish()
            logger.info("W&B run finished.")
        # logger.info("Experiment cleanup complete") # Already logged by run usually
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass 
