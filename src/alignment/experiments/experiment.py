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
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from alignment.config import Config
from alignment.datasets import DataSet, load_dataset
from alignment.models import load_model, load_model_family
from Code.alignment.src.alignment.metrics_rem import get_metric

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

    def __init__(self, config: Union[Dict, str, DictConfig], 
                 working_dir: Optional[str] = None, 
                 setup_logger: bool = True):
        """
        Initialize an experiment.
        
        Args:
            config: Configuration in dict, filepath, or omegaconf format
            working_dir: Directory to store results in
            setup_logger: Whether to set up logging
        """
        self.start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        if isinstance(config, str):
            if os.path.isfile(config):
                self.config = Config.load(config)
                self.config_path = os.path.basename(config)
            else:
                raise ValueError(f"Config file {config} not found")
        elif isinstance(config, (dict, DictConfig)):
            self.config = Config.from_dict(config)
            self.config_path = "config_dict"
        else:
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
            log_file = os.path.join(self.working_dir, "experiment.log")
            setup_logging(log_file=log_file, log_level=self.config.get("log_level", "INFO"))
        
        # Initialize random seeds
        self._set_random_seeds()
        
        # Save config
        self._save_config()
        
        # Set up WandB if requested
        self._setup_wandb()
        
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
        """Set random seeds for reproducibility."""
        seed = getattr(self.config, "seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")
    
    def _save_config(self) -> None:
        """Save the configuration to a file."""
        config_path = os.path.join(self.working_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(self.config, f)
        logger.info(f"Saved config to {config_path}")
    
    def _setup_wandb(self) -> None:
        """Set up Weights & Biases for experiment tracking if configured."""
        if not hasattr(self.config, "checkpointing") or not hasattr(self.config.checkpointing, "use_wandb"):
            return
            
        if not self.config.checkpointing.use_wandb:
            return
            
        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed, skipping wandb initialization")
            return
        
        wandb_config = {
            "experiment_name": getattr(self.config, "experiment_name", self.get_basename()),
            "seed": getattr(self.config, "seed", 42),
        }
        
        # Add other relevant config items
        if hasattr(self.config, "model"):
            wandb_config["model"] = OmegaConf.to_container(self.config.model)
        if hasattr(self.config, "dataset"):
            wandb_config["dataset"] = OmegaConf.to_container(self.config.dataset)
        if hasattr(self.config, "training"):
            wandb_config["training"] = OmegaConf.to_container(self.config.training)
        
        # Start WandB run
        project_name = getattr(self.config.checkpointing, "wandb_project", "alignment")
        entity = getattr(self.config.checkpointing, "wandb_entity", None)
        
        wandb.init(
            project=project_name,
            entity=entity,
            config=wandb_config,
            name=getattr(self.config, "experiment_name", None),
            dir=self.working_dir,
        )
        
        logger.info(f"Initialized WandB for project {project_name}")
    
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
        # Close WandB if it was used
        if hasattr(self.config, "checkpointing") and hasattr(self.config.checkpointing, "use_wandb"):
            if self.config.checkpointing.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.finish()
        
        logger.info("Experiment cleanup complete")
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass 