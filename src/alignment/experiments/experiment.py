"""
Base experiment module for alignment experiments.

This module defines an abstract base class for all alignment experiments.
This class provides common functionality for experiment lifecycle, checkpointing, 
result handling and visualization.
"""

import os
import sys
import datetime
import gc
import glob
import logging
import pickle
import shutil
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignment.config import BaseConfig, ModelConfig
from alignment.datasets import get_dataset
from alignment.metrics import AlignmentMetric
from alignment.models import create_model, get_model_constructor
from alignment.training import evaluate_model, load_checkpoint, train_model
from alignment.utils import (
    get_git_revision_hash,
    initialize_logger,
    init_seed,
    save_pickle,
    setup_logging,
)

logger = logging.getLogger(__name__)


class Experiment(ABC):
    """Base class for all alignment experiments.

    This class contains common functionality for:
    - Experiment lifecycle management
    - Checkpointing and storage
    - Results handling and visualization

    It is designed to be subclassed for specific experiment implementations.
    Subclasses need to implement:
    - get_basename: Provides a unique name for the experiment
    - prepare_path: Creates any additional directories needed
    - create_networks: Instantiates the neural networks
    - main: Main experiment logic
    - plot: Plotting functionality for experiment results
    """

    def __init__(
        self,
        config: BaseConfig,
        device: Optional[Union[str, torch.device]] = None,
        log_level: str = "INFO",
    ):
        """Initialize the experiment with the given configuration.

        Args:
            config: Configuration object containing experiment parameters
            device: Device to run the experiment on. If None, use cuda if available
            log_level: Logging level to use for this experiment
        """
        self.config = config
        self._resolved_device = None
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path = None
        self.checkpoint_path = None
        self.results_path = None
        self.figure_path = None
        self.timestamp = None
        self.timestamp_history = []

        # Setup logging
        self.log_level = log_level
        initialize_logger(self.log_level)

    def report(self) -> Dict[str, Any]:
        """Report experiment details.

        Returns:
            Dictionary with experiment details
        """
        return {
            "config": self.config.to_dict() if self.config else {},
            "timestamp": self.timestamp,
            "timestamp_history": self.timestamp_history,
            "git_hash": get_git_revision_hash(),
        }

    def register_timestamp(self) -> None:
        """Register the current timestamp."""
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp_history.append(self.timestamp)

    @property
    def device(self) -> torch.device:
        """Get the current device.

        Returns:
            torch.device: The current device
        """
        return self._resolved_device

    @device.setter
    def device(self, device_str: Union[str, torch.device]) -> None:
        """Set device and resolve to torch.device.

        Args:
            device_str: Device specification ('cuda', 'cpu', or torch.device)
        """
        if isinstance(device_str, torch.device):
            self._resolved_device = device_str
        else:
            self._resolved_device = torch.device(device_str)

        logger.info(f"Using device: {self._resolved_device}")

    @abstractmethod
    def get_basename(self) -> str:
        """Get a unique basename for the experiment.

        Returns:
            String with a unique basename for the experiment
        """
        pass

    def get_dir(self) -> str:
        """Get the output directory for this experiment.

        Returns:
            Path to the output directory
        """
        if not self.timestamp:
            self.register_timestamp()

        # Use results_path from ExperimentConfig or output_dir from BaseConfig if available
        output_dir = getattr(self.config, 'results_path', getattr(self.config, 'output_dir', 'results'))
        
        return os.path.join(
            output_dir,
            self.get_basename(),
            self.timestamp,
        )

    def setup_paths(self) -> Tuple[str, str, str, str]:
        """Set up and create directories for the experiment.

        Returns:
            Tuple of (base_path, checkpoint_path, results_path, figure_path)
        """
        path = self.get_dir()
        os.makedirs(path, exist_ok=True)

        checkpoint_path = os.path.join(path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)

        results_path = os.path.join(path, "results")
        os.makedirs(results_path, exist_ok=True)

        figure_path = os.path.join(path, "figures")
        os.makedirs(figure_path, exist_ok=True)

        self.path = path
        self.checkpoint_path = checkpoint_path
        self.results_path = results_path
        self.figure_path = figure_path

        logger.info(f"Created experiment directory: {path}")
        logger.info(f"Checkpoint directory: {checkpoint_path}")
        logger.info(f"Results directory: {results_path}")
        logger.info(f"Figure directory: {figure_path}")

        return path, checkpoint_path, results_path, figure_path

    def prepare_path(self) -> None:
        """Prepare additional directories if needed.

        This method should be overridden by subclasses if they need
        additional directories beyond the standard ones.
        """
        pass

    def save(self, filename: str = "experiment.pkl") -> str:
        """Save the experiment state.

        Args:
            filename: Filename for the saved experiment

        Returns:
            Path to the saved experiment file
        """
        if not self.path:
            self.setup_paths()

        save_path = os.path.join(self.path, filename)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Saved experiment to {save_path}")
        return save_path

    @staticmethod
    def load(path: str) -> "Experiment":
        """Load an experiment from a file.

        Args:
            path: Path to the saved experiment file

        Returns:
            Loaded experiment instance
        """
        with open(path, "rb") as f:
            experiment = pickle.load(f)

        logger.info(f"Loaded experiment from {path}")
        return experiment

    @abstractmethod
    def create_networks(self) -> Dict[str, nn.Module]:
        """Create the neural networks for this experiment.

        Returns:
            Dictionary mapping network names to nn.Module instances
        """
        pass

    @abstractmethod
    def main(self) -> Dict[str, Any]:
        """Run the main experiment logic.

        Returns:
            Dictionary with experiment results
        """
        pass

    def run(self) -> Dict[str, Any]:
        """Run the entire experiment.

        This method:
        1. Sets up paths
        2. Prepares any additional directories
        3. Sets the random seed
        4. Runs the main experiment logic
        5. Saves the experiment state

        Returns:
            Dictionary with experiment results
        """
        self.setup_paths()
        self.prepare_path()

        # Initialize random seed if available
        seed = getattr(self.config, 'seed', None)
        if seed is None and hasattr(self.config, 'training') and hasattr(self.config.training, 'seed'):
            seed = self.config.training.seed
            
        if seed is not None:
            init_seed(seed)
            logger.info(f"Initialized random seed: {seed}")

        results = self.main()
        self.save()

        return results

    @abstractmethod
    def plot(self, results: Dict[str, Any]) -> List[plt.Figure]:
        """Create plots for the experiment results.

        Args:
            results: Experiment results dictionary

        Returns:
            List of matplotlib figures
        """
        pass

    def save_figure(
        self, figure: plt.Figure, filename: str, close: bool = True
    ) -> str:
        """Save a figure to the figure directory.

        Args:
            figure: Matplotlib figure to save
            filename: Filename for the figure
            close: Whether to close the figure after saving

        Returns:
            Path to the saved figure
        """
        if not self.figure_path:
            self.setup_paths()

        filename = os.path.splitext(filename)[0]
        filepath = os.path.join(self.figure_path, f"{filename}.png")
        figure.savefig(filepath)

        if close:
            plt.close(figure)

        logger.info(f"Saved figure to {filepath}")
        return filepath
        
    def plot_ready(self, filename: str) -> None:
        """Finalize and save the current matplotlib figure.
        
        This method is provided for backward compatibility with existing
        plotting utilities that expect an Experiment object with a plot_ready method.
        It gets the current figure and saves it using save_figure.
        
        Args:
            filename: Name of the file to save the figure as
        """
        try:
            figure = plt.gcf()  # Get current figure
            self.save_figure(figure, filename)
        except Exception as e:
            logger.error(f"Error in plot_ready: {str(e)}", exc_info=True) 