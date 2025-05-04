"""
Base experiment module for alignment experiments.

This module defines the abstract base class for all experiments in the alignment framework,
providing common functionality for experiment lifecycle management, checkpointing,
result handling, and visualization.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
from natsort import natsorted

from alignment.config import ExperimentConfig
from alignment.utils import setup_logging


logger = logging.getLogger(__name__)


class Experiment(ABC):
    """
    Abstract base class for all alignment experiments.
    
    This class provides the foundation for all experiment implementations,
    with standardized methods for experiment setup, execution, result handling,
    and visualization. Specific experiment types should subclass this and
    implement the abstract methods.
    
    Attributes:
        config: Configuration for the experiment
        timestamp: Experiment timestamp for unique identification
        device: Device to run the experiment on (CPU/CUDA)
        meta_args: Arguments that control experiment behavior but aren't part of the experimental parameters
        basepath: Base directory path for storing experiment results
        basename: Base name for the experiment
    """
    
    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize the experiment with the provided configuration.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        self.config = config
        self.meta_args = [
            "no_save", "just_plot", "save_networks", 
            "show_params", "show_all", "device"
        ]
        
        # Resolve device
        self.device = self._resolve_device(config.device)
        
        # Set up experiment paths
        self.basename = self.get_basename()
        self.basepath = Path(self.config.results_path) / self.basename
        
        # Validate timestamp if just plotting
        if self.config.use_timestamp and self.config.just_plot:
            assert self.config.timestamp is not None, "If use_timestamp=True and just_plot=True, need a timestamp"
        
        # Register timestamp and set up logging
        self.register_timestamp()
        setup_logging(self.get_dir() / "experiment.log")
        
        # Configure wandb if enabled
        self.wandb_run = self._configure_wandb()
        
        logger.info(f"Initialized experiment: {self.basename}")
        logger.info(f"Results will be saved to: {self.get_dir()}")
        logger.info(f"Using device: {self.device}")
    
    def _resolve_device(self, device_arg: Optional[str] = None) -> torch.device:
        """
        Resolve and validate the device to run the experiment on.
        
        Args:
            device_arg: Optional device specification
            
        Returns:
            Resolved torch.device
        """
        if device_arg is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device_arg
            
        # Validate the device string
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
        
        device = torch.device(device_str)
        
        if device.type == "cuda":
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
            logger.info(f"CUDA memory available: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        
        return device
    
    def register_timestamp(self) -> None:
        """
        Register a timestamp for the experiment for unique identification.
        
        If a timestamp is provided in the configuration, use that one.
        Otherwise, generate a new timestamp and update the configuration.
        """
        if self.config.timestamp is not None:
            self.timestamp = self.config.timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.config.use_timestamp:
                self.config.timestamp = self.timestamp
    
    def report(self, init: bool = False, args: bool = False, meta_args: bool = False) -> None:
        """
        Report experiment details for debugging and tracking.
        
        Args:
            init: Whether to report initialization details
            args: Whether to report experiment parameters
            meta_args: Whether to report meta arguments
        """
        if init:
            logger.info("Experiment details:")
            logger.info(f"basename: {self.basename}")
            logger.info(f"basepath: {self.basepath}")
            logger.info(f"experiment folder: {self.get_exp_path()}")
            logger.info(f"using device: {self.device}")
            
            if self.config.save_networks and self.config.no_save:
                logger.warning("no_save=True conflicts with save_networks=True. Nothing will be saved.")
        
        if args:
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
            for key, val in config_dict.items():
                if key in self.meta_args:
                    continue
                logger.info(f"{key}={val}")
        
        if meta_args:
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
            for key, val in config_dict.items():
                if key not in self.meta_args:
                    continue
                logger.info(f"{key}={val}")
    
    def get_dir(self, create: bool = True) -> Path:
        """
        Get the experiment directory path.
        
        Args:
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Path to the experiment directory
        """
        exp_path = self.basepath / self.get_exp_path()
        if create and not exp_path.exists():
            exp_path.mkdir(parents=True)
        return exp_path
    
    def get_exp_path(self) -> Path:
        """
        Get the experiment-specific path component.
        
        Returns:
            Experiment-specific path component
        """
        exp_path = Path("/".join(self.prepare_path()))
        if self.config.use_timestamp:
            exp_path = exp_path / self.timestamp
        return exp_path
    
    def get_path(self, name: str, create: bool = True) -> Path:
        """
        Get a path for a specific file in the experiment directory.
        
        Args:
            name: Name of the file
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Path to the file
        """
        return self.get_dir(create=create) / name
    
    def _configure_wandb(self):
        """
        Configure Weights & Biases for experiment tracking.
        
        Returns:
            Wandb run object if enabled, None otherwise
        """
        try:
            import wandb
            
            if self.config.checkpointing.use_wandb:
                wandb.login()
                run = wandb.init(
                    project=self.get_basename(),
                    name=f"{self.basename}_{self.timestamp}",
                    config=self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
                )
                
                # Set Weights & Biases to offline mode if running on certain clusters
                if str(self.basepath).startswith("/n/home"):
                    os.environ["WANDB_MODE"] = "offline"
                    logger.info("Running Weights & Biases in offline mode")
                
                return run
        except ImportError:
            logger.warning("wandb not installed, disabling Weights & Biases integration")
        except Exception as e:
            logger.warning(f"Failed to configure wandb: {str(e)}")
        
        return None
    
    @abstractmethod
    def get_basename(self) -> str:
        """
        Get the base name for the experiment.
        
        Returns:
            Base name string
        """
        pass
    
    @abstractmethod
    def prepare_path(self) -> List[str]:
        """
        Prepare the experiment path components.
        
        Returns:
            List of path components
        """
        pass
    
    @abstractmethod
    def create_networks(self) -> List[nn.Module]:
        """
        Create the neural networks for the experiment.
        
        Returns:
            List of neural network modules
        """
        pass
    
    @abstractmethod
    def main(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Main experiment execution method.
        
        Returns:
            Tuple of (results dictionary, list of networks)
        """
        pass
    
    @abstractmethod
    def plot(self, results: Dict) -> None:
        """
        Plot experiment results.
        
        Args:
            results: Dictionary containing experiment results
        """
        pass
    
    def get_prms_path(self) -> Path:
        """
        Get the path for the parameters file.
        
        Returns:
            Path to the parameters file
        """
        return self.get_dir() / "prms.pth"
    
    def get_config_path(self) -> Path:
        """
        Get the path for the configuration file.
        
        Returns:
            Path to the configuration file
        """
        return self.get_dir() / "config.json"
    
    def get_results_path(self) -> Path:
        """
        Get the path for the results file.
        
        Returns:
            Path to the results file
        """
        return self.get_dir() / "results.pth"
    
    def get_network_path(self, name: str) -> Path:
        """
        Get the path for a network checkpoint file.
        
        Args:
            name: Name of the network
            
        Returns:
            Path to the network checkpoint file
        """
        return self.get_dir() / f"{name}.pt"
    
    def get_checkpoint_path(self) -> Path:
        """
        Get the path for the experiment checkpoint file.
        
        Returns:
            Path to the checkpoint file
        """
        return self.get_dir() / "checkpoint.tar"
    
    def save_experiment(self, results: Dict) -> None:
        """
        Save experiment configuration and results.
        
        Args:
            results: Dictionary containing experiment results
        """
        if hasattr(self.config, 'to_dict'):
            # Save as JSON if the config has a to_dict method
            with open(self.get_config_path(), 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        else:
            # Fall back to torch.save for backward compatibility
            torch.save(vars(self.config), self.get_prms_path())
        
        torch.save(results, self.get_results_path())
        logger.info(f"Saved experiment configuration and results to {self.get_dir()}")
    
    def load_experiment(self, no_results: bool = False) -> Optional[Dict]:
        """
        Load experiment configuration and results.
        
        Args:
            no_results: Whether to skip loading results
            
        Returns:
            Results dictionary if no_results is False, None otherwise
        """
        config_loaded = False
        
        # Try loading from JSON first
        if self.get_config_path().exists():
            try:
                with open(self.get_config_path(), 'r') as f:
                    config_dict = json.load(f)
                
                # Update config with loaded values
                for k, v in config_dict.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)
                
                config_loaded = True
                logger.info(f"Loaded configuration from {self.get_config_path()}")
            except Exception as e:
                logger.warning(f"Failed to load config from JSON: {str(e)}")
        
        # Fall back to legacy format
        if not config_loaded and self.get_prms_path().exists():
            try:
                prms = torch.load(self.get_prms_path())
                self._update_config(prms)
                config_loaded = True
                logger.info(f"Loaded parameters from {self.get_prms_path()}")
            except Exception as e:
                logger.error(f"Failed to load parameters: {str(e)}")
        
        if not config_loaded:
            raise ValueError(f"No saved configuration found at: {self.get_config_path()} or {self.get_prms_path()}")
        
        if no_results:
            return None
        
        if not self.get_results_path().exists():
            raise ValueError(f"No saved results at: {self.get_results_path()}")
        
        results = torch.load(self.get_results_path())
        logger.info(f"Loaded results from {self.get_results_path()}")
        return results
    
    def _update_config(self, prms: Dict) -> None:
        """
        Update configuration with values from loaded parameters.
        
        Args:
            prms: Dictionary of parameters
            
        Raises:
            ValueError: If parameters contain unknown keys
        """
        config_keys = vars(self.config).keys()
        if not set(prms.keys()).issubset(config_keys):
            diff = set(prms.keys()).difference(config_keys)
            raise ValueError(f"Saved parameters contain unknown keys: {diff}")
        
        for key in config_keys:
            if key in self.meta_args:
                continue
            if key in prms and prms[key] != getattr(self.config, key):
                logger.info(f"Updating config {key} from {getattr(self.config, key)} to {prms[key]}")
                setattr(self.config, key, prms[key])
    
    def save_networks(self, nets: List[nn.Module], id: Optional[str] = None) -> None:
        """
        Save network checkpoints.
        
        Args:
            nets: List of networks to save
            id: Optional identifier to add to network names
        """
        name = f"net_{id}_" if id else "net_"
        for idx, net in enumerate(nets):
            cname = name + f"{idx}"
            torch.save(net.state_dict(), self.get_network_path(cname))
        logger.info(f"Saved {len(nets)} networks with prefix '{name}'")
    
    def load_networks(
        self, 
        nets: List[nn.Module], 
        id: Optional[str] = None,
        check_number: bool = True
    ) -> List[nn.Module]:
        """
        Load network checkpoints.
        
        Args:
            nets: List of networks to load into
            id: Optional identifier used when saving
            check_number: Whether to check that the number of saved networks matches
                the number of provided networks
                
        Returns:
            List of loaded networks
            
        Raises:
            AssertionError: If check_number is True and the number of saved networks
                doesn't match the number of provided networks
        """
        name = f"net_{id}_" if id else "net_"
        pattern = self.get_network_path(name + "*").name
        matches = natsorted([match.stem for match in self.get_dir().rglob(pattern)])
        
        if check_number:
            msg = f"# networks in checkpoint {len(matches)} != needed {len(nets)}"
            assert len(matches) == len(nets), msg
        
        for idx, match in enumerate(matches):
            if idx >= len(nets):
                break
            
            c_state = torch.load(self.get_network_path(match))
            nets[idx].load_state_dict(c_state)
        
        logger.info(f"Loaded {min(len(matches), len(nets))} networks with prefix '{name}'")
        return nets
    
    def plot_ready(self, name: str) -> None:
        """
        Finalize and save a plot.
        
        Args:
            name: Name of the plot
        """
        try:
            import matplotlib.pyplot as plt
            import wandb
            
            if not self.config.no_save:
                plt.savefig(str(self.get_path(name)))
                logger.info(f"Saved plot to {self.get_path(name)}")
            
            if self.wandb_run is not None:
                self.wandb_run.log({name: wandb.Image(plt)})
            
            if not self.config.show_all:
                plt.show()
        except ImportError as e:
            logger.warning(f"Failed to process plot: {str(e)}")
        except Exception as e:
            logger.error(f"Error in plot_ready: {str(e)}")
    
    def run(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Run the experiment.
        
        Returns:
            Tuple of (results dictionary, list of networks)
        """
        logger.info(f"Starting experiment: {self.basename}")
        
        if self.config.just_plot:
            self.plot_from_existing()
            return {}, []
        
        try:
            results, nets = self.main()
            
            if not self.config.no_save:
                self.save_experiment(results)
                if self.config.save_networks:
                    self.save_networks(nets)
            
            if not self.config.just_plot:
                self.plot(results)
            
            logger.info(f"Experiment completed successfully: {self.basename}")
            return results, nets
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            raise
    
    def plot_from_existing(self) -> None:
        """
        Load existing results and plot them.
        """
        try:
            stored = self.load_experiment(no_results=False)
            logger.info("Loaded existing results. Now plotting.")
            self.plot(stored)
        except Exception as e:
            logger.error(f"Failed to plot from existing results: {str(e)}", exc_info=True)
            raise 