"""
Base classes for experiments.

This module provides the foundation for all experiment types,
handling common functionality like checkpointing, logging, and metrics.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch
import logging
from abc import ABC, abstractmethod
import json
import time
from datetime import datetime

from alignment.core.base import BaseExperiment as CoreBaseExperiment
from alignment.core.registry import get_metric, get_model, get_dataset, DATASET_REGISTRY
from alignment.models import ModelWrapper
from alignment.data.loaders import create_distributed_loader

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Experiment identification
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Model configuration
    model_name: str = "resnet18"
    model_config: Dict[str, Any] = field(default_factory=dict)
    pretrained: bool = False
    
    # Dataset configuration
    dataset_name: str = "cifar10"
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    data_path: Optional[str] = None
    
    # Training configuration
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    train_before_dropout: bool = True
    training_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Metrics configuration
    metrics: List[str] = field(default_factory=lambda: ["rayleigh_quotient"])
    metric_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tracked_layers: Optional[List[str]] = None
    scale_by_norm: bool = False  # Whether to scale alignment scores by weight norm
    force_cpu_for_large_metric_ops: bool = True  # Move large operations to CPU
    cnn_rq_aggregation_op: str = "mean"  # "mean", "max", "var", "sum" for CNN RQ
    exclude_classification_layer: bool = True  # Whether to exclude classification layer from analysis
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000
    save_best: bool = True
    
    # Logging
    log_dir: str = "./logs"
    log_interval: int = 100
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class BaseExperiment(CoreBaseExperiment):
    """
    Extended base experiment class with common functionality.
    
    This class provides:
    - Model and dataset initialization
    - Metric computation
    - Checkpointing
    - Logging
    - Result tracking
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config.name, config.to_dict())
        self._experiment_config = config
        
        # Set random seed
        self._set_seed(self._experiment_config.seed)
        
        # Initialize components
        self.model = None
        self.wrapped_model = None
        self.dataset = None
        self.data_loader = None
        self.metrics = {}
        self.results = {
            'config': config.to_dict(),
            'metrics': {},
            'checkpoints': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self._initialize_components()
    
    @property
    def config(self) -> ExperimentConfig:
        """Access the experiment configuration."""
        return self._experiment_config
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize model, dataset, and metrics."""
        # Initialize model
        self._initialize_model()
        
        # Initialize dataset
        self._initialize_dataset()
        
        # Initialize metrics
        self._initialize_metrics()
        
        logger.info(f"Initialized experiment: {self.config.name}")
    
    def _initialize_model(self):
        """Initialize model and wrapper."""
        # Check if model was already provided in config
        if hasattr(self.config, 'model') and self.config.model is not None:
            self.model = self.config.model
        else:
            # Get model from registry or create directly
            if hasattr(get_model, self.config.model_name):
                model_class = get_model(self.config.model_name)
                self.model = model_class(**self.config.model_config)
            else:
                # Try to get from torchvision
                import torchvision.models as models
                if hasattr(models, self.config.model_name):
                    model_fn = getattr(models, self.config.model_name)
                    self.model = model_fn(
                        pretrained=self.config.pretrained,
                        **self.config.model_config
                    )
                else:
                    raise ValueError(f"Unknown model: {self.config.model_name}")
        
        # Move to device
        device = torch.device(self.config.device)
        self.model = self.model.to(device)
        
        # Wrap model
        self.wrapped_model = ModelWrapper(
            self.model,
            tracked_layers=self.config.tracked_layers
        )
        
        logger.info(f"Initialized model: {self.config.model_name}")
        logger.info(f"Tracked layers: {self.wrapped_model.tracked_layers}")
    
    def _initialize_dataset(self):
        """Initialize dataset and data loader."""
        # Get dataset class from registry (not instance)
        dataset_class = DATASET_REGISTRY.get(self.config.dataset_name)
        
        # Debug logging
        logger.info(f"Creating dataset with data_path: {self.config.data_path}")
        logger.info(f"Dataset config: {self.config.dataset_config}")
        
        # Create dataset
        self.dataset = dataset_class(
            data_path=self.config.data_path,
            **self.config.dataset_config
        )
        
        # Create data loader
        self.data_loader = create_distributed_loader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Initialized dataset: {self.config.dataset_name}")
        logger.info(f"Dataset size: {len(self.dataset)}")
    
    def _initialize_metrics(self):
        """Initialize metrics."""
        from alignment.core.registry import METRIC_REGISTRY
        
        for metric_name in self.config.metrics:
            # Get the metric class from registry
            metric_class = METRIC_REGISTRY.get(metric_name)
            metric_config = self.config.metric_configs.get(metric_name, {})
            
            # Add global metric options if not already specified
            if 'scale_by_norm' not in metric_config:
                metric_config['scale_by_norm'] = self.config.scale_by_norm
            if 'force_cpu' not in metric_config:
                metric_config['force_cpu'] = self.config.force_cpu_for_large_metric_ops
            if 'aggregation_op' not in metric_config and 'cnn' in metric_name.lower():
                metric_config['aggregation_op'] = self.config.cnn_rq_aggregation_op
                
            # Create metric instance
            self.metrics[metric_name] = metric_class(**metric_config)
        
        logger.info(f"Initialized metrics: {list(self.metrics.keys())}")
    
    def compute_metrics(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute all configured metrics.
        
        Args:
            inputs: Input data
            targets: Target labels (optional)
            return_all: Whether to return all metric values or just means
            
        Returns:
            Dictionary mapping metric names to layer results
        """
        results = {}
        
        # Forward pass with activation tracking
        outputs, activations = self.wrapped_model.forward_with_activations(inputs)
        
        # Get weights
        weights = self.wrapped_model.get_layer_weights()
        
        # Compute each metric
        for metric_name, metric in self.metrics.items():
            metric_results = {}
            
            for layer_name in self.wrapped_model.tracked_layers:
                # Get layer inputs and weights
                layer_inputs = activations.get(f"{layer_name}_input")
                layer_weights = weights.get(layer_name)
                
                if layer_inputs is None or layer_weights is None:
                    continue
                
                # Compute metric
                try:
                    if hasattr(metric, 'requires_outputs') and metric.requires_outputs:
                        # Some metrics need outputs
                        scores = metric.compute(
                            inputs=layer_inputs,
                            weights=layer_weights,
                            outputs=activations.get(f"{layer_name}_output")
                        )
                    else:
                        scores = metric.compute(
                            inputs=layer_inputs,
                            weights=layer_weights
                        )
                    
                    if return_all:
                        metric_results[layer_name] = scores
                    else:
                        metric_results[layer_name] = scores.mean().item()
                
                except Exception as e:
                    logger.error(f"Error computing {metric_name} for {layer_name}: {e}")
                    metric_results[layer_name] = float('nan')
            
            results[metric_name] = metric_results
        
        return results
    
    def save_checkpoint(self, step: int, metrics: Optional[Dict[str, Any]] = None):
        """
        Save experiment checkpoint.
        
        Args:
            step: Current step/iteration
            metrics: Optional metrics to save
        """
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{self.config.name}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Track in results
        self.results['checkpoints'].append({
            'step': step,
            'path': str(checkpoint_path),
            'metrics': metrics
        })
        
        logger.info(f"Saved checkpoint at step {step}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load experiment checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Log metrics.
        
        Args:
            step: Current step
            metrics: Metrics to log
        """
        # Store in results
        if step not in self.results['metrics']:
            self.results['metrics'][step] = {}
        self.results['metrics'][step].update(metrics)
        
        # Log to console
        logger.info(f"Step {step}: {metrics}")
        
        # Log to wandb if configured
        if self.config.wandb_project:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")
    
    def save_results(self):
        """Save experiment results."""
        self.results['end_time'] = datetime.now().isoformat()
        results_path = Path(self.config.log_dir) / f"{self.config.name}_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def setup(self) -> None:
        """Setup the experiment (implementation of abstract method from CoreBaseExperiment)."""
        # Setup is already done in __init__, so this is just to satisfy the abstract method
        pass
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Returns:
            Experiment results
        """
        pass 