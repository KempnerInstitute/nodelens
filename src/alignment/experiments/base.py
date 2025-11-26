"""
Base classes for experiments.

This module provides the foundation for all experiment types,
handling common functionality like checkpointing, logging, and metrics.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from alignment.core.base import BaseExperiment as CoreBaseExperiment
from alignment.core.registry import DATASET_REGISTRY, get_dataset, get_metric, get_model
from alignment.dataops.loaders import create_distributed_loader
from alignment.models import ModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    # Experiment identification
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    experiment_type: str = "alignment_analysis"

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
    scheduler: Optional[str] = None
    weight_decay: float = 0.0
    momentum: float = 0.9

    # Multi-network configuration
    num_networks: int = 1

    # Training control flags
    do_train: bool = True

    # Metrics configuration
    metrics: List[str] = field(default_factory=lambda: ["rayleigh_quotient"])
    metric_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tracked_layers: Optional[List[str]] = None
    scale_by_norm: bool = False  # Whether to scale alignment scores by weight norm
    force_cpu_for_large_metric_ops: bool = True  # Move large operations to CPU
    cnn_rq_aggregation_op: str = "mean"  # "mean", "max", "var", "sum" for CNN RQ
    exclude_classification_layer: bool = True  # Whether to exclude classification layer from analysis

    # Alignment-specific configuration
    alignment_methods: List[str] = field(default_factory=lambda: ["rayleigh_quotient"])
    compute_alignment: bool = True
    save_alignment_history: bool = True
    measure_alignment_during_training: bool = True
    alignment_frequency: int = 1
    alignment_data_num_samples: int = 1
    alignment_computation_texts: List[str] = field(default_factory=list)
    alignment_composite_weights: Dict[str, float] = field(default_factory=dict)
    supernode_config: Dict[str, Any] = field(default_factory=dict)

    # CNN-specific configuration
    cnn_mode: str = "unfold"  # Options: "unfold", "patchwise", "batch_patch_combined"

    # Analysis control flags
    do_dropout_analysis: bool = False
    do_eigenfeature_analysis: bool = False
    do_pruning_experiments: bool = False

    # Dropout analysis configuration
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"  # "scaled" or "unscaled"

    # Distribution analysis
    measure_expected_distribution: bool = True
    distribution_bins: int = 50

    # Pruning configuration
    # pruning_strategies: List of metrics to use for pruning (derived from metrics.enabled)
    pruning_strategies: List[str] = field(default_factory=lambda: ["rayleigh_quotient"])
    pruning_amounts: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    # pruning_selection_mode: Can be str or List[str] - "low", "high", "random"
    pruning_selection_mode: Union[str, List[str]] = field(default_factory=lambda: ["low", "high", "random"])
    fine_tune_after_pruning: bool = True
    fine_tune_epochs: int = 5
    # pruning_alignment_metric: Backward compatibility - single metric fallback
    pruning_alignment_metric: str = "rayleigh_quotient"
    pruning_hybrid_alpha: float = 0.5
    pruning_scope: str = "layer"  # "global" or "layer"
    fine_tune_learning_rate: Optional[float] = None  # Will default to learning_rate * 0.1
    alignment_structured_pruning: bool = False  # Use structured pruning for alignment
    cascading_direction: str = "forward"  # Direction for cascading pruning
    dependency_aware_pruning: bool = False  # Propagate masks across dependent layers
    # Single-layer pruning: specify a layer name to prune only that layer
    # None = prune all layers, string = prune only that layer
    pruning_target_layer: Optional[str] = None

    # Plotting and visualization
    generate_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    visualization_options: Dict[str, Any] = field(default_factory=dict)
    
    # Post-experiment analysis (runs after experiment completes)
    # When set, AnalysisRunner generates additional visualizations from results
    post_analysis: Dict[str, Any] = field(default_factory=dict)

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000
    save_best: bool = True

    # Logging
    log_dir: str = "./logs"
    log_interval: int = 100
    plots_dir: str = "./plots"  # Directory for saving plots
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0

    # Evaluation / LLM
    do_perplexity_computation: bool = False
    evaluation_dataset: str = "wikitext"
    evaluation_num_samples: int = 100

    # SCAR / supernode-specific options for LLMs
    do_scar_metrics: bool = False  # Whether to compute SCAR-style supernode metrics (T_i, R_i, L_i)
    scar_num_samples: int = 0      # Number of calibration samples for SCAR (0 => align with alignment_data_num_samples)
    scar_max_length: int = 512     # Max sequence length for SCAR calibration passes

    # Performance optimization
    eval_batches: Optional[int] = None  # Limit evaluation to N batches (None = all)
    use_tensorized_training: bool = True  # Always enabled
    use_tensorized_pruning: bool = True   # Always enabled
    use_ultra_parallel_eval: bool = True  # Always enabled

    # Misc
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    analysis_options: Dict[str, Any] = field(default_factory=dict)
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path: Union[str, Path]):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
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
        self.results = {"config": config.to_dict(), "metrics": {}, "checkpoints": [], "start_time": datetime.now().isoformat()}

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
        if hasattr(self.config, "model") and self.config.model is not None:
            self.model = self.config.model
        else:
            # Try to get model from registry first
            try:
                from alignment.core.registry import MODEL_REGISTRY

                # Handle parameter mapping for specific models
                model_kwargs = self.config.model_config.copy()

                # Remove 'name' from kwargs if it exists to avoid conflict
                model_kwargs.pop("name", None)
                # Remove cnn_mode as it's not a model parameter but a wrapper parameter
                model_kwargs.pop("cnn_mode", None)
                # Remove other model-specific configs that don't apply to current model
                model_kwargs.pop("cnn_config", None)
                model_kwargs.pop("external_config", None)
                model_kwargs.pop("model_backend", None)

                # Special handling for MLP model
                if self.config.model_name.lower() == "mlp":
                    # Extract mlp_config parameters if present
                    if "mlp_config" in model_kwargs:
                        mlp_config = model_kwargs.pop("mlp_config")
                        # Merge mlp_config parameters into model_kwargs
                        model_kwargs.update(mlp_config)

                    # Map common parameter names
                    if "num_classes" in model_kwargs and "output_dim" not in model_kwargs:
                        model_kwargs["output_dim"] = model_kwargs.pop("num_classes")
                    # Remove parameters that MLP doesn't accept
                    for param in ["pretrained", "num_layers", "dropout", "norm_type", "use_batchnorm"]:
                        model_kwargs.pop(param, None)
                    # Set default input_dim for MNIST if using MNIST dataset
                    if self.config.dataset_name.lower() == "mnist" and "input_dim" not in model_kwargs:
                        model_kwargs["input_dim"] = 784
                    # Map activation to activation_type if present
                    if "activation" in model_kwargs and "activation_type" not in model_kwargs:
                        model_kwargs["activation_type"] = model_kwargs.pop("activation")
                    # Map dropout to dropout_rate if present
                    if "dropout" in model_kwargs and "dropout_rate" not in model_kwargs:
                        model_kwargs["dropout_rate"] = model_kwargs.pop("dropout")

                self.model = MODEL_REGISTRY.create(name=self.config.model_name, **model_kwargs)
                logger.info(f"Created model '{self.config.model_name}' from registry")
            except KeyError:
                # Model not in registry, try torchvision
                import torchvision.models as models

                if hasattr(models, self.config.model_name):
                    model_fn = getattr(models, self.config.model_name)
                    # Prepare model kwargs, avoiding duplicate 'pretrained'
                    model_kwargs = self.config.model_config.copy()
                    if "pretrained" not in model_kwargs:
                        model_kwargs["pretrained"] = self.config.pretrained
                    self.model = model_fn(**model_kwargs)
                    logger.info(f"Created model '{self.config.model_name}' from torchvision")
                else:
                    raise ValueError(f"Unknown model: {self.config.model_name}")

        # Move to device (handle "auto" device)
        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            # Update config so all code uses the resolved device
            object.__setattr__(self.config, 'device', device_str)
        device = torch.device(device_str)
        if self.config.model_name.lower() == "hf_causal_lm":
            # For HuggingFace causal LMs we may be using accelerate's device_map.
            # Only move the model if no explicit device_map was provided.
            device_map = self.config.model_config.get("device_map") or self.config.model_config.get("hf_device_map")
            if device_map is None:
                self.model = self.model.to(device)
        else:
            self.model = self.model.to(device)

        # Wrap model
        wrapper_kwargs = {"tracked_layers": self.config.tracked_layers}

        # Add CNN mode if specified
        if hasattr(self.config, "cnn_mode"):
            wrapper_kwargs["cnn_mode"] = self.config.cnn_mode

        self.wrapped_model = ModelWrapper(self.model, **wrapper_kwargs)

        logger.info(f"Initialized model: {self.config.model_name}")
        logger.info(f"Tracked layers: {self.wrapped_model.tracked_layers}")

        if self.config.model_name.lower() == "hf_causal_lm":
            model_id = self.config.model_config.get("model_id")
            if model_id is None:
                raise ValueError("model_id must be set in model_config for hf_causal_lm")
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                logger.info(f"Loaded tokenizer for HF model '{model_id}'")
            except ImportError:
                logger.warning("transformers not installed; tokenizer not loaded")
                self.tokenizer = None

    def _initialize_dataset(self):
        """Initialize dataset and data loader."""
        # Get dataset class from registry (not instance). Some experiment types
        # (e.g., LLM alignment) manage their own text datasets, so we fall back
        # gracefully if the dataset is not registered.
        try:
            dataset_class = DATASET_REGISTRY.get(self.config.dataset_name)
        except KeyError:
            if self.config.experiment_type in {"llm_alignment", "llm_supernode", "llm"}:
                logger.info(
                    f"No registry dataset found for '{self.config.dataset_name}' in "
                    f"LLM experiment '{self.config.experiment_type}'; dataset will "
                    f"be initialized by the experiment class."
                )
                self.dataset = None
                self.data_loader = None
                return
            # For non-LLM experiments, surface the original error
            raise

        # Debug logging
        logger.info(f"Creating dataset with data_path: {self.config.data_path}")
        logger.info(f"Dataset config: {self.config.dataset_config}")

        # Prepare dataset kwargs, avoiding duplicate 'data_path'
        dataset_kwargs = self.config.dataset_config.copy()
        # Remove 'name' from kwargs if it exists to avoid conflict
        dataset_kwargs.pop("name", None)
        # Remove DataLoader parameters that don't belong in dataset initialization
        dataset_kwargs.pop("batch_size", None)
        dataset_kwargs.pop("num_workers", None)
        # Remove other parameters that torchvision datasets don't accept
        dataset_kwargs.pop("augmentation", None)
        dataset_kwargs.pop("train_split", None)
        dataset_kwargs.pop("val_split", None)
        dataset_kwargs.pop("augmentation_config", None)
        dataset_kwargs.pop("normalize", None)
        # Keep download parameter as it's needed for torchvision datasets
        if self.config.data_path is not None and "data_path" not in dataset_kwargs:
            dataset_kwargs["data_path"] = self.config.data_path

        # If model (e.g., HF model) has a tokenizer, include it
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            dataset_kwargs["tokenizer"] = self.tokenizer

        # Create dataset
        self.dataset = dataset_class(**dataset_kwargs)

        # Create data loader
        self.data_loader = create_distributed_loader(
            self.dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers, pin_memory=True
        )

        logger.info(f"Initialized dataset: {self.config.dataset_name}")
        logger.info(f"Dataset size: {len(self.dataset)}")

    def _initialize_metrics(self):
        """Initialize metrics."""
        import inspect
        from alignment.core.registry import METRIC_REGISTRY

        # Combine primary metrics and alignment-specific methods so that
        # alignment-only metrics (e.g., synergy / redundancy) are also
        # instantiated and available during training.
        metric_names = set(self.config.metrics)
        alignment_methods = getattr(self.config, "alignment_methods", None)
        if alignment_methods:
            metric_names.update(alignment_methods)

        for metric_name in metric_names:
            # Get the metric class from registry
            metric_class = METRIC_REGISTRY.get(metric_name)
            metric_config = self.config.metric_configs.get(metric_name, {}).copy()

            # Get the parameters accepted by this metric's __init__
            try:
                sig = inspect.signature(metric_class.__init__)
                accepted_params = set(sig.parameters.keys()) - {"self"}
                # Check if metric accepts **kwargs
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
            except (ValueError, TypeError):
                # If we can't inspect, assume it accepts everything
                accepted_params = set()
                accepts_kwargs = True

            # Only add global options if the metric accepts them
            if accepts_kwargs or "scale_by_norm" in accepted_params:
                if "scale_by_norm" not in metric_config:
                    metric_config["scale_by_norm"] = self.config.scale_by_norm
            if accepts_kwargs or "force_cpu" in accepted_params:
                if "force_cpu" not in metric_config:
                    metric_config["force_cpu"] = self.config.force_cpu_for_large_metric_ops
            if accepts_kwargs or "aggregation_op" in accepted_params:
                if "aggregation_op" not in metric_config and "cnn" in metric_name.lower():
                    metric_config["aggregation_op"] = self.config.cnn_rq_aggregation_op

            # Filter out any config keys not accepted by this metric
            if not accepts_kwargs and accepted_params:
                metric_config = {k: v for k, v in metric_config.items() if k in accepted_params}

            # Create metric instance
            self.metrics[metric_name] = metric_class(**metric_config)

        logger.info(f"Initialized metrics: {list(self.metrics.keys())}")

    def compute_metrics(
        self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, return_all: bool = False
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
                    if hasattr(metric, "requires_outputs") and metric.requires_outputs:
                        # Some metrics need outputs
                        scores = metric.compute(
                            inputs=layer_inputs, weights=layer_weights, outputs=activations.get(f"{layer_name}_output") or activations.get(layer_name)
                        )
                    else:
                        scores = metric.compute(inputs=layer_inputs, weights=layer_weights)

                    if return_all:
                        metric_results[layer_name] = scores
                    else:
                        metric_results[layer_name] = scores.mean().item()

                except Exception as e:
                    logger.error(f"Error computing {metric_name} for {layer_name}: {e}")
                    metric_results[layer_name] = float("nan")

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
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Save checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{self.config.name}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Track in results
        self.results["checkpoints"].append({"step": step, "path": str(checkpoint_path), "metrics": metrics})

        logger.info(f"Saved checkpoint at step {step}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load experiment checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
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
        if step not in self.results["metrics"]:
            self.results["metrics"][step] = {}
        self.results["metrics"][step].update(metrics)

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
        self.results["end_time"] = datetime.now().isoformat()
        results_path = Path(self.config.log_dir) / f"{self.config.name}_results.json"

        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(self.results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved results to {results_path}")

    def _make_serializable(self, obj):
        """Convert PyTorch tensors to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(v) for v in obj)
        else:
            return obj

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
