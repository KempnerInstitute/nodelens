"""
Base classes for the alignment metrics framework.

These classes provide common functionality and ensure consistency
across all implementations.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Base class for all alignment metrics."""

    def __init__(self, name: Optional[str] = None, **config: Any):
        """
        Initialize the metric.

        Args:
            name: Optional custom name for the metric
            **config: Metric-specific configuration
        """
        self._name = name or self.__class__.__name__
        self.config = config
        self._force_cpu_for_large_ops = config.get('force_cpu_for_large_ops', True)
        self._cpu_threshold = config.get('cpu_threshold', 1e8)  # 100M elements

    @property
    def name(self) -> str:
        """Unique name identifier for the metric."""
        return self._name

    @property
    @abstractmethod
    def requires_inputs(self) -> bool:
        """Whether this metric requires layer inputs."""
        pass

    @property
    @abstractmethod
    def requires_weights(self) -> bool:
        """Whether this metric requires layer weights."""
        pass

    @property
    @abstractmethod
    def requires_outputs(self) -> bool:
        """Whether this metric requires layer outputs."""
        pass

    @abstractmethod
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute the metric values.

        Args:
            inputs: Input activations to the layer [batch_size, input_features]
            weights: Layer weights [output_features, input_features]
            outputs: Output activations from the layer [batch_size, output_features]
            **kwargs: Additional metric-specific parameters

        Returns:
            Tensor of metric values, typically [output_features] for per-node metrics
        """
        pass

    def compute_distributed(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        world_size: int = 1,
        rank: int = 0,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute metric in distributed setting with automatic reduction.

        Args:
            inputs: Input activations (already distributed)
            weights: Layer weights
            outputs: Output activations (already distributed)
            world_size: Number of distributed processes
            rank: Rank of current process
            **kwargs: Additional parameters

        Returns:
            Metric values with proper distributed reduction
        """
        # Compute local metric values
        local_values = self.compute(inputs, weights, outputs, **kwargs)

        if world_size > 1 and dist.is_initialized():
            # Perform all-reduce to aggregate across processes
            dist.all_reduce(local_values, op=dist.ReduceOp.SUM)
            local_values = local_values / world_size

        return local_values

    def _should_use_cpu(self, *tensors: torch.Tensor) -> bool:
        """Check if computation should be moved to CPU based on tensor sizes."""
        if not self._force_cpu_for_large_ops:
            return False

        total_elements = sum(t.numel() for t in tensors if t is not None)
        return total_elements > self._cpu_threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseModel(ABC):
    """Base class for model wrappers."""

    def __init__(
        self,
        model: nn.Module,
        tracked_layers: Optional[List[str]] = None,
        **config: Any
    ):
        """
        Initialize the model wrapper.

        Args:
            model: PyTorch model to wrap
            tracked_layers: List of layer names to track
            **config: Additional configuration
        """
        self._model = model
        self._tracked_layers = tracked_layers or []
        self.config = config
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

        if self._tracked_layers:
            self._register_hooks()

    @property
    def model(self) -> nn.Module:
        """The underlying PyTorch model."""
        return self._model

    @property
    def tracked_layers(self) -> List[str]:
        """List of layer names being tracked for alignment."""
        return self._tracked_layers

    def _register_hooks(self) -> None:
        """Register forward hooks for activation collection."""
        for name, module in self._model.named_modules():
            if name in self._tracked_layers:
                hook = module.register_forward_hook(
                    self._create_activation_hook(name)
                )
                self._hooks.append(hook)

    def _create_activation_hook(self, layer_name: str):
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            self._activation_cache[layer_name] = output.detach()
        return hook

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_layer_activations(
        self,
        inputs: torch.Tensor,
        layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Get activations for specified layers."""
        layers = layers or self._tracked_layers

        # Clear previous activations
        self._activation_cache.clear()

        # Forward pass to collect activations
        with torch.no_grad():
            _ = self._model(inputs)

        # Return requested activations
        return {
            layer: self._activation_cache[layer]
            for layer in layers
            if layer in self._activation_cache
        }

    @abstractmethod
    def get_layer_weights(
        self,
        layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Get weights for specified layers."""
        pass

    def forward_with_activations(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass that also returns intermediate activations."""
        self._activation_cache.clear()
        outputs = self._model(inputs)
        activations = self._activation_cache.copy()
        return outputs, activations

    def __del__(self):
        """Clean up hooks when object is destroyed."""
        self._clear_hooks()


class BaseDataset(ABC):
    """Base class for dataset wrappers."""

    def __init__(
        self,
        name: str,
        data_path: Optional[str] = None,
        **config: Any
    ):
        """
        Initialize the dataset wrapper.

        Args:
            name: Dataset name
            data_path: Path to dataset files
            **config: Additional configuration
        """
        self._name = name
        self._data_path = data_path
        self.config = config

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._name

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """Shape of a single input sample (excluding batch dimension)."""
        pass

    @abstractmethod
    def get_train_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs: Any
    ) -> DataLoader:
        """Get training data loader."""
        pass

    @abstractmethod
    def get_val_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        **kwargs: Any
    ) -> DataLoader:
        """Get validation data loader."""
        pass

    @abstractmethod
    def get_test_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        **kwargs: Any
    ) -> DataLoader:
        """Get test data loader."""
        pass


class BaseExperiment(ABC):
    """Base class for experiments."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        output_dir: Optional[str] = None
    ):
        """
        Initialize the experiment.

        Args:
            name: Experiment name
            config: Experiment configuration
            output_dir: Directory for saving results
        """
        self._name = name
        self._config = config
        self._output_dir = Path(output_dir) if output_dir else Path("results")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        """Experiment configuration."""
        return self._config

    @abstractmethod
    def setup(self) -> None:
        """Setup the experiment (called once before running)."""
        pass

    @abstractmethod
    def run(
        self,
        models: Union[nn.Module, List[nn.Module]],
        dataset: BaseDataset,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run the experiment."""
        pass

    def save_results(self, results: Dict[str, Any], filename: str = "results.json") -> None:
        """Save experiment results."""
        output_path = self._output_dir / filename

        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved results to {output_path}")

    def load_results(self, filename: str = "results.json") -> Dict[str, Any]:
        """Load experiment results."""
        input_path = self._output_dir / filename

        with open(input_path, 'r') as f:
            results = json.load(f)

        return results

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj

    def save_checkpoint(
        self,
        models: Union[nn.Module, List[nn.Module]],
        epoch: int,
        **additional_state: Any
    ) -> None:
        """Save a checkpoint during experiment."""
        checkpoint_dir = self._output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        if not isinstance(models, list):
            models = [models]

        state = {
            'epoch': epoch,
            'models': [m.state_dict() for m in models],
            'config': self._config,
            **additional_state
        }

        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(
        self,
        models: Union[nn.Module, List[nn.Module]],
        epoch: int
    ) -> Dict[str, Any]:
        """Load a checkpoint."""
        checkpoint_path = self._output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path)

        if not isinstance(models, list):
            models = [models]

        for model, state_dict in zip(models, state['models']):
            model.load_state_dict(state_dict)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return state
