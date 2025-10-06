"""
Protocol definitions for the alignment metrics framework.

These protocols define the interfaces that all implementations must follow,
ensuring consistency and enabling easy extension of the framework.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AlignmentMetric(Protocol):
    """Protocol for all alignment metrics."""

    @property
    def name(self) -> str:
        """Unique name identifier for the metric."""
        ...

    @property
    def requires_inputs(self) -> bool:
        """Whether this metric requires layer inputs."""
        ...

    @property
    def requires_weights(self) -> bool:
        """Whether this metric requires layer weights."""
        ...

    @property
    def requires_outputs(self) -> bool:
        """Whether this metric requires layer outputs."""
        ...

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
        ...

    def compute_distributed(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        world_size: int = 1,
        rank: int = 0,
        **kwargs: Any
    ) -> torch.Tensor:
        """Compute metric in distributed setting with automatic reduction."""
        ...


class ModelWrapper(Protocol):
    """Protocol for model wrappers that support alignment analysis."""

    @property
    def model(self) -> nn.Module:
        """The underlying PyTorch model."""
        ...

    @property
    def tracked_layers(self) -> List[str]:
        """List of layer names being tracked for alignment."""
        ...

    def get_layer_activations(
        self,
        inputs: torch.Tensor,
        layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get activations for specified layers.

        Args:
            inputs: Input tensor to the model
            layers: Specific layers to get activations for (None = all tracked)

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        ...

    def get_layer_weights(
        self,
        layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Get weights for specified layers."""
        ...

    def forward_with_activations(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass that also returns intermediate activations."""
        ...

    def apply_dropout_mask(
        self,
        dropout_masks: Dict[str, torch.Tensor],
        mode: str = "multiplicative"
    ) -> None:
        """Apply dropout masks to specified layers."""
        ...


class DatasetWrapper(Protocol):
    """Protocol for dataset wrappers."""

    @property
    def name(self) -> str:
        """Dataset name."""
        ...

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        ...

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Shape of a single input sample (excluding batch dimension)."""
        ...

    def get_train_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs: Any
    ) -> DataLoader:
        """Get training data loader."""
        ...

    def get_val_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        **kwargs: Any
    ) -> DataLoader:
        """Get validation data loader."""
        ...

    def get_test_loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        **kwargs: Any
    ) -> DataLoader:
        """Get test data loader."""
        ...


class Experiment(Protocol):
    """Protocol for experiments."""

    @property
    def name(self) -> str:
        """Experiment name."""
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Experiment configuration."""
        ...

    def setup(self) -> None:
        """Setup the experiment (called once before running)."""
        ...

    def run(
        self,
        models: Union[nn.Module, List[nn.Module]],
        dataset: DatasetWrapper,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Run the experiment.

        Args:
            models: Model or list of models to experiment on
            dataset: Dataset to use
            **kwargs: Additional experiment-specific parameters

        Returns:
            Dictionary of experiment results
        """
        ...

    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """Save experiment results."""
        ...

    def load_results(self, path: str) -> Dict[str, Any]:
        """Load experiment results."""
        ...


class MetricAggregator(Protocol):
    """Protocol for metric aggregation strategies."""

    @property
    def name(self) -> str:
        """Aggregator name."""
        ...

    def aggregate(
        self,
        metrics: Dict[str, torch.Tensor],
        mode: str = "layer",
        **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Aggregate metrics across layers or other dimensions.

        Args:
            metrics: Dictionary of metric values by layer
            mode: Aggregation mode ("layer", "global", "network", etc.)
            **kwargs: Additional aggregation parameters

        Returns:
            Aggregated metrics
        """
        ...


class ResultReporter(Protocol):
    """Protocol for result reporting and visualization."""

    @property
    def name(self) -> str:
        """Reporter name."""
        ...

    def report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Generate report from results.

        Args:
            results: Experiment results to report
            output_path: Optional path to save report
            **kwargs: Additional reporting parameters
        """
        ...

    def visualize(
        self,
        results: Dict[str, Any],
        plot_type: str,
        **kwargs: Any
    ) -> Any:
        """Generate visualizations from results."""
        ...
