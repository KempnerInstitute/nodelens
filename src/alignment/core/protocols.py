"""
Protocol definitions for the alignment metrics framework.

These protocols define the interfaces that all implementations must follow,
ensuring consistency and enabling easy extension of the framework.

Protocols serve as contracts - any class implementing a protocol can be used
interchangeably, enabling plugin-based extensibility.

Available Protocols:
- AlignmentMetric: Per-neuron/channel metrics (RQ, MI, redundancy, etc.)
- Analyzer: Analysis pipelines (clustering, halo analysis, etc.)
- Pruner: Pruning strategies
- Visualizer: Visualization components
- Evaluator: Model evaluation (accuracy, perplexity, etc.)
- ModelWrapper: Model wrappers for activation extraction
- DatasetWrapper: Dataset abstractions
- Experiment: Full experiment pipelines

Example - Creating a custom metric:

    from alignment.core.protocols import AlignmentMetric
    from alignment.core.registry import register_metric
    
    @register_metric("my_custom_metric", category="custom", tags=["experimental"])
    class MyCustomMetric:
        '''My custom alignment metric.'''
        
        name = "my_custom_metric"
        requires_inputs = False
        requires_weights = True
        requires_outputs = True
        
        def compute(self, outputs, weights, **kwargs):
            # Your metric computation here
            return per_neuron_scores
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, Union, runtime_checkable

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
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
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
        **kwargs: Any,
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

    def get_layer_activations(self, inputs: torch.Tensor, layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Get activations for specified layers.

        Args:
            inputs: Input tensor to the model
            layers: Specific layers to get activations for (None = all tracked)

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        ...

    def get_layer_weights(self, layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Get weights for specified layers."""
        ...

    def forward_with_activations(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass that also returns intermediate activations."""
        ...

    def apply_dropout_mask(self, dropout_masks: Dict[str, torch.Tensor], mode: str = "multiplicative") -> None:
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

    def get_train_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 4, **kwargs: Any) -> DataLoader:
        """Get training data loader."""
        ...

    def get_val_loader(self, batch_size: int, shuffle: bool = False, num_workers: int = 4, **kwargs: Any) -> DataLoader:
        """Get validation data loader."""
        ...

    def get_test_loader(self, batch_size: int, shuffle: bool = False, num_workers: int = 4, **kwargs: Any) -> DataLoader:
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

    def run(self, models: Union[nn.Module, List[nn.Module]], dataset: DatasetWrapper, **kwargs: Any) -> Dict[str, Any]:
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

    def aggregate(self, metrics: Dict[str, torch.Tensor], mode: str = "layer", **kwargs: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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

    def report(self, results: Dict[str, Any], output_path: Optional[str] = None, **kwargs: Any) -> None:
        """
        Generate report from results.

        Args:
            results: Experiment results to report
            output_path: Optional path to save report
            **kwargs: Additional reporting parameters
        """
        ...

    def visualize(self, results: Dict[str, Any], plot_type: str, **kwargs: Any) -> Any:
        """Generate visualizations from results."""
        ...


# =============================================================================
# NEW PROTOCOLS FOR ENHANCED MODULARITY
# =============================================================================

@runtime_checkable
class Analyzer(Protocol):
    """
    Protocol for analysis components (clustering, halo analysis, cross-layer, etc.).
    
    Analyzers take metric results and produce higher-level insights.
    
    Example implementations:
    - KMeansClustering: Cluster neurons by metric values
    - HaloAnalysis: Analyze cross-layer dependencies
    - SupernodeDetection: Identify outlier neurons
    """

    @property
    def name(self) -> str:
        """Analyzer name."""
        ...
    
    @property
    def requires(self) -> List[str]:
        """List of required inputs (metric names, activations, etc.)."""
        ...
    
    @property
    def provides(self) -> List[str]:
        """List of outputs this analyzer produces."""
        ...

    def analyze(
        self,
        metrics: Dict[str, Any],
        model: Optional[nn.Module] = None,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Perform analysis on metrics/activations.

        Args:
            metrics: Dictionary of metric results (per layer, per neuron)
            model: Optional model for weight-based analysis
            activations: Optional pre-computed activations
            **kwargs: Additional analyzer-specific parameters

        Returns:
            Dictionary of analysis results
        """
        ...
    
    def visualize(
        self,
        results: Dict[str, Any],
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Generate visualizations from analysis results.
        
        Args:
            results: Analysis results from analyze()
            output_dir: Directory to save figures
            **kwargs: Visualization parameters
            
        Returns:
            List of paths to generated figures
        """
        ...


@runtime_checkable
class Pruner(Protocol):
    """
    Protocol for pruning strategies.
    
    Pruners compute importance scores and apply pruning to models.
    
    Example implementations:
    - MagnitudePruning: Prune by weight magnitude
    - GradientPruning: Prune by gradient-based importance
    - AlignmentPruning: Prune by alignment metrics (RQ, MI, etc.)
    - ClusterAwarePruning: Prune while respecting cluster structure
    """

    @property
    def name(self) -> str:
        """Pruner name."""
        ...
    
    @property
    def structured(self) -> bool:
        """Whether this is structured (channel/neuron) or unstructured (weight) pruning."""
        ...

    def compute_importance(
        self,
        model: nn.Module,
        layer_name: str,
        activations: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute importance scores for neurons/channels in a layer.

        Args:
            model: The model being pruned
            layer_name: Name of the layer to compute importance for
            activations: Optional activations for this layer
            gradients: Optional gradients for this layer
            **kwargs: Additional pruner-specific parameters

        Returns:
            Tensor of importance scores [num_neurons] or [num_channels]
        """
        ...

    def select_to_prune(
        self,
        scores: torch.Tensor,
        amount: float,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Select which neurons/channels to prune based on scores.

        Args:
            scores: Importance scores from compute_importance()
            amount: Fraction or number of neurons to prune
            **kwargs: Additional selection parameters

        Returns:
            Boolean mask or indices of neurons to prune
        """
        ...

    def apply(
        self,
        model: nn.Module,
        prune_mask: Dict[str, torch.Tensor],
        **kwargs: Any
    ) -> nn.Module:
        """
        Apply pruning to the model.

        Args:
            model: Model to prune
            prune_mask: Dictionary mapping layer names to prune masks
            **kwargs: Additional parameters

        Returns:
            Pruned model
        """
        ...


@runtime_checkable
class Visualizer(Protocol):
    """
    Protocol for visualization components.
    
    Visualizers generate plots and figures from various data types.
    
    Example implementations:
    - MetricHistogramVisualizer: Plot metric distributions
    - PruningCurveVisualizer: Plot accuracy vs sparsity
    - ClusterScatterVisualizer: Plot metric space clusters
    """

    @property
    def name(self) -> str:
        """Visualizer name."""
        ...
    
    @property
    def plot_types(self) -> List[str]:
        """List of plot types this visualizer can generate."""
        ...

    def plot(
        self,
        data: Any,
        plot_type: str,
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Generate a plot.

        Args:
            data: Data to visualize (format depends on plot_type)
            plot_type: Type of plot to generate
            save_path: Optional path to save the figure
            **kwargs: Plot-specific parameters (figsize, title, etc.)

        Returns:
            Matplotlib figure or other visualization object
        """
        ...
    
    def plot_batch(
        self,
        data: Dict[str, Any],
        output_dir: str,
        **kwargs: Any
    ) -> List[str]:
        """
        Generate multiple plots from a batch of data.
        
        Args:
            data: Dictionary of data to visualize
            output_dir: Directory to save figures
            **kwargs: Common plot parameters
            
        Returns:
            List of paths to generated figures
        """
        ...


@runtime_checkable
class Evaluator(Protocol):
    """
    Protocol for model evaluation.
    
    Evaluators compute performance metrics on models.
    
    Example implementations:
    - AccuracyEvaluator: Classification accuracy
    - PerplexityEvaluator: Language model perplexity
    - BenchmarkEvaluator: Run standard benchmarks (MMLU, etc.)
    """

    @property
    def name(self) -> str:
        """Evaluator name."""
        ...
    
    @property
    def metrics(self) -> List[str]:
        """List of metrics this evaluator computes."""
        ...

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            model: Model to evaluate
            dataloader: Data to evaluate on
            device: Device to run evaluation on
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of metric name -> value
        """
        ...


@runtime_checkable
class Preprocessor(Protocol):
    """
    Protocol for data/activation preprocessing.
    
    Preprocessors transform raw data or activations before metric computation.
    
    Example implementations:
    - CNNUnfoldPreprocessor: Unfold conv activations for covariance computation
    - NormalizePreprocessor: Normalize activations
    - PatchPreprocessor: Extract patches from spatial activations
    """

    @property
    def name(self) -> str:
        """Preprocessor name."""
        ...

    def preprocess(
        self,
        data: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Preprocess data.

        Args:
            data: Input tensor to preprocess
            **kwargs: Preprocessing parameters

        Returns:
            Preprocessed tensor
        """
        ...


# =============================================================================
# BASE CLASSES (Optional abstract implementations)
# =============================================================================

class BaseMetric(ABC):
    """
    Abstract base class for metrics with common functionality.
    
    Inherit from this for convenience, or just implement the AlignmentMetric protocol.
    """
    
    name: str = "base_metric"
    requires_inputs: bool = False
    requires_weights: bool = False
    requires_outputs: bool = True
    
    @abstractmethod
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Compute the metric."""
        pass
    
    def compute_distributed(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        world_size: int = 1,
        rank: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute metric with distributed reduction."""
        result = self.compute(inputs, weights, outputs, **kwargs)
        if world_size > 1:
            import torch.distributed as dist
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
            result = result / world_size
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseAnalyzer(ABC):
    """Abstract base class for analyzers."""
    
    name: str = "base_analyzer"
    requires: List[str] = []
    provides: List[str] = []
    
    @abstractmethod
    def analyze(
        self,
        metrics: Dict[str, Any],
        model: Optional[nn.Module] = None,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Perform analysis."""
        pass
    
    def visualize(
        self,
        results: Dict[str, Any],
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """Default visualization (override for custom plots)."""
        return []


class BasePruner(ABC):
    """Abstract base class for pruning strategies."""
    
    name: str = "base_pruner"
    structured: bool = True
    
    @abstractmethod
    def compute_importance(
        self,
        model: nn.Module,
        layer_name: str,
        activations: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Compute importance scores."""
        pass
    
    def select_to_prune(
        self,
        scores: torch.Tensor,
        amount: float,
        **kwargs: Any
    ) -> torch.Tensor:
        """Select neurons to prune (default: lowest scores)."""
        n_prune = int(len(scores) * amount) if amount < 1 else int(amount)
        _, indices = torch.sort(scores)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[indices[:n_prune]] = True
        return mask
    
    def apply(
        self,
        model: nn.Module,
        prune_mask: Dict[str, torch.Tensor],
        **kwargs: Any
    ) -> nn.Module:
        """Apply pruning masks to model."""
        for name, mask in prune_mask.items():
            layer = dict(model.named_modules()).get(name)
            if layer is not None and hasattr(layer, 'weight'):
                with torch.no_grad():
                    layer.weight.data[mask] = 0
        return model


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class AnalyzerConfig:
    """Configuration for an analyzer."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class PrunerConfig:
    """Configuration for a pruner."""
    name: str
    amount: float = 0.5
    selection: str = "low"  # low, high, random
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
