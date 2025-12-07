"""
Alignment-based pruning strategies.

This module implements pruning based on alignment metrics like Rayleigh quotient,
allowing pruning decisions to be guided by neuron-input alignment measures.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...metrics import get_metric
from ..base import BasePruningStrategy

logger = logging.getLogger(__name__)


class AlignmentPruning(BasePruningStrategy):
    """
    Alignment-based pruning strategy.

    This strategy prunes based on alignment metrics between neurons and their inputs.
    Since alignment metrics are computed per neuron, this naturally supports
    structured pruning (removing entire neurons/channels).

    For structured pruning (default):
        - Removes entire neurons based on their alignment scores
        - All weights connected to pruned neurons are removed

    For unstructured pruning:
        - Distributes neuron scores to individual weights
        - Less meaningful since alignment is a neuron-level property

    Examples:
        >>> from alignment.pruning.strategies import AlignmentPruning
        >>> from alignment.pruning import PruningConfig
        >>>
        >>> # Structured pruning - remove entire neurons with low alignment
        >>> config = PruningConfig(
        ...     amount=0.5,
        ...     pruning_mode='low',
        ...     structured=True  # Default for alignment
        ... )
        >>> strategy = AlignmentPruning(metric='rayleigh_quotient', config=config)
        >>>
        >>> # Need to provide inputs for alignment computation
        >>> inputs = torch.randn(batch_size, input_dim)
        >>> mask = strategy.prune(layer, inputs=inputs)
    """

    def __init__(self, metric: str = "rayleigh_quotient", config=None, **metric_kwargs):
        """
        Initialize alignment-based pruning strategy.

        Args:
            metric: Name of alignment metric to use
                Options: 'rayleigh_quotient', 'mutual_information', 'cka',
                        'weight_cosine_similarity', 'gradient_similarity'
            config: Pruning configuration. Note: structured=True is recommended
                    since alignment is a neuron-level property
            **metric_kwargs: Additional arguments for the metric
        """
        super().__init__(config)
        self.metric_name = metric
        self.metric_kwargs = metric_kwargs

        # Default to structured pruning for alignment-based methods
        if config and not config.structured:
            logger.info("AlignmentPruning defaulting to structured=True (neuron pruning)")
            config.structured = True

        # Initialize the metric
        try:
            metric_instance = get_metric(metric, **metric_kwargs)
            if metric_instance is None:
                raise ValueError(f"Metric '{metric}' not found in registry")
            # Instantiate the metric with any kwargs
            self.metric = metric_instance
        except Exception as e:
            logger.error(f"Failed to initialize metric {metric}: {e}")
            raise

    def compute_importance_scores(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute importance scores based on alignment metrics.

        For structured pruning (default):
            Returns neuron-wise scores that will be used to prune entire neurons.

        For unstructured pruning:
            Expands neuron scores to individual weights (less meaningful).

        Args:
            module: Module to compute scores for
            inputs: Input activations to the module (required)
            **kwargs: Additional arguments

        Returns:
            Tensor of importance scores
            - Structured: Shape [num_output_neurons]
            - Unstructured: Shape matching module weights

        Raises:
            ValueError: If inputs are not provided or module has no weights
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")

        if inputs is None:
            raise ValueError("AlignmentPruning requires inputs to compute alignment. " "Pass inputs to the prune() method.")

        weights = module.weight.data

        # Compute alignment scores (neuron-wise)
        # Shape: [num_output_neurons]
        # Pass kwargs (e.g., targets for conditional metrics)
        alignment_scores = self.metric.compute(inputs=inputs, weights=weights, **kwargs)

        # Ensure alignment_scores is on the same device as weights
        if alignment_scores.device != weights.device:
            alignment_scores = alignment_scores.to(weights.device)

        # For structured pruning, return neuron-wise scores directly
        # The base class will handle creating masks for entire neurons
        if self.config.structured:
            return alignment_scores

        # For unstructured pruning, expand to weight-wise scores
        # Note: This is less meaningful since alignment is inherently per-neuron
        if len(weights.shape) == 2:  # Linear layer
            # Each weight in a neuron gets the same importance score
            importance = alignment_scores.unsqueeze(1).expand_as(weights)

        elif len(weights.shape) >= 3:  # Conv layer
            # Each weight in a channel gets the same importance score
            out_channels = weights.shape[0]
            importance = alignment_scores.view(out_channels, 1, 1, 1)
            importance = importance.expand_as(weights)

        else:
            raise ValueError(f"Unsupported weight shape: {weights.shape}")

        return importance

    def prune(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, amount: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Prune module based on alignment scores.

        Overrides base prune() to ensure structured pruning works correctly
        for neuron-based alignment scores.

        Args:
            module: Module to prune
            inputs: Input activations (required)
            amount: Fraction to prune
            **kwargs: Additional arguments

        Returns:
            Pruning mask
        """
        # Get importance scores
        importance_scores = self.compute_importance_scores(module, inputs, **kwargs)

        # For structured pruning with neuron-wise scores
        if self.config.structured and importance_scores.dim() == 1:
            # Create mask with special handling for neuron-wise scores
            weights = module.weight.data

            # Determine which neurons to keep/prune
            amount = amount if amount is not None else self.config.amount
            k = int(amount * importance_scores.numel())

            if k == 0:
                return torch.ones_like(weights)

            # Handle different pruning modes using topk for exact k selection
            # This avoids non-monotonic behavior from ties at threshold values
            keep_mask = torch.ones(importance_scores.numel(), dtype=torch.bool, device=importance_scores.device)
            
            if self.config.pruning_mode == "random":
                # Random selection of neurons to prune
                indices_to_prune = torch.randperm(importance_scores.numel(), device=importance_scores.device)[:k]
            elif self.config.pruning_mode == "low":
                # Prune k neurons with LOWEST scores
                _, indices_to_prune = torch.topk(importance_scores, k, largest=False)
            else:  # 'high' mode
                # Prune k neurons with HIGHEST scores
                _, indices_to_prune = torch.topk(importance_scores, k, largest=True)
            
            keep_mask[indices_to_prune] = False

            # Expand mask to all weights in the neuron/channel
            if len(weights.shape) == 2:  # Linear
                mask = keep_mask.unsqueeze(1).expand_as(weights).float()
            else:  # Conv
                mask = keep_mask.view(-1, 1, 1, 1).expand_as(weights).float()

            # Apply the mask
            self.apply_pruning(module, mask)
            return mask

        # For unstructured pruning, use base class implementation
        mask = self.create_pruning_mask(importance_scores, amount)
        self.apply_pruning(module, mask)
        return mask


class HybridPruning(BasePruningStrategy):
    """
    Hybrid pruning strategy combining magnitude and alignment information.

    This strategy combines traditional magnitude-based importance with
    alignment metrics for more informed pruning decisions.

    Examples:
        >>> from alignment.pruning.strategies import HybridPruning
        >>>
        >>> # Combine magnitude and Rayleigh quotient
        >>> strategy = HybridPruning(
        ...     alignment_metric='rayleigh_quotient',
        ...     alpha=0.5  # Equal weighting
        ... )
        >>> mask = strategy.prune(layer, inputs=inputs, amount=0.5)
    """

    def __init__(self, alignment_metric: str = "rayleigh_quotient", alpha: float = 0.5, config=None, **metric_kwargs):
        """
        Initialize hybrid pruning strategy.

        Args:
            alignment_metric: Name of alignment metric to use
            alpha: Weight for alignment score (1-alpha for magnitude)
                0 = pure magnitude, 1 = pure alignment
            config: Pruning configuration
            **metric_kwargs: Additional arguments for the metric
        """
        super().__init__(config)
        self.alignment_metric_name = alignment_metric
        self.alpha = alpha
        self.metric_kwargs = metric_kwargs

        # Initialize the alignment metric
        try:
            from ...metrics import get_metric

            metric_class = get_metric(alignment_metric)
            if metric_class is None:
                raise ValueError(f"Alignment metric '{alignment_metric}' not found in registry")
            # Instantiate the metric with any kwargs
            self.alignment_metric = metric_class(**metric_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize metric {alignment_metric}: {e}")
            raise

    def compute_importance_scores(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute importance scores combining magnitude and alignment.

        Args:
            module: Module to compute scores for
            inputs: Input activations (required for alignment)
            **kwargs: Additional arguments

        Returns:
            Combined importance scores
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")

        weights = module.weight.data

        # Magnitude-based importance
        magnitude_importance = weights.abs()

        # Normalize magnitude scores
        mag_min = magnitude_importance.min()
        mag_max = magnitude_importance.max()
        if mag_max > mag_min:
            magnitude_importance = (magnitude_importance - mag_min) / (mag_max - mag_min)

        if inputs is None or self.alpha == 0:
            # No inputs provided or pure magnitude
            return magnitude_importance

        # Alignment-based importance
        alignment_scores = self.alignment_metric.compute(inputs=inputs, weights=weights)

        # Ensure alignment_scores is on the same device
        if alignment_scores.device != weights.device:
            alignment_scores = alignment_scores.to(weights.device)

        # Expand alignment scores to match weight dimensions
        if len(weights.shape) == 2:  # Linear
            alignment_importance = alignment_scores.unsqueeze(1).expand_as(weights)
        elif len(weights.shape) >= 3:  # Conv
            out_channels = weights.shape[0]
            alignment_importance = alignment_scores.view(out_channels, 1, 1, 1)
            alignment_importance = alignment_importance.expand_as(weights)
        else:
            raise ValueError(f"Unsupported weight shape: {weights.shape}")

        # Normalize alignment scores
        align_min = alignment_importance.min()
        align_max = alignment_importance.max()
        if align_max > align_min:
            alignment_importance = (alignment_importance - align_min) / (align_max - align_min)

        # Combine scores
        combined_importance = self.alpha * alignment_importance + (1 - self.alpha) * magnitude_importance

        return combined_importance


class GlobalAlignmentPruning(AlignmentPruning):
    """
    Global alignment-based pruning strategy.

    This strategy collects alignment scores from all neurons across all layers,
    sorts them globally, and prunes the globally least aligned neurons.

    This is different from layer-wise pruning where each layer is pruned
    independently to achieve the same sparsity level.

    Examples:
        >>> from alignment.pruning.strategies import GlobalAlignmentPruning
        >>> from alignment.pruning import PruningConfig
        >>>
        >>> # Global pruning - removes 50% of neurons globally
        >>> config = PruningConfig(
        ...     amount=0.5,
        ...     global_pruning=True,
        ...     structured=True  # Always true for alignment
        ... )
        >>> strategy = GlobalAlignmentPruning(metric='rayleigh_quotient', config=config)
        >>>
        >>> # Need to provide inputs for each layer
        >>> masks = strategy.prune_model(model, layer_inputs_dict)
    """

    def __init__(self, metric: str = "rayleigh_quotient", config=None, **metric_kwargs):
        """Initialize global alignment pruning strategy."""
        super().__init__(metric, config, **metric_kwargs)
        # Ensure global pruning is enabled
        if self.config:
            self.config.global_pruning = True

    def prune_model(self, model: nn.Module, layer_inputs: Dict[str, torch.Tensor], amount: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Prune entire model globally based on alignment scores.

        Args:
            model: Model to prune
            layer_inputs: Dictionary mapping layer names to their input tensors
            amount: Global sparsity level to achieve

        Returns:
            Dictionary mapping layer names to their pruning masks
        """
        amount = amount if amount is not None else self.config.amount

        # Collect all alignment scores and layer info
        all_scores = []
        layer_info = []

        for name, module in model.named_modules():
            if hasattr(module, "weight") and name in layer_inputs:
                # Compute alignment scores for this layer
                inputs = layer_inputs[name]
                weights = module.weight.data

                # Get neuron-wise alignment scores
                alignment_scores = self.metric.compute(inputs=inputs, weights=weights)

                # Store scores and info
                all_scores.append(alignment_scores.cpu())
                layer_info.append(
                    {
                        "name": name,
                        "module": module,
                        "scores": alignment_scores,
                        "num_neurons": alignment_scores.numel(),
                        "weight_shape": weights.shape,
                    }
                )

        if not all_scores:
            logger.warning("No layers found for global pruning")
            return {}

        # Concatenate all scores
        global_scores = torch.cat(all_scores)

        # Find global threshold
        k = int(amount * global_scores.numel())
        if k == 0:
            return {}

        # Get indices to prune based on pruning mode
        if self.config.pruning_mode == "low":
            # Prune neurons with lowest alignment
            _, sorted_indices = torch.sort(global_scores)
            prune_indices = sorted_indices[:k]
        elif self.config.pruning_mode == "high":
            # Prune neurons with highest alignment
            _, sorted_indices = torch.sort(global_scores, descending=True)
            prune_indices = sorted_indices[:k]
        elif self.config.pruning_mode == "random":
            # Randomly prune k neurons globally
            prune_indices = torch.randperm(global_scores.numel())[:k]
        else:
            raise ValueError(f"Unknown pruning_mode: {self.config.pruning_mode}")

        # Convert global indices to per-layer masks using prefix sums (efficient)
        masks = {}
        prefix_counts = []
        running = 0
        for layer in layer_info:
            prefix_counts.append(running)
            running += layer["num_neurons"]
        prefix_counts.append(running)

        # For each layer, select indices that fall into its range
        for layer_idx, layer in enumerate(layer_info):
            start = prefix_counts[layer_idx]
            end = prefix_counts[layer_idx + 1]
            in_layer_mask = (prune_indices >= start) & (prune_indices < end)
            layer_global = prune_indices[in_layer_mask]
            local_indices = (layer_global - start).to(torch.long)

            num_neurons = layer["num_neurons"]
            layer_prune_mask = torch.zeros(num_neurons, dtype=torch.bool)
            if local_indices.numel() > 0:
                layer_prune_mask[local_indices] = True

            keep_mask = ~layer_prune_mask
            weights = layer["module"].weight
            if len(weights.shape) == 2:
                mask = keep_mask.unsqueeze(1).expand_as(weights).float()
            else:
                mask = keep_mask.view(-1, 1, 1, 1).expand_as(weights).float()

            self.apply_pruning(layer["module"], mask)
            masks[layer["name"]] = mask

            pruned_neurons = layer_prune_mask.sum().item()
            logger.info(f"Layer {layer['name']}: pruned {pruned_neurons}/{num_neurons} neurons " f"({pruned_neurons/num_neurons*100:.1f}%)")

        # Log global statistics
        total_neurons = sum(layer["num_neurons"] for layer in layer_info)
        logger.info(f"Global pruning complete: {k}/{total_neurons} neurons pruned ({amount*100:.1f}%)")

        return masks
