"""
Cascading alignment-based pruning strategy.

This module implements pruning where layers are pruned sequentially,
with alignment scores recomputed after each layer to account for
the effects of previous pruning.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...metrics import get_metric
from ..base import BasePruningStrategy, PruningConfig

logger = logging.getLogger(__name__)


class CascadingAlignmentPruning(BasePruningStrategy):
    """
    Cascading alignment-based pruning strategy.

    This strategy prunes layers sequentially, recomputing alignment scores
    after each layer to account for how previous pruning affects subsequent layers.

    Key differences from standard pruning:
    1. Prunes layers in order (forward or backward)
    2. Recomputes alignment scores after each layer
    3. Later layers see the effects of earlier pruning

    Examples:
        >>> from alignment.pruning.strategies import CascadingAlignmentPruning
        >>> from alignment.pruning import PruningConfig
        >>>
        >>> config = PruningConfig(
        ...     amount=0.5,
        ...     structured=True,
        ...     pruning_mode='low'
        ... )
        >>> strategy = CascadingAlignmentPruning(
        ...     metric='rayleigh_quotient',
        ...     direction='forward',  # or 'backward'
        ...     config=config
        ... )
        >>> masks = strategy.prune_model(model, get_layer_inputs_fn)
    """

    def __init__(self, metric: str = "rayleigh_quotient", direction: str = "forward", config: Optional[PruningConfig] = None, **metric_kwargs):
        """
        Initialize cascading alignment pruning.

        Args:
            metric: Alignment metric to use
            direction: 'forward' (input→output) or 'backward' (output→input)
            config: Pruning configuration
            **metric_kwargs: Additional metric arguments
        """
        super().__init__(config)
        self.metric_name = metric
        self.direction = direction
        self.metric_kwargs = metric_kwargs

        # Default to structured pruning
        if config and not hasattr(config, "structured"):
            config.structured = True

        # Initialize metric
        try:
            metric_class = get_metric(metric)
            if metric_class is None:
                raise ValueError(f"Metric '{metric}' not found")
            self.metric = metric_class(**metric_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize metric {metric}: {e}")
            raise

    def compute_importance_scores(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute alignment scores for a single module.

        Args:
            module: Module to compute scores for
            inputs: Input activations
            **kwargs: Additional arguments

        Returns:
            Alignment scores (neuron-wise)
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")

        if inputs is None:
            raise ValueError("CascadingAlignmentPruning requires inputs")

        weights = module.weight.data

        # Compute neuron-wise alignment scores
        alignment_scores = self.metric.compute(inputs=inputs, weights=weights)

        if alignment_scores.device != weights.device:
            alignment_scores = alignment_scores.to(weights.device)

        return alignment_scores

    def prune_model(
        self, model: nn.Module, get_layer_inputs_fn: callable, amount: Optional[float] = None, exclude_layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prune model using cascading approach.

        Args:
            model: Model to prune
            get_layer_inputs_fn: Function that returns current inputs for each layer
                                Should return Dict[layer_name, tensor]
            amount: Amount to prune (overrides config)
            exclude_layers: Layer names to exclude from pruning

        Returns:
            Dictionary of masks for each layer
        """
        amount = amount if amount is not None else self.config.amount
        exclude_layers = exclude_layers or []

        # Get prunable layers in order
        layers = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) >= 2 and name not in exclude_layers:
                layers.append((name, module))

        # Reverse if backward direction
        if self.direction == "backward":
            layers = layers[::-1]

        logger.info(f"Cascading {self.direction} pruning of {len(layers)} layers")

        # Prune layers sequentially
        masks = {}

        for idx, (layer_name, module) in enumerate(layers):
            logger.info(f"Pruning layer {idx+1}/{len(layers)}: {layer_name}")

            # Get current inputs for this layer
            # This should reflect any previous pruning
            layer_inputs_dict = get_layer_inputs_fn()

            if layer_name not in layer_inputs_dict:
                logger.warning(f"No inputs found for layer {layer_name}, skipping")
                continue

            layer_inputs = layer_inputs_dict[layer_name]

            # Compute alignment scores with current network state
            alignment_scores = self.compute_importance_scores(module, layer_inputs)

            # Create mask for structured pruning
            if self.config.structured and alignment_scores.dim() == 1:
                # Determine which neurons to prune
                num_neurons = alignment_scores.numel()
                k = int(amount * num_neurons)

                if k == 0:
                    mask = torch.ones_like(module.weight)
                else:
                    # Get neurons to prune based on mode
                    if self.config.pruning_mode == "low":
                        threshold = alignment_scores.kthvalue(k).values
                        keep_mask = alignment_scores > threshold
                    else:  # 'high' mode
                        threshold = alignment_scores.kthvalue(num_neurons - k).values
                        keep_mask = alignment_scores < threshold

                    # Expand to weight dimensions
                    if len(module.weight.shape) == 2:  # Linear
                        mask = keep_mask.unsqueeze(1).expand_as(module.weight).float()
                    else:  # Conv
                        mask = keep_mask.view(-1, 1, 1, 1).expand_as(module.weight).float()
            else:
                # Unstructured pruning
                mask = self.create_pruning_mask(alignment_scores, amount)

            # Apply pruning
            self.apply_pruning(module, mask)
            masks[layer_name] = mask

            # Log statistics
            sparsity = 1 - (mask != 0).float().mean().item()
            logger.info(f"  Layer {layer_name}: {sparsity*100:.1f}% pruned")

        return masks

    def _get_layer_order(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Get layers in processing order based on direction."""
        layers = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                layers.append((name, module))

        if self.direction == "backward":
            layers = layers[::-1]

        return layers
