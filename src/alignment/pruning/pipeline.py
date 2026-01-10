"""
Shared structured pruning pipeline.

Provides a thin wrapper around existing pruning utilities (distribution
manager, dependency-aware pruning, mask ops) so experiments can invoke
pruning consistently through configuration instead of bespoke loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .dependency_aware import DependencyAwarePruning
from .distribution import PruningDistributionManager
from ..services.mask_ops import MaskOperations

logger = logging.getLogger(__name__)


@dataclass
class PruningPipelineOptions:
    """Options controlling the pruning pipeline behaviour."""

    distribution: str = "uniform"
    dependency_aware: bool = False
    min_amount: float = 0.0
    max_amount: float = 0.95


def _ensure_tensor(scores) -> torch.Tensor:
    if isinstance(scores, torch.Tensor):
        return scores
    return torch.as_tensor(scores, dtype=torch.float32)


def _apply_masks_to_modules(layer_modules: Dict[str, nn.Module], masks: Dict[str, torch.Tensor]) -> None:
    """Apply channel masks directly to module weights/biases."""
    for layer_name, mask in masks.items():
        module = layer_modules.get(layer_name)
        if module is None or not hasattr(module, "weight"):
            continue

        mask = mask.to(module.weight.device)
        weight_mask = MaskOperations.expand_neuron_mask_to_weights(mask, module.weight.data.shape, dim=0)
        masked_weights = MaskOperations.apply_mask_to_weights(module.weight.data, weight_mask, mode="zero")

        with torch.no_grad():
            module.weight.data.copy_(masked_weights)
            if getattr(module, "bias", None) is not None and mask.numel() == module.bias.data.numel():
                bias_mask = mask.bool()
                module.bias.data[~bias_mask] = 0.0


def run_pruning_pipeline(
    model: nn.Module,
    layer_scores: Dict[str, torch.Tensor],
    *,
    layer_modules: Optional[Dict[str, nn.Module]] = None,
    target_sparsity: float,
    selection_mode: str = "low",
    options: Optional[PruningPipelineOptions] = None,
) -> Dict[str, Any]:
    """
    Execute pruning using shared infrastructure.

    Args:
        model: Model to prune.
        layer_scores: Dict mapping layer_name -> channel scores tensor.
        layer_modules: Optional mapping layer_name -> module (defaults to model.named_modules()).
        target_sparsity: Overall sparsity target (0-1).
        selection_mode: 'low', 'high', or 'random'.
        options: Additional pipeline options.

    Returns:
        Dict with 'masks' and optional stats from dependency-aware pruning.
    """
    if not layer_scores:
        logger.warning("No layer scores provided; skipping pruning")
        return {"masks": {}}

    options = options or PruningPipelineOptions()
    layer_modules = layer_modules or dict(model.named_modules())

    tensor_scores = {name: _ensure_tensor(score) for name, score in layer_scores.items()}
    layer_names = [name for name in tensor_scores.keys() if name in layer_modules]

    if not layer_names:
        logger.warning("Layer score names do not match model modules; skipping pruning")
        return {"masks": {}}

    distribution = options.distribution or "uniform"

    if options.dependency_aware:
        manager = PruningDistributionManager(
            strategy=distribution,
            target_sparsity=target_sparsity,
            min_amount=options.min_amount,
            max_amount=options.max_amount,
        )
        per_layer_amounts = manager.compute_distribution(model, layer_names, layer_scores=tensor_scores)

        dep_pruner = DependencyAwarePruning(model)
        result = dep_pruner.prune(
            tensor_scores,
            amount=target_sparsity,
            mode=selection_mode,
            per_layer_amounts=per_layer_amounts,
        )
        flat_masks = {}
        for name, data in result["masks"].items():
            output_mask = data.get("output_mask") if isinstance(data, dict) else None
            if output_mask is not None:
                flat_masks[name] = output_mask
        result["masks"] = flat_masks
        return result

    if distribution in {"global_threshold", "global"}:
        masks = MaskOperations.global_threshold_mask(tensor_scores, global_amount=target_sparsity, mode=selection_mode)
    else:
        manager = PruningDistributionManager(
            strategy=distribution,
            target_sparsity=target_sparsity,
            min_amount=options.min_amount,
            max_amount=options.max_amount,
        )
        per_layer_amounts = manager.compute_distribution(model, layer_names, layer_scores=tensor_scores)
        masks = {}
        for name in layer_names:
            amount = per_layer_amounts.get(name, target_sparsity)
            masks[name] = MaskOperations.create_structured_mask(tensor_scores[name], amount=amount, mode=selection_mode)

    _apply_masks_to_modules(layer_modules, masks)

    stats = {name: MaskOperations.get_mask_statistics(mask) for name, mask in masks.items()}
    return {"masks": masks, "stats": stats}
