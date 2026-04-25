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

import numpy as np
import torch
import torch.nn as nn

from ..services.mask_ops import MaskOperations
from .dependency_aware import DependencyAwarePruning
from .distribution import PruningDistributionManager

logger = logging.getLogger(__name__)


@dataclass
class PruningPipelineOptions:
    """Options controlling the pruning pipeline behaviour."""

    distribution: str = "uniform"
    dependency_aware: bool = False
    min_amount: float = 0.0
    max_amount: float = 0.95
    # Safety cap for per-layer sparsity when using global-threshold style distributions.
    # Set to 1.0 to disable (legacy behavior), or e.g. 0.90 to avoid pruning entire layers.
    max_per_layer_sparsity_cap: float = 1.00
    # If True, convert per-layer prune amounts into integer channel counts and
    # enforce exact global channel-budget matching (when feasible).
    enforce_exact_global_channel_budget: bool = False


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


def _layer_output_channels(module: nn.Module) -> int:
    if hasattr(module, "out_channels"):
        try:
            return int(getattr(module, "out_channels"))
        except Exception:
            return 0
    if hasattr(module, "out_features"):
        try:
            return int(getattr(module, "out_features"))
        except Exception:
            return 0
    return 0


def _allocate_exact_channel_budget(
    *,
    layer_names: list[str],
    layer_modules: Dict[str, nn.Module],
    per_layer_amounts: Dict[str, float],
    target_ratio: float,
    min_amount: float,
    max_amount: float,
    cap_amount: float,
) -> Dict[str, int]:
    """
    Convert per-layer fractional amounts into integer prune counts and enforce:
      sum(pruned_channels) == round(target_ratio * total_prunable_channels)
    whenever feasible under layer-wise min/max bounds.
    """
    min_amount = max(0.0, min(1.0, float(min_amount)))
    max_amount = max(0.0, min(1.0, float(max_amount)))
    cap_amount = max(0.0, min(1.0, float(cap_amount)))
    if max_amount < min_amount:
        max_amount = min_amount

    raw_counts: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    min_counts: Dict[str, int] = {}
    max_counts: Dict[str, int] = {}
    layer_sizes: Dict[str, int] = {}

    total_channels = 0
    for ln in layer_names:
        module = layer_modules.get(ln)
        if module is None:
            continue
        n = _layer_output_channels(module)
        if n <= 0:
            continue
        layer_sizes[ln] = n
        total_channels += n

        amount = float(per_layer_amounts.get(ln, target_ratio))
        amount = max(min_amount, min(max_amount, amount))
        raw = amount * n

        lo = int(np.floor(min_amount * n + 1e-12))
        hi = int(np.floor(min(max_amount, cap_amount) * n + 1e-12))
        hi = max(lo, min(hi, n))
        c0 = int(np.floor(raw + 1e-12))
        c0 = max(lo, min(c0, hi))

        raw_counts[ln] = raw
        counts[ln] = c0
        min_counts[ln] = lo
        max_counts[ln] = hi

    if total_channels <= 0 or not counts:
        return counts

    target_total = int(round(float(target_ratio) * float(total_channels)))
    min_total = int(sum(min_counts.values()))
    max_total = int(sum(max_counts.values()))
    target_total = max(min_total, min(target_total, max_total))

    current_total = int(sum(counts.values()))
    if current_total < target_total:
        deficit = int(target_total - current_total)
        add_order = sorted(
            counts.keys(),
            key=lambda ln: (
                float(raw_counts.get(ln, 0.0) - np.floor(raw_counts.get(ln, 0.0))),
                float(raw_counts.get(ln, 0.0)),
                int(layer_sizes.get(ln, 0)),
                ln,
            ),
            reverse=True,
        )
        while deficit > 0:
            progressed = False
            for ln in add_order:
                if deficit <= 0:
                    break
                room = int(max_counts[ln] - counts[ln])
                if room <= 0:
                    continue
                counts[ln] += 1
                deficit -= 1
                progressed = True
            if not progressed:
                break
    elif current_total > target_total:
        excess = int(current_total - target_total)
        drop_order = sorted(
            counts.keys(),
            key=lambda ln: (
                float(raw_counts.get(ln, 0.0) - np.floor(raw_counts.get(ln, 0.0))),
                float(raw_counts.get(ln, 0.0)),
                int(layer_sizes.get(ln, 0)),
                ln,
            ),
        )
        while excess > 0:
            progressed = False
            for ln in drop_order:
                if excess <= 0:
                    break
                room = int(counts[ln] - min_counts[ln])
                if room <= 0:
                    continue
                counts[ln] -= 1
                excess -= 1
                progressed = True
            if not progressed:
                break

    return counts


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
            max_per_layer_sparsity_cap=getattr(options, "max_per_layer_sparsity_cap", 1.00),
        )
        per_layer_amounts = manager.compute_distribution(model, layer_names, layer_scores=tensor_scores)
        if bool(getattr(options, "enforce_exact_global_channel_budget", False)):
            exact_counts = _allocate_exact_channel_budget(
                layer_names=layer_names,
                layer_modules=layer_modules,
                per_layer_amounts=per_layer_amounts,
                target_ratio=target_sparsity,
                min_amount=options.min_amount,
                max_amount=options.max_amount,
                cap_amount=getattr(options, "max_per_layer_sparsity_cap", 1.00),
            )
            adjusted_amounts: Dict[str, float] = {}
            for name in layer_names:
                n = _layer_output_channels(layer_modules[name])
                if n > 0 and name in exact_counts:
                    adjusted_amounts[name] = float(exact_counts[name] / float(n))
            if adjusted_amounts:
                per_layer_amounts = adjusted_amounts

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

    # Non-dependency-aware path.
    #
    # For global_threshold distribution, use the original MaskOperations.global_threshold_mask
    # which computes a single threshold across all layers and applies it directly. This is
    # the legacy behavior that Jan 20 runs used and produces the expected results.
    #
    # For other distributions (uniform, size_proportional, etc.), use the distribution
    # manager which computes per-layer amounts.
    if distribution in {"global_threshold", "global"}:
        # Legacy behavior: direct global threshold without per-layer caps.
        # This can prune entire layers if all their scores fall below threshold, but
        # it matches the original behavior that produced good results.
        masks = MaskOperations.global_threshold_mask(tensor_scores, global_amount=target_sparsity, mode=selection_mode)
    else:
        manager = PruningDistributionManager(
            strategy=distribution,
            target_sparsity=target_sparsity,
            min_amount=options.min_amount,
            max_amount=options.max_amount,
            max_per_layer_sparsity_cap=getattr(options, "max_per_layer_sparsity_cap", 1.00),
        )
        per_layer_amounts = manager.compute_distribution(model, layer_names, layer_scores=tensor_scores)
        if bool(getattr(options, "enforce_exact_global_channel_budget", False)):
            exact_counts = _allocate_exact_channel_budget(
                layer_names=layer_names,
                layer_modules=layer_modules,
                per_layer_amounts=per_layer_amounts,
                target_ratio=target_sparsity,
                min_amount=options.min_amount,
                max_amount=options.max_amount,
                cap_amount=getattr(options, "max_per_layer_sparsity_cap", 1.00),
            )
            adjusted_amounts: Dict[str, float] = {}
            for name in layer_names:
                n = _layer_output_channels(layer_modules[name])
                if n > 0 and name in exact_counts:
                    adjusted_amounts[name] = float(exact_counts[name] / float(n))
            if adjusted_amounts:
                per_layer_amounts = adjusted_amounts
        masks = {}
        for name in layer_names:
            amount = per_layer_amounts.get(name, target_sparsity)
            masks[name] = MaskOperations.create_structured_mask(tensor_scores[name], amount=amount, mode=selection_mode)

    _apply_masks_to_modules(layer_modules, masks)

    stats = {name: MaskOperations.get_mask_statistics(mask) for name, mask in masks.items()}
    return {"masks": masks, "stats": stats}
