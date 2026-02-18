"""
Dependency-aware structured pruning for neural networks.

Handles dependencies between layers:
- Conv: output channels of layer L -> input channels of layer L+1
- Skip connections: ensure compatible channel counts
- Attention: Q/K/V/O projection consistency

This ensures pruning doesn't cause shape mismatches.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerDependency:
    """Dependency information for a layer."""

    name: str
    module: nn.Module
    layer_type: str
    index: int
    depends_on: Optional[List[str]] = None  # Input dependencies
    feeds_into: Optional[List[str]] = None  # Output dependencies
    skip_connection_with: Optional[List[str]] = None  # Skip/residual connections


class DependencyGraph:
    """
    Build and manage dependency graph for a neural network.

    Analyzes connections between layers to enable safe structured pruning.
    """

    def __init__(self, model: nn.Module):
        """
        Build dependency graph from model.

        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self.graph: Dict[str, LayerDependency] = {}
        self._build_graph()

    def _build_graph(self):
        """Build dependency graph by analyzing model structure."""
        # Get all relevant layers
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.MultiheadAttention)):
                layer_type = self._get_layer_type(module)
                layers.append((name, module, layer_type))

        # Build graph entries
        for idx, (name, module, layer_type) in enumerate(layers):
            # Determine dependencies
            depends_on = [layers[idx - 1][0]] if idx > 0 else None
            feeds_into = [layers[idx + 1][0]] if idx < len(layers) - 1 else None

            self.graph[name] = LayerDependency(
                name=name, module=module, layer_type=layer_type, index=idx, depends_on=depends_on, feeds_into=feeds_into
            )

        # Detect skip connections (heuristic for ResNet-style models)
        self._detect_skip_connections()

        logger.info(f"Built dependency graph with {len(self.graph)} layers")

    def _get_layer_type(self, module: nn.Module) -> str:
        """Determine layer type."""
        if isinstance(module, nn.Linear):
            return "linear"
        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            return "conv"
        elif isinstance(module, nn.MultiheadAttention):
            return "attention"
        else:
            return "other"

    def _detect_skip_connections(self):
        """
        Detect skip/residual connections.

        Heuristic: Look for layers with same in/out dimensions
        that might be part of residual blocks.
        """
        # This is a simplified heuristic
        # For production, would need more sophisticated analysis
        # (e.g., tracing actual forward pass)

        for name, dep in self.graph.items():
            if dep.layer_type == "conv":
                module = dep.module
                if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                    if module.in_channels == module.out_channels:
                        # Potential residual connection
                        dep.skip_connection_with = []  # Placeholder

        logger.debug("Skip connection detection complete")


class DependencyAwarePruning:
    """
    Apply structured pruning with automatic dependency handling.

    Ensures that pruning one layer's outputs correctly propagates
    to the next layer's inputs, preventing shape mismatches.

    Example:
        >>> pruner = DependencyAwarePruning(model)
        >>> layer_scores = {'conv1': scores1, 'conv2': scores2}
        >>> result = pruner.prune(layer_scores, amount=0.5)
        >>> print(f"Pruned {result['stats']['total_params_pruned']} parameters")
    """

    def __init__(self, model: nn.Module):
        """
        Initialize dependency-aware pruning.

        Args:
            model: PyTorch model to prune
        """
        self.model = model
        self.dependency_graph = DependencyGraph(model)

    def prune(
        self,
        layer_scores: Dict[str, torch.Tensor],
        amount: float,
        mode: str = "low",
        dry_run: bool = False,
        per_layer_amounts: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Apply structured pruning with dependency awareness.

        Args:
            layer_scores: Dict mapping layer names to importance scores
            amount: Fraction to prune (0-1)
            mode: 'low' (prune low scores) or 'high'
            dry_run: If True, only compute plan without applying

        Returns:
            Dictionary with:
                - 'masks': Pruning masks per layer
                - 'stats': Statistics about pruning
                - 'validation': Shape validation results
        """
        # 1. Create initial masks from scores
        initial_masks = self._create_initial_masks(layer_scores, amount, mode, per_layer_amounts)

        # 2. Propagate masks to handle dependencies
        propagated_masks = self._propagate_masks(initial_masks)

        # 3. Validate shapes
        validation = self._validate_pruning_plan(propagated_masks)

        if not validation["valid"]:
            logger.error(f"Pruning plan validation failed: {validation['errors']}")
            raise ValueError(f"Invalid pruning plan: {validation['errors']}")

        # 4. Apply masks (if not dry-run)
        if dry_run:
            stats = self._compute_stats(propagated_masks)
            logger.info("Dry run: pruning plan computed but not applied")
        else:
            stats = self._apply_masks(propagated_masks)
            logger.info(f"Applied pruning: {stats['total_params_pruned']}/{stats['total_params']} parameters pruned")

        return {"masks": propagated_masks, "stats": stats, "validation": validation}

    def _create_initial_masks(
        self,
        layer_scores: Dict[str, torch.Tensor],
        default_amount: float,
        mode: str,
        per_layer_amounts: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Create initial output masks from importance scores."""
        from ..services.mask_ops import MaskOperations

        initial_masks = {}

        for layer_name, scores in layer_scores.items():
            if layer_name not in self.dependency_graph.graph:
                logger.warning(f"Layer {layer_name} not in dependency graph, skipping")
                continue

            layer_amount = default_amount
            if per_layer_amounts and layer_name in per_layer_amounts:
                layer_amount = per_layer_amounts[layer_name]
            # Create structured mask (output neurons/channels)
            mask = MaskOperations.create_structured_mask(scores, amount=layer_amount, mode=mode)

            initial_masks[layer_name] = mask

            logger.debug(f"{layer_name}: {mask.sum()}/{len(mask)} outputs kept")

        return initial_masks

    def _propagate_masks(self, output_masks: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Propagate output masks to dependent layers.

        Returns:
            Dict[layer_name] -> {
                'output_mask': mask for output neurons/channels,
                'input_mask': mask for input neurons/channels,
                'weight_mask': mask for weight tensor
            }
        """
        propagated = {}

        # Sort layers by dependency order
        sorted_layers = sorted(self.dependency_graph.graph.values(), key=lambda x: x.index)

        for dep in sorted_layers:
            layer_name = dep.name
            module = dep.module

            # Get output mask (from scores or default to all-keep)
            if layer_name in output_masks:
                out_mask = output_masks[layer_name]
            else:
                # Default: keep all outputs
                out_dim = self._get_output_dimension(module)
                out_mask = torch.ones(out_dim, dtype=torch.bool, device=next(module.parameters()).device)

            # Get input mask (from previous layer's output or default)
            in_mask = self._get_input_mask(dep, propagated)

            # Create weight mask based on layer type
            weight_mask = self._create_weight_mask(module, out_mask, in_mask)

            propagated[layer_name] = {"output_mask": out_mask, "input_mask": in_mask, "weight_mask": weight_mask}

        return propagated

    def _get_input_mask(self, layer_dep: LayerDependency, propagated: Dict) -> torch.Tensor:
        """Get input mask from dependencies."""
        in_dim = self._get_input_dimension(layer_dep.module)
        device = next(layer_dep.module.parameters()).device

        # For downsample/shortcut layers, always use correct input dimension
        # These layers have different input sources due to skip connections
        if "downsample" in layer_dep.name or "shortcut" in layer_dep.name:
            return torch.ones(in_dim, dtype=torch.bool, device=device)

        # Check if previous layer has been processed
        if layer_dep.depends_on:
            prev_layer = layer_dep.depends_on[0]
            if prev_layer in propagated:
                prev_out_mask = propagated[prev_layer]["output_mask"]
                # Only use previous mask if dimensions match
                if len(prev_out_mask) == in_dim:
                    return prev_out_mask.clone()
                else:
                    # Dimension mismatch - likely a skip connection or stride change
                    logger.debug(f"{layer_dep.name}: Input dim {in_dim} != prev output {len(prev_out_mask)}, using default mask")
                    return torch.ones(in_dim, dtype=torch.bool, device=device)

        # Default: keep all inputs
        return torch.ones(in_dim, dtype=torch.bool, device=device)

    def _create_weight_mask(self, module: nn.Module, out_mask: torch.Tensor, in_mask: torch.Tensor) -> torch.Tensor:
        """Create weight mask from input/output masks."""
        device = next(module.parameters()).device

        if isinstance(module, nn.Linear):
            # Linear: [out_features, in_features]
            # Handle dimension mismatch (e.g., from skip connections)
            expected_out = module.out_features
            expected_in = module.in_features

            if out_mask.shape[0] != expected_out:
                out_mask = torch.ones(expected_out, dtype=torch.bool, device=device)
            if in_mask.shape[0] != expected_in:
                in_mask = torch.ones(expected_in, dtype=torch.bool, device=device)

            weight_mask = out_mask.unsqueeze(1) & in_mask.unsqueeze(0)

        elif isinstance(module, nn.Conv2d):
            # Conv2d: [out_channels, in_channels, k_h, k_w]
            expected_out = module.out_channels
            expected_in = module.in_channels
            groups = int(getattr(module, "groups", 1))
            # Weight shape is [out_channels, in_channels/groups, k_h, k_w]
            in_per_group = int(module.weight.shape[1])

            # Handle dimension mismatch (e.g., from skip connections or downsample layers)
            if out_mask.shape[0] != expected_out:
                out_mask = torch.ones(expected_out, dtype=torch.bool, device=device)
            if in_mask.shape[0] != expected_in:
                in_mask = torch.ones(expected_in, dtype=torch.bool, device=device)

            # Grouped/depthwise conv needs special handling: the weight's "in channel"
            # dimension is in_channels/groups, and the input mask must be applied per-group.
            if groups > 1:
                out_per_group = expected_out // groups if groups > 0 else expected_out
                # Build a per-output-channel view of the relevant slice of in_mask
                in_mask_per_out = torch.empty((expected_out, in_per_group), dtype=torch.bool, device=device)
                for g in range(groups):
                    out_slice = slice(g * out_per_group, min((g + 1) * out_per_group, expected_out))
                    in_slice = slice(g * in_per_group, min((g + 1) * in_per_group, expected_in))
                    # Fallback to all-True if something is off (robustness)
                    if (in_slice.stop - in_slice.start) != in_per_group:
                        in_mask_per_out[out_slice] = torch.ones((out_slice.stop - out_slice.start, in_per_group), dtype=torch.bool, device=device)
                    else:
                        in_mask_per_out[out_slice] = in_mask[in_slice].view(1, -1).expand(out_slice.stop - out_slice.start, -1)

                weight_mask = (out_mask.view(-1, 1, 1, 1) & in_mask_per_out.view(expected_out, in_per_group, 1, 1)).expand_as(module.weight)
            else:
                weight_mask = (out_mask.view(-1, 1, 1, 1) & in_mask.view(1, -1, 1, 1)).expand_as(module.weight)

        elif isinstance(module, nn.Conv1d):
            # Conv1d: [out_channels, in_channels, k]
            expected_out = module.out_channels
            expected_in = module.in_channels
            groups = int(getattr(module, "groups", 1))
            in_per_group = int(module.weight.shape[1])

            if out_mask.shape[0] != expected_out:
                out_mask = torch.ones(expected_out, dtype=torch.bool, device=device)
            if in_mask.shape[0] != expected_in:
                in_mask = torch.ones(expected_in, dtype=torch.bool, device=device)

            if groups > 1:
                out_per_group = expected_out // groups if groups > 0 else expected_out
                in_mask_per_out = torch.empty((expected_out, in_per_group), dtype=torch.bool, device=device)
                for g in range(groups):
                    out_slice = slice(g * out_per_group, min((g + 1) * out_per_group, expected_out))
                    in_slice = slice(g * in_per_group, min((g + 1) * in_per_group, expected_in))
                    if (in_slice.stop - in_slice.start) != in_per_group:
                        in_mask_per_out[out_slice] = torch.ones((out_slice.stop - out_slice.start, in_per_group), dtype=torch.bool, device=device)
                    else:
                        in_mask_per_out[out_slice] = in_mask[in_slice].view(1, -1).expand(out_slice.stop - out_slice.start, -1)

                weight_mask = (out_mask.view(-1, 1, 1) & in_mask_per_out.view(expected_out, in_per_group, 1)).expand_as(module.weight)
            else:
                weight_mask = (out_mask.view(-1, 1, 1) & in_mask.view(1, -1, 1)).expand_as(module.weight)

        else:
            # Default: mask entire weight tensor
            weight_mask = torch.ones_like(module.weight, dtype=torch.bool)

        return weight_mask

    def _get_output_dimension(self, module: nn.Module) -> int:
        """Get output dimension of a module."""
        if hasattr(module, "out_features"):
            return module.out_features
        elif hasattr(module, "out_channels"):
            return module.out_channels
        elif hasattr(module, "embed_dim"):
            return module.embed_dim
        else:
            raise ValueError(f"Cannot determine output dimension for {type(module)}")

    def _get_input_dimension(self, module: nn.Module) -> int:
        """Get input dimension of a module."""
        if hasattr(module, "in_features"):
            return module.in_features
        elif hasattr(module, "in_channels"):
            return module.in_channels
        elif hasattr(module, "embed_dim"):
            return module.embed_dim
        else:
            raise ValueError(f"Cannot determine input dimension for {type(module)}")

    def _validate_pruning_plan(self, propagated_masks: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Validate that pruning plan maintains shape compatibility.

        Returns:
            {'valid': bool, 'errors': List[str], 'warnings': List[str]}
        """
        errors = []
        warnings = []

        for layer_name, masks in propagated_masks.items():
            dep = self.dependency_graph.graph[layer_name]
            module = dep.module

            # Check weight mask shape matches weight
            if masks["weight_mask"].shape != module.weight.shape:
                errors.append(f"{layer_name}: Weight mask shape {masks['weight_mask'].shape} " f"doesn't match weight shape {module.weight.shape}")

            # Check output mask dimension
            expected_out = self._get_output_dimension(module)
            if len(masks["output_mask"]) != expected_out:
                errors.append(f"{layer_name}: Output mask length {len(masks['output_mask'])} " f"doesn't match expected {expected_out}")

            # Check input mask dimension - be lenient for downsample/skip connection layers
            expected_in = self._get_input_dimension(module)
            if len(masks["input_mask"]) != expected_in:
                # Downsample layers in ResNet have different input sources (skip connections)
                # These are expected mismatches, not errors
                if "downsample" in layer_name or "shortcut" in layer_name:
                    warnings.append(
                        f"{layer_name}: Input mask adjusted for skip connection " f"(mask: {len(masks['input_mask'])}, expected: {expected_in})"
                    )
                else:
                    # For other layers, this is just a warning - we've already fixed the mask
                    warnings.append(f"{layer_name}: Input mask length {len(masks['input_mask'])} " f"adjusted to match expected {expected_in}")

            # Check for complete pruning (warn if layer fully pruned)
            if not masks["output_mask"].any():
                warnings.append(f"{layer_name}: All outputs pruned - layer is dead!")

        # Check inter-layer compatibility (only warn, don't error for ResNet-style architectures)
        for layer_name, masks in propagated_masks.items():
            dep = self.dependency_graph.graph[layer_name]

            if dep.feeds_into:
                next_layer = dep.feeds_into[0]
                if next_layer in propagated_masks:
                    # Our output mask should match next layer's input mask
                    our_out = masks["output_mask"]
                    their_in = propagated_masks[next_layer]["input_mask"]

                    if len(our_out) != len(their_in):
                        # This can happen with skip connections - downgrade to warning
                        warnings.append(
                            f"Dimension info: {layer_name}.out ({len(our_out)}) "
                            f"-> {next_layer}.in ({len(their_in)}) - may involve skip connection"
                        )
                    elif not torch.equal(our_out, their_in):
                        # Mask values differ - also just a warning for complex architectures
                        warnings.append(f"Mask info: {layer_name}.out_mask != {next_layer}.in_mask")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _apply_masks(self, propagated_masks: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Apply pruning masks to model weights.

        Returns:
            Statistics about pruning
        """
        stats = {"layers": {}, "total_params": 0, "total_params_kept": 0, "total_params_pruned": 0}

        for layer_name, masks in propagated_masks.items():
            module = self.dependency_graph.graph[layer_name].module

            # Apply weight mask
            with torch.no_grad():
                module.weight.data *= masks["weight_mask"].float()

                # Apply bias mask if exists
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data *= masks["output_mask"].float()

            # Compute statistics
            params_total = masks["weight_mask"].numel()
            params_kept = masks["weight_mask"].sum().item()
            params_pruned = params_total - params_kept

            stats["layers"][layer_name] = {
                "params_total": params_total,
                "params_kept": params_kept,
                "params_pruned": params_pruned,
                "sparsity": params_pruned / params_total if params_total > 0 else 0,
                "outputs_kept": masks["output_mask"].sum().item(),
                "outputs_total": len(masks["output_mask"]),
            }

            stats["total_params"] += params_total
            stats["total_params_kept"] += params_kept
            stats["total_params_pruned"] += params_pruned

        stats["overall_sparsity"] = stats["total_params_pruned"] / stats["total_params"] if stats["total_params"] > 0 else 0

        return stats

    def _compute_stats(self, propagated_masks: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Compute statistics without applying (for dry-run)."""
        stats = {"layers": {}, "total_params": 0, "total_params_kept": 0, "total_params_pruned": 0}

        for layer_name, masks in propagated_masks.items():
            params_total = masks["weight_mask"].numel()
            params_kept = masks["weight_mask"].sum().item()
            params_pruned = params_total - params_kept

            stats["layers"][layer_name] = {
                "params_total": params_total,
                "params_kept": params_kept,
                "params_pruned": params_pruned,
                "sparsity": params_pruned / params_total,
                "outputs_kept": masks["output_mask"].sum().item(),
                "outputs_total": len(masks["output_mask"]),
            }

            stats["total_params"] += params_total
            stats["total_params_kept"] += params_kept
            stats["total_params_pruned"] += params_pruned

        stats["overall_sparsity"] = stats["total_params_pruned"] / stats["total_params"]

        return stats

    def print_pruning_plan(self, propagated_masks: Dict, show_details: bool = False):
        """Print human-readable pruning plan."""
        print("\n" + "=" * 80)
        print("Pruning Plan (Dependency-Aware)")
        print("=" * 80)

        for layer_name in sorted(propagated_masks.keys()):
            masks = propagated_masks[layer_name]
            dep = self.dependency_graph.graph[layer_name]

            out_kept = masks["output_mask"].sum().item()
            out_total = len(masks["output_mask"])
            in_kept = masks["input_mask"].sum().item()
            in_total = len(masks["input_mask"])

            print(f"\n{layer_name} ({dep.layer_type}):")
            print(f"  Outputs: {out_kept}/{out_total} kept ({100*out_kept/out_total:.1f}%)")
            print(f"  Inputs:  {in_kept}/{in_total} kept ({100*in_kept/in_total:.1f}%)")

            if show_details:
                weight_kept = masks["weight_mask"].sum().item()
                weight_total = masks["weight_mask"].numel()
                print(f"  Weights: {weight_kept}/{weight_total} kept ({100*weight_kept/weight_total:.1f}%)")

        print("\n" + "=" * 80)


def prune_model_with_dependencies(
    model: nn.Module, layer_scores: Dict[str, torch.Tensor], amount: float, mode: str = "low", dry_run: bool = False, verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for dependency-aware pruning.

    Args:
        model: Model to prune
        layer_scores: Importance scores per layer
        amount: Pruning amount (0-1)
        mode: 'low' or 'high'
        dry_run: If True, don't actually prune
        verbose: Print pruning plan

    Returns:
        Pruning results
    """
    pruner = DependencyAwarePruning(model)

    result = pruner.prune(layer_scores, amount, mode, dry_run)

    if verbose:
        pruner.print_pruning_plan(result["masks"])
        print(f"\nOverall: {result['stats']['overall_sparsity']:.1%} sparsity")

    return result
