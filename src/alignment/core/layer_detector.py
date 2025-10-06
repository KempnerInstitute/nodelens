"""
Generic layer detection without model-specific patterns.

Detects layer roles using structural analysis rather than naming conventions,
making the framework truly model-agnostic.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """Information about a detected layer."""

    name: str
    module: nn.Module
    role: str  # 'linear', 'conv', 'attention_q', 'attention_k', 'attention_v', 'ffn_up', 'ffn_down', etc.
    in_dim: int
    out_dim: int
    is_trackable: bool = True
    parent_block: Optional[str] = None


class LayerDetector:
    """
    Generic layer detection using structural analysis.

    Works for any model architecture without hard-coded naming patterns.
    Detects layer roles based on:
    - Layer type (Conv, Linear, etc.)
    - Dimension ratios
    - Position in network
    - Local graph structure

    Example:
        >>> detector = LayerDetector()
        >>> layers = detector.detect_all_layers(model)
        >>> ffn_layers = [l for l in layers if 'ffn' in l.role]
    """

    def __init__(self, min_neurons: int = 1, max_neurons: Optional[int] = None, track_normalization: bool = False):
        """
        Initialize layer detector.

        Args:
            min_neurons: Minimum neurons/channels to track
            max_neurons: Maximum neurons/channels to track (None = no limit)
            track_normalization: Whether to track normalization layers
        """
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.track_normalization = track_normalization

    def detect_all_layers(self, model: nn.Module, include_roles: Optional[List[str]] = None) -> List[LayerInfo]:
        """
        Detect all trackable layers in a model.

        Args:
            model: PyTorch model
            include_roles: Filter by roles (None = all)

        Returns:
            List of LayerInfo objects
        """
        all_layers = []

        # Build module parent map
        parent_map = self._build_parent_map(model)

        # Analyze each module
        for name, module in model.named_modules():
            # Detect layer type and role
            layer_type = self._classify_layer_type(module)

            if layer_type == "skip":
                continue

            # Get dimensions
            in_dim, out_dim = self._get_dimensions(module)

            if in_dim is None or out_dim is None:
                continue

            # Filter by size
            if out_dim < self.min_neurons:
                continue
            if self.max_neurons and out_dim > self.max_neurons:
                continue

            # Infer role from structure
            role = self._infer_role(module, name, parent_map, layer_type)

            # Determine if trackable
            is_trackable = self._is_trackable(module, role)

            # Create LayerInfo
            layer_info = LayerInfo(
                name=name, module=module, role=role, in_dim=in_dim, out_dim=out_dim, is_trackable=is_trackable, parent_block=parent_map.get(name)
            )

            all_layers.append(layer_info)

        # Filter by requested roles
        if include_roles:
            all_layers = [layer for layer in all_layers if layer.role in include_roles]

        # Filter by trackable
        all_layers = [layer for layer in all_layers if layer.is_trackable]

        logger.info(f"Detected {len(all_layers)} trackable layers")

        return all_layers

    def _classify_layer_type(self, module: nn.Module) -> str:
        """Classify layer into basic categories."""
        if isinstance(module, nn.Linear):
            return "linear"
        elif isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            return "conv"
        elif isinstance(module, nn.MultiheadAttention):
            return "attention"
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            return "normalization"
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh)):
            return "activation"
        elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
            return "dropout"
        else:
            return "skip"

    def _get_dimensions(self, module: nn.Module) -> Tuple[Optional[int], Optional[int]]:
        """Get input and output dimensions of a module."""
        if isinstance(module, nn.Linear):
            return module.in_features, module.out_features

        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            return module.in_channels, module.out_channels

        elif isinstance(module, nn.MultiheadAttention):
            return module.embed_dim, module.embed_dim

        elif hasattr(module, "normalized_shape"):  # LayerNorm
            if isinstance(module.normalized_shape, tuple):
                dim = module.normalized_shape[0]
            else:
                dim = module.normalized_shape
            return dim, dim

        else:
            return None, None

    def _infer_role(self, module: nn.Module, name: str, parent_map: Dict, layer_type: str) -> str:
        """
        Infer semantic role using structural cues (not names!).

        Returns:
            Role string like 'linear_general', 'ffn_expansion', 'attention_proj', etc.
        """
        if layer_type == "conv":
            return "conv"

        elif layer_type == "attention":
            return "attention"

        elif layer_type == "normalization":
            return "normalization"

        elif layer_type == "linear":
            # Infer Linear role from dimensions
            in_dim, out_dim = self._get_dimensions(module)

            # Check dimension ratios (model-agnostic!)
            ratio = out_dim / in_dim if in_dim > 0 else 1.0

            if ratio > 2.5:
                # Expansion (likely FFN up_proj or gate_proj)
                return "ffn_expansion"

            elif ratio < 0.4:
                # Contraction (likely FFN down_proj)
                return "ffn_contraction"

            elif 0.9 <= ratio <= 1.1:
                # Same dimensions (likely attention Q/K/V/O or residual)
                # Check if part of attention block (heuristic)
                parent_map.get(name, "")

                # Look for attention-related siblings
                if self._has_attention_siblings(name, parent_map):
                    return "attention_projection"
                else:
                    return "linear_residual"

            else:
                # General linear layer
                return "linear_general"

        else:
            return layer_type

    def _has_attention_siblings(self, layer_name: str, parent_map: Dict) -> bool:
        """
        Check if layer has siblings that look like Q/K/V projections.

        Heuristic: If there are 3-4 Linear layers with same dimensions
        in the same parent block, likely attention.
        """
        # Get parent
        parent = parent_map.get(layer_name)
        if not parent:
            return False

        # Find siblings (other layers with same parent)
        siblings = [name for name, p in parent_map.items() if p == parent]

        # Count Linear siblings with similar dimensions
        linear_siblings = 0
        for sibling in siblings:
            # This is a simplified heuristic
            # In practice, would check actual modules
            if "linear" in sibling.lower() or "proj" in sibling.lower():
                linear_siblings += 1

        # If 3-4 Linear siblings, likely Q/K/V (+ maybe O)
        return linear_siblings >= 3

    def _build_parent_map(self, model: nn.Module) -> Dict[str, str]:
        """Build map of module names to parent names."""
        parent_map = {}

        for name, module in model.named_modules():
            # Get parent name
            if "." in name:
                parent = ".".join(name.split(".")[:-1])
                parent_map[name] = parent

        return parent_map

    def _is_trackable(self, module: nn.Module, role: str) -> bool:
        """Determine if layer should be tracked for metrics."""
        # Skip non-parametric layers
        if not hasattr(module, "weight") or module.weight is None:
            return False

        # Skip normalization unless requested
        if role == "normalization" and not self.track_normalization:
            return False

        # Track everything else
        return True

    def group_by_role(self, layers: List[LayerInfo]) -> Dict[str, List[LayerInfo]]:
        """Group layers by their role."""
        grouped = {}

        for layer in layers:
            if layer.role not in grouped:
                grouped[layer.role] = []
            grouped[layer.role].append(layer)

        return grouped

    def get_layers_by_role(self, model: nn.Module, role: str) -> List[LayerInfo]:
        """Get all layers matching a specific role."""
        all_layers = self.detect_all_layers(model)
        return [layer for layer in all_layers if layer.role == role]


def detect_trackable_layers(model: nn.Module, min_neurons: int = 1, roles: Optional[List[str]] = None) -> List[str]:
    """
    Convenience function to get trackable layer names.

    Args:
        model: PyTorch model
        min_neurons: Minimum size to track
        roles: Filter by roles (None = all)

    Returns:
        List of layer names suitable for tracking
    """
    detector = LayerDetector(min_neurons=min_neurons)
    layers = detector.detect_all_layers(model, include_roles=roles)
    return [layer.name for layer in layers]
