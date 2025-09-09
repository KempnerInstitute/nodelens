"""
Transformer-specific model wrapper to capture MLP/attention block activations.

Works with Hugging Face causal LMs and vision transformers that expose module
names for attention and MLP/FFN blocks. It registers hooks for the specified
submodules and normalizes their outputs/inputs to 2D for metric computation.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import logging

from .base import BaseModelWrapper
from ..core.registry import register_model

logger = logging.getLogger(__name__)


@register_model("transformer_wrapper")
class TransformerWrapper(BaseModelWrapper):
    """
    Wrap a transformer-style model (HF or timm/vit) and track specified blocks.

    Args:
        model: nn.Module (e.g., HF AutoModel/AutoModelForCausalLM)
        tracked_layers: list of module name substrings to track (matched by contains)
        flatten_activations: whether to flatten to 2D for metrics
    """

    def __init__(
        self,
        model: nn.Module,
        tracked_layers: Optional[List[str]] = None,
        flatten_activations: bool = True,
        **config: Any,
    ):
        # If tracked_layers are substrings, expand them to exact module names
        concrete_layers = self._resolve_layers(model, tracked_layers) if tracked_layers else None
        super().__init__(model, concrete_layers, flatten_activations=flatten_activations, **config)

    def _resolve_layers(self, model: nn.Module, substrings: List[str]) -> List[str]:
        names = [name for name, _ in model.named_modules()]
        chosen: List[str] = []
        for ss in substrings:
            hits = [n for n in names if ss in n]
            chosen.extend(hits)
        # Deduplicate preserving order
        seen = set()
        ordered = []
        for n in chosen:
            if n not in seen:
                ordered.append(n)
                seen.add(n)
        logger.info(f"TransformerWrapper resolved {len(ordered)} layers from substrings {substrings}")
        return ordered

    def preprocess_activations(self, activations: Dict[str, torch.Tensor], mode: str = "flatten") -> Dict[str, torch.Tensor]:
        # Override to better handle [B, T, D] or [B, Heads, T, Dh] tensors
        if mode == "none":
            return activations
        processed: Dict[str, torch.Tensor] = {}
        for name, act in activations.items():
            x = act
            # Reduce heads/time by flattening along feature dim; keep batch leading
            if x.ndim >= 3:
                b = x.shape[0]
                x = x.reshape(b, -1)
            processed[name] = x
        return processed


