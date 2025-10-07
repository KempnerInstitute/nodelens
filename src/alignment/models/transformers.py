"""
Enhanced transformer wrapper with Q/K/V tracking and per-head analysis.

Supports:
- Q/K/V projection tracking
- Per-head metric computation
- Token-level or sequence-level aggregation
- Attention head pruning

Works with:
- HuggingFace models (GPT, LLaMA, BERT, etc.)
- Torch vision transformers
- Custom transformer implementations
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.registry import register_model
from .base import BaseModelWrapper

logger = logging.getLogger(__name__)


@register_model("transformer_enhanced")
class TransformerWrapperEnhanced(BaseModelWrapper):
    """
    Enhanced wrapper for transformer models with detailed attention tracking.

    Features:
    - Tracks Q/K/V projections separately
    - Extracts per-head representations
    - Supports token-level or sequence-level analysis
    - Enables head-level pruning

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
        >>> wrapper = TransformerWrapperEnhanced(
        ...     model,
        ...     track_qkv=True,
        ...     track_per_head=True
        ... )
        >>> # Compute per-head redundancy
        >>> head_scores = compute_head_importance(wrapper, inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        tracked_layers: Optional[List[str]] = None,
        track_qkv: bool = True,
        track_per_head: bool = False,
        aggregation: str = "sequence_mean",  # 'sequence_mean' or 'token_level'
        num_heads: Optional[int] = None,  # Auto-detect if None
        head_dim: Optional[int] = None,  # Auto-detect if None
        **config: Any,
    ):
        """
        Initialize enhanced transformer wrapper.

        Args:
            model: Transformer model (HF or custom)
            tracked_layers: Layers to track (None = auto-discover)
            track_qkv: Whether to track Q/K/V projections separately
            track_per_head: Whether to extract per-head representations
            aggregation: How to aggregate sequence dimension
                - 'sequence_mean': Average over tokens [B, Heads*Dh]
                - 'token_level': Keep all tokens [B*T, Heads*Dh]
            num_heads: Number of attention heads (auto-detect if None)
            head_dim: Dimension per head (auto-detect if None)
            **config: Additional configuration
        """
        super().__init__(model, tracked_layers, **config)

        self.track_qkv = track_qkv
        self.track_per_head = track_per_head
        self.aggregation = aggregation
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Auto-detect architecture parameters
        if self.num_heads is None or self.head_dim is None:
            self._detect_architecture_params()

        # Discover and track attention-specific layers
        if track_qkv:
            self.qkv_layers = self._discover_qkv_layers()
            logger.info(f"Discovered {len(self.qkv_layers)} Q/K/V projections")

    def _detect_architecture_params(self):
        """Auto-detect number of heads and head dimension from model."""
        # Try common attributes (HuggingFace models)
        if hasattr(self._model, "config"):
            config = self._model.config
            if hasattr(config, "num_attention_heads"):
                self.num_heads = config.num_attention_heads
            if hasattr(config, "hidden_size") and self.num_heads:
                self.head_dim = config.hidden_size // self.num_heads

            logger.info(f"Auto-detected: {self.num_heads} heads, {self.head_dim} dim/head")

        # Fallback: try to infer from module inspection
        if self.num_heads is None:
            for name, module in self._model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    self.num_heads = module.num_heads
                    self.head_dim = module.embed_dim // module.num_heads
                    logger.info(f"Auto-detected from MultiheadAttention: {self.num_heads} heads")
                    break

    def _discover_qkv_layers(self) -> Dict[str, List[str]]:
        """
        Discover Q/K/V projection layers generically.

        Uses LayerDetector when available, pattern matching as fallback.

        Returns:
            Dict with 'query', 'key', 'value' lists of layer names
        """
        try:
            # Generic detection
            from ..core.layer_detector import LayerDetector

            detector = LayerDetector()
            all_layers = detector.detect_all_layers(self._model)

            qkv = {"query": [], "key": [], "value": []}

            # Attention projections detected by role
            for layer_info in all_layers:
                if layer_info.role == "attention_projection":
                    # Infer Q/K/V from name patterns as secondary heuristic
                    name_lower = layer_info.name.lower()
                    if any(p in name_lower for p in ["q_proj", "query", ".q", "wq"]):
                        qkv["query"].append(layer_info.name)
                    elif any(p in name_lower for p in ["k_proj", "key", ".k", "wk"]):
                        qkv["key"].append(layer_info.name)
                    elif any(p in name_lower for p in ["v_proj", "value", ".v", "wv"]):
                        qkv["value"].append(layer_info.name)

            logger.info(f"Q/K/V layers discovered: {len(qkv['query'])} Q, {len(qkv['key'])} K, {len(qkv['value'])} V")
            return qkv

        except ImportError:
            # Fallback to pattern matching (legacy)
            qkv = {"query": [], "key": [], "value": []}

            for name, module in self._model.named_modules():
                if isinstance(module, nn.Linear):
                    name_lower = name.lower()
                    if any(pattern in name_lower for pattern in ["q_proj", "query", ".q", "wq"]):
                        qkv["query"].append(name)
                    elif any(pattern in name_lower for pattern in ["k_proj", "key", ".k", "wk"]):
                        qkv["key"].append(name)
                    elif any(pattern in name_lower for pattern in ["v_proj", "value", ".v", "wv"]):
                        qkv["value"].append(name)

            return qkv

    def extract_attention_heads(
        self, attention_output: torch.Tensor, num_heads: Optional[int] = None, head_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract per-head representations from attention output.

        Args:
            attention_output: Attention output tensor
                - [B, T, D] for standard output
                - [B, Heads, T, Dh] for some models
            num_heads: Number of heads (uses self.num_heads if None)
            head_dim: Dimension per head (uses self.head_dim if None)

        Returns:
            Per-head representation based on aggregation mode:
                - sequence_mean: [B, Heads, Dh] → [B, Heads*Dh]
                - token_level: [B*T, Heads*Dh]
        """
        num_heads = num_heads or self.num_heads
        head_dim = head_dim or self.head_dim

        if num_heads is None or head_dim is None:
            raise ValueError("num_heads and head_dim must be specified or auto-detected")

        # Handle different input formats
        if attention_output.ndim == 4:
            # Already in [B, Heads, T, Dh] format
            B, H, T, Dh = attention_output.shape

            if self.aggregation == "sequence_mean":
                # Average over tokens: [B, Heads, Dh] → [B, Heads*Dh]
                return attention_output.mean(dim=2).reshape(B, H * Dh)
            else:
                # Keep tokens: [B, T, Heads, Dh] → [B*T, Heads*Dh]
                return attention_output.permute(0, 2, 1, 3).reshape(B * T, H * Dh)

        elif attention_output.ndim == 3:
            # [B, T, D] format where D = Heads * Dh
            B, T, D = attention_output.shape

            # Reshape to [B, T, Heads, Dh]
            reshaped = attention_output.reshape(B, T, num_heads, head_dim)

            if self.aggregation == "sequence_mean":
                # [B, Heads, Dh] → [B, Heads*Dh]
                return reshaped.mean(dim=1).reshape(B, num_heads * head_dim)
            else:
                # [B*T, Heads*Dh]
                return reshaped.reshape(B * T, num_heads * head_dim)

        else:
            # Fallback: flatten
            logger.warning(f"Unexpected attention output shape: {attention_output.shape}, flattening")
            return attention_output.reshape(attention_output.shape[0], -1)

    def get_qkv_activations(self, layer_prefix: str) -> Dict[str, torch.Tensor]:
        """
        Get Q/K/V activations for a specific attention layer.

        Args:
            layer_prefix: Prefix of attention layer (e.g., 'model.layers.0.self_attn')

        Returns:
            Dict with 'query', 'key', 'value' activations
        """
        qkv_acts = {}

        for proj_type in ["query", "key", "value"]:
            # Find matching layer in QKV layers
            matching = [name for name in self.qkv_layers.get(proj_type, []) if name.startswith(layer_prefix)]

            if matching:
                layer_name = matching[0]
                if f"{layer_name}_output" in self._activation_cache:
                    qkv_acts[proj_type] = self._activation_cache[f"{layer_name}_output"]

        return qkv_acts

    def get_ffn_activations(self, layer_prefix: str) -> Dict[str, torch.Tensor]:
        """
        Get FFN (MLP) activations for transformer layer.

        For LLaMA-3 style models:
        - up_proj: [hidden_size → intermediate_size] (e.g., 4096 → 11008)
        - down_proj: [intermediate_size → hidden_size] (e.g., 11008 → 4096)
        - gate_proj: (if exists) gating mechanism

        Args:
            layer_prefix: Prefix of FFN/MLP layer (e.g., 'model.layers.0.mlp')

        Returns:
            Dict with 'up_proj', 'down_proj', 'gate_proj' (if exists)
        """
        ffn_acts = {}

        # Common FFN component names
        ffn_patterns = ["up_proj", "down_proj", "gate_proj", "fc1", "fc2", "wi", "wo"]

        for pattern in ffn_patterns:
            full_name = f"{layer_prefix}.{pattern}"

            # Check if this layer exists and has cached activations
            if f"{full_name}_input" in self._activation_cache:
                ffn_acts[f"{pattern}_input"] = self._activation_cache[f"{full_name}_input"]
            if f"{full_name}_output" in self._activation_cache:
                ffn_acts[f"{pattern}_output"] = self._activation_cache[f"{full_name}_output"]

        return ffn_acts

    def compute_per_head_representations(self, activations: Dict[str, torch.Tensor], layer_name: str) -> torch.Tensor:
        """
        Compute per-head representations for metric computation.

        Args:
            activations: Activation dictionary
            layer_name: Name of attention layer

        Returns:
            Per-head representation [B, num_heads * head_dim]
            where each "neuron" represents one head
        """
        # Get attention output
        attn_output = activations.get(f"{layer_name}_output")

        if attn_output is None:
            raise ValueError(f"No output found for {layer_name}")

        # Extract heads
        return self.extract_attention_heads(attn_output)

    def _discover_layers(self) -> List[str]:
        """
        Override to discover transformer-specific layers.

        Uses generic LayerDetector for model-agnostic detection,
        with fallback to pattern matching for backward compatibility.
        """
        try:
            # Use generic detector (model-agnostic!)
            from ..core.layer_detector import detect_trackable_layers

            trackable_layers = detect_trackable_layers(
                self._model, min_neurons=1, roles=["linear_general", "ffn_expansion", "ffn_contraction", "attention_projection", "attention"]
            )

            logger.info(f"Auto-discovered {len(trackable_layers)} transformer layers (generic detector)")
            return trackable_layers

        except ImportError:
            # Fallback to pattern matching for backward compatibility
            logger.warning("LayerDetector not available, using pattern matching (legacy)")

            trackable_layers = []

            for name, module in self._model.named_modules():
                # Track Linear layers in attention and FFN
                if isinstance(module, nn.Linear):
                    # Check if part of attention or FFN (pattern-based fallback)
                    if any(
                        pattern in name
                        for pattern in [
                            "attn",
                            "attention",
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "mlp",
                            "ffn",
                            "fc1",
                            "fc2",
                            "up_proj",
                            "down_proj",
                            "gate_proj",
                        ]
                    ):
                        trackable_layers.append(name)

                # Track MultiheadAttention modules
                elif isinstance(module, nn.MultiheadAttention):
                    trackable_layers.append(name)

            logger.info(f"Auto-discovered {len(trackable_layers)} transformer layers (pattern matching)")
            return trackable_layers

    def preprocess_activations(self, activations: Dict[str, torch.Tensor], mode: str = "auto") -> Dict[str, torch.Tensor]:
        """
        Preprocess transformer activations.

        Handles:
        - [B, T, D] sequence outputs
        - [B, Heads, T, Dh] multi-head outputs
        - Token vs sequence aggregation
        """
        if mode == "none":
            return activations

        processed = {}

        for name, activation in activations.items():
            if activation.ndim == 3:
                # [B, T, D] - sequence format
                B, T, D = activation.shape

                if mode == "auto" or mode == "flatten":
                    # Flatten to [B, T*D] or [B*T, D] depending on use case
                    # For metrics: [B, D] by averaging over T
                    processed[name] = activation.mean(dim=1)  # [B, D]

                elif mode == "sequence_mean":
                    processed[name] = activation.mean(dim=1)  # [B, D]

                elif mode == "token_level":
                    processed[name] = activation.reshape(B * T, D)  # [B*T, D]

                else:
                    processed[name] = activation

            elif activation.ndim == 4:
                # [B, Heads, T, Dh] - multi-head format
                # Extract per-head
                if self.track_per_head:
                    processed[name] = self.extract_attention_heads(activation)
                else:
                    # Flatten
                    processed[name] = activation.reshape(activation.shape[0], -1)

            else:
                # 2D or other - keep as is or flatten
                processed[name] = activation.reshape(activation.shape[0], -1) if activation.ndim > 2 else activation

        return processed


@register_model("llama_wrapper")
class LLaMAWrapper(TransformerWrapperEnhanced):
    """
    Specialized wrapper for LLaMA models (LLaMA-2, LLaMA-3, etc.).

    LLaMA architecture:
    - Attention: Q/K/V projections + output projection
    - FFN: gate_proj, up_proj, down_proj (SwiGLU activation)

    Example for LLaMA-3:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')
        >>> wrapper = LLaMAWrapper(model)
        >>>
        >>> # Track specific layer
        >>> wrapper.track_layer('model.layers.0.mlp.up_proj')
        >>>
        >>> # Compute per-neuron scores on FFN
        >>> # up_proj: [hidden_size, intermediate_size] e.g., [4096, 11008]
        >>> scores = compute_neuron_scores(wrapper, 'model.layers.0.mlp.up_proj')
        >>> # scores.shape = [11008] - one per FFN neuron
    """

    def __init__(self, model: nn.Module, track_ffn: bool = True, track_attention: bool = True, **config):
        """
        Initialize LLaMA wrapper.

        Args:
            model: LLaMA model from HuggingFace
            track_ffn: Whether to track FFN (MLP) layers
            track_attention: Whether to track attention layers
            **config: Additional configuration
        """
        super().__init__(model, **config)

        self.track_ffn = track_ffn
        self.track_attention = track_attention

        # Discover LLaMA-specific layers
        self.ffn_layers = self._discover_ffn_layers() if track_ffn else {}
        self.attention_layers = self._discover_attention_layers() if track_attention else {}

        logger.info(f"LLaMA model: {len(self.ffn_layers)} FFN layers, {len(self.attention_layers)} attention layers")

    def _discover_ffn_layers(self) -> Dict[str, List[str]]:
        """
        Discover FFN/MLP layers generically using structural analysis.

        Uses LayerDetector to identify expansion/contraction based on
        dimension ratios, not hard-coded names.

        Returns:
            Dict with 'expansion', 'contraction', 'gate' layer names
        """
        try:
            # Generic detection (model-agnostic!)
            from ..core.layer_detector import LayerDetector

            detector = LayerDetector()
            all_layers = detector.detect_all_layers(self._model)

            # Group by role (detected generically)
            ffn = {"expansion": [], "contraction": [], "gate": []}

            for layer_info in all_layers:
                if layer_info.role == "ffn_expansion":
                    ffn["expansion"].append(layer_info.name)
                elif layer_info.role == "ffn_contraction":
                    ffn["contraction"].append(layer_info.name)
                # Gate layers detected by additional heuristic
                elif "gate" in layer_info.name.lower() and layer_info.role == "ffn_expansion":
                    ffn["gate"].append(layer_info.name)

            logger.info(f"FFN layers discovered generically: {len(ffn['expansion'])} expansion, " f"{len(ffn['contraction'])} contraction")

            # Backward compatibility: also use old names
            self.ffn_legacy_names = {"up_proj": ffn["expansion"], "down_proj": ffn["contraction"], "gate_proj": ffn["gate"]}

            return ffn

        except ImportError:
            # Fallback to pattern matching (legacy)
            logger.warning("LayerDetector not available, using pattern matching")

            ffn = {"expansion": [], "contraction": [], "gate": []}

            for name, module in self._model.named_modules():
                if isinstance(module, nn.Linear):
                    # Pattern-based fallback
                    if "mlp.up_proj" in name or "mlp.wi" in name:
                        ffn["expansion"].append(name)
                    elif "mlp.down_proj" in name or "mlp.wo" in name:
                        ffn["contraction"].append(name)
                    elif "mlp.gate_proj" in name:
                        ffn["gate"].append(name)

            return ffn

    def _discover_attention_layers(self) -> Dict[str, List[str]]:
        """
        Discover attention layers generically.

        Uses LayerDetector or falls back to pattern matching.
        """
        try:
            # Generic detection
            from ..core.layer_detector import LayerDetector

            detector = LayerDetector()
            all_layers = detector.detect_all_layers(self._model)

            attn = {"attention": [], "q": [], "k": [], "v": [], "o": []}

            for layer_info in all_layers:
                if layer_info.role == "attention_projection":
                    # Try to infer Q/K/V/O from position or dimensions
                    # For now, add to general attention list
                    attn["attention"].append(layer_info.name)

                    # Heuristic: check name for hints (as fallback)
                    if "q" in layer_info.name.lower():
                        attn["q"].append(layer_info.name)
                    elif "k" in layer_info.name.lower():
                        attn["k"].append(layer_info.name)
                    elif "v" in layer_info.name.lower():
                        attn["v"].append(layer_info.name)
                    elif "o" in layer_info.name.lower() or "out" in layer_info.name.lower():
                        attn["o"].append(layer_info.name)

            logger.info(f"Attention layers discovered generically")
            return attn

        except ImportError:
            # Fallback to pattern matching
            logger.warning("LayerDetector not available, using pattern matching")

            attn = {"attention": [], "q": [], "k": [], "v": [], "o": []}

            for name, module in self._model.named_modules():
                if "self_attn" in name or "attention" in name:
                    if isinstance(module, nn.Linear):
                        if "q_proj" in name or ".q" in name:
                            attn["q"].append(name)
                        elif "k_proj" in name or ".k" in name:
                            attn["k"].append(name)
                        elif "v_proj" in name or ".v" in name:
                            attn["v"].append(name)
                        elif "o_proj" in name or "out_proj" in name:
                            attn["o"].append(name)
                        else:
                            attn["attention"].append(name)

            return attn

    def get_layer_info_detailed(self, layer_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a LLaMA layer.

        Args:
            layer_name: Layer name (e.g., 'model.layers.0.mlp.up_proj')

        Returns:
            Detailed layer information including:
            - Type (FFN, attention, etc.)
            - Dimensions
            - Number of neurons/heads
        """
        info = super().get_layer_info(layer_name)

        # Add LLaMA-specific information
        if "mlp" in layer_name:
            info["component"] = "FFN"
            if "up_proj" in layer_name or "gate_proj" in layer_name:
                info["ffn_role"] = "expansion"
                info["neurons"] = info.get("out_features", "unknown")
            elif "down_proj" in layer_name:
                info["ffn_role"] = "projection"
                info["neurons"] = info.get("in_features", "unknown")

        elif "self_attn" in layer_name or "attention" in layer_name:
            info["component"] = "Attention"
            if "q_proj" in layer_name:
                info["projection"] = "Query"
            elif "k_proj" in layer_name:
                info["projection"] = "Key"
            elif "v_proj" in layer_name:
                info["projection"] = "Value"
            elif "o_proj" in layer_name:
                info["projection"] = "Output"

        return info


def compute_per_head_scores(wrapper: TransformerWrapperEnhanced, layer_name: str, inputs: torch.Tensor, metric: Any, **metric_kwargs) -> torch.Tensor:
    """
    Compute importance scores per attention head.

    Args:
        wrapper: Enhanced transformer wrapper
        layer_name: Attention layer name
        inputs: Input batch
        metric: Metric to compute (e.g., redundancy, RQ)
        **metric_kwargs: Additional metric arguments

    Returns:
        Scores per head [num_heads]
    """
    # Get activations
    outputs, activations = wrapper.forward_with_activations(inputs)

    # Extract per-head representations
    head_outputs = wrapper.compute_per_head_representations(activations, layer_name)

    # Compute metric treating each head as a "neuron"
    scores = metric.compute(outputs=head_outputs, **metric_kwargs)

    return scores
