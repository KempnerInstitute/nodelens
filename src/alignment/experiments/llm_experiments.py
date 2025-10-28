import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from alignment.experiments.base import ExperimentConfig, BaseExperiment
from alignment.metrics import get_metric
from alignment.models.transformers import TransformerWrapperEnhanced as TransformerWrapper
from alignment.pruning import AlignmentPruning, PruningConfig
from alignment.training.base import BaseTrainer  # kept for compatibility if used elsewhere

logger = logging.getLogger(__name__)


@dataclass
class LLMAlignmentConfig(ExperimentConfig):
    """
    LLM-specific configuration that extends the generic ExperimentConfig.
    Keeps compatibility with the rest of your codebase while adding LLM fields.
    """

    # override some defaults useful for LLMs
    model_name: str = "hf_causal_lm"  # special name to indicate HF causal LM
    model_id: Optional[str] = None  # HuggingFace model id (e.g., "meta-llama/..." )
    model_backend: str = "hf"  # "hf" (huggingface) or "registry" (use MODEL_REGISTRY)
    torch_dtype: Optional[str] = None  # e.g. "bfloat16", "float16", "float32"
    hf_device_map: Optional[Dict[str, Union[str, int]]] = None  # for device_map if desired

    # wrapper/tracking
    wrapper_name: str = "transformer_wrapper"
    tracked_layer_patterns: List[str] = field(default_factory=lambda: ["model.layers.*.mlp"])

    # Alignment
    alignment_methods: List[str] = field(default_factory=lambda: ["activation_l2_norm"])
    importance_computation_texts: List[str] = field(default_factory=list)
    importance_num_samples: int = 1

    # dataset
    dataset_name: str = "wikitext-2-v1"

    # Alignment
    importance_computation_texts: List[str] = field(default_factory=list)
    importance_num_samples: int = 1

    # Misc
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class LLMAlignmentExperiment(BaseExperiment):
    def __init__(self, config: LLMAlignmentConfig):
        if not isinstance(config, LLMAlignmentConfig):
            config = LLMAlignmentConfig.from_dict(config) if isinstance(config, dict) else config

        super().__init__(config)
        self.tokenizer = None
        self.importance_scores: Dict[str, Dict[str, torch.Tensor]] = {}

    @property
    def llm_config(self) -> LLMAlignmentConfig:
        return self.config  # type: ignore[return-value]

    def setup(self):
        """Setup LLM alignment experiment components."""
        logger.info("Setting up LLM alignment experiment...")

        # If using HuggingFace backend, load tokenizer & HF model then wrap
        if self.llm_config.model_backend == "hf":
            self._load_hf_tokenizer_and_model()
        else:
            # If not HF, rely on BaseExperiment's initialization (already called in __init__).
            logger.info("Using registry or torchvision model; BaseExperiment has initialized it.")

        # Expand tracked layer patterns into actual layer names for the wrapper
        if self.llm_config.tracked_layer_patterns is not None:
            underlying_model = self._get_underlying_model()
            expanded = self._expand_layer_patterns(self.llm_config.tracked_layer_patterns, underlying_model)

            if expanded:
                # Directly set the internal storage for tracked layers
                if hasattr(self.wrapped_model, "_tracked_layers"):
                    self.wrapped_model._tracked_layers = expanded
                else:
                    # fallback if internal attribute differs
                    setattr(self.wrapped_model, "_tracked_layers", expanded)

                logger.info(f"Tracked layers expanded to {len(expanded)} layers")

        print("underlying_model: ", underlying_model)
        print("expanded: ", expanded)

    def evaluate_perplexity(self, dataset: str = "wikitext", split: str = "test", num_samples: int = 100) -> float:
        """
        Evaluate model perplexity on a dataset.

        Args:
            dataset: Dataset name
            split: Dataset split
            num_samples: Number of samples to evaluate

        Returns:
            Perplexity value
        """
        from ..data.datasets.text_datasets import load_text_dataset

        logger.info(f"Evaluating perplexity on {dataset} ({split})...")

        # Load dataset
        dataset_obj = load_text_dataset(dataset, self.tokenizer, split=split, max_samples=num_samples)

        # Compute perplexity
        self.model.eval()
        nlls = []
        total_length = 0

        with torch.no_grad():
            for i, batch in enumerate(dataset_obj):
                if i >= num_samples:
                    break

                input_ids = batch["input_ids"].unsqueeze(0).to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                try:
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs.loss
                    nlls.append(loss * input_ids.size(1))
                    total_length += input_ids.size(1)
                except Exception as e:
                    logger.warning(f"Error on sample {i}: {e}")
                    continue

        ppl = torch.exp(torch.stack(nlls).sum() / total_length)
        perplexity = ppl.item()

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def _get_underlying_model(self) -> nn.Module:
        """
        Return underlying raw nn.Module inside the wrapper.
        Supports wrappers that store model as .model or ._model.
        """
        if hasattr(self.wrapped_model, "model"):
            return getattr(self.wrapped_model, "model")
        if hasattr(self.wrapped_model, "_model"):
            return getattr(self.wrapped_model, "_model")
        # Fall back to attribute 'module' or the wrapper itself
        if hasattr(self.wrapped_model, "module"):
            return getattr(self.wrapped_model, "module")
        return self.wrapped_model  # type: ignore[return-value]

    def _load_hf_tokenizer_and_model(self):
        """Load HuggingFace tokenizer + causal LM and wrap it."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import AutoConfig

        model_id = self.llm_config.model_id
        if not model_id:
            raise ValueError("LLMAlignmentExperiment requires config.model_id for HF backend")

        logger.info(f"Loading tokenizer for {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **self.llm_config.tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load model config and model with dtype/device options
        model_kwargs = dict(self.llm_config.model_kwargs or {})
        torch_dtype = None
        if self.llm_config.torch_dtype:
            # map string to torch dtype if possible
            try:
                torch_dtype = getattr(torch, self.llm_config.torch_dtype)
            except Exception:
                torch_dtype = None

        # Use device_map when provided; otherwise load to CPU/GPU according to config.device
        device_map = self.llm_config.hf_device_map

        logger.info(f"Loading HF model {model_id} with dtype={self.llm_config.torch_dtype} device_map={device_map}")
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device_map, **model_kwargs)

        # Move model to explicit device if device_map not used
        if device_map is None:
            device = torch.device(self.llm_config.device)
            hf_model = hf_model.to(device)

        # Wrap with TransformerWrapper (expects an nn.Module)
        # Wrapper constructor signature may vary; try to pass tracked layers and other opts
        wrapper_kwargs = {"tracked_layers": getattr(self.llm_config, "tracked_layer_patterns", None)}
        try:
            wrapped = TransformerWrapper(hf_model, **wrapper_kwargs)
        except Exception:
            # fallback to a minimal wrapper creation if signature differs
            wrapped = TransformerWrapper(hf_model)

        # store references
        self.tokenizer = tokenizer
        self.model = hf_model
        self.wrapped_model = wrapped

        logger.info("HuggingFace model + tokenizer loaded and wrapped.")

    def _expand_layer_patterns(self, patterns: List[str], model: nn.Module) -> List[str]:
        """
        Expand layer patterns with wildcards to actual layer names.

        Supports patterns like:
          - "model.layers.*.mlp" -> ["model.layers.0.mlp", "model.layers.1.mlp", ...]
          - "model.layers.[0-15].self_attn" -> first 16 attention layers
        """
        import re

        expanded: List[str] = []
        all_names = [name for name, _ in model.named_modules()]

        for pattern in patterns:
            if "*" in pattern:
                # convert simple glob to regex: '*' -> \d+ (numbers for indices)
                regex_pattern = pattern.replace(".", r"\.").replace("*", r"\d+")
                regex = re.compile(f"^{regex_pattern}$")
                matches = [name for name in all_names if regex.match(name)]
                expanded.extend(matches)
            elif "[" in pattern and "]" in pattern:
                # Range like [0-15]
                m = re.search(r"\[(\d+)-(\d+)\]", pattern)
                if m:
                    start, end = int(m.group(1)), int(m.group(2))
                    base_pattern = pattern[: m.start()] + "{}" + pattern[m.end() :]
                    for i in range(start, end + 1):
                        candidate = base_pattern.format(i)
                        if candidate in all_names:
                            expanded.append(candidate)
            else:
                if pattern in all_names:
                    expanded.append(pattern)

        # deduplicate while preserving order
        seen = set()
        deduped = []
        for name in expanded:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped
    

    def compute_importance_scores(self, num_samples: int = 1) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute importance scores for tracked layers using configured metrics.
        Returns mapping {layer_name: {metric_name: scores_tensor}}
        """
        logger.info("Computing importance scores for LLM tracked layers...")

        calibration_texts = self.dataset.texts
        num_samples = min(num_samples, len(calibration_texts))
        self.llm_config.importance_computation_texts = calibration_texts[:num_samples]

        self.model.eval()

        all_activations = {}

        for text in calibration_texts[:num_samples]:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.llm_config.device) for k, v in inputs.items()}

            outputs, activations = self.wrapped_model.forward_with_activations(inputs)

            # Accumulate activations
            for key, value in activations.items():
                if key not in all_activations:
                    all_activations[key] = []
                all_activations[key].append(value)

        # Average activations if multiple samples
        if len(calibration_texts) > 1:
            all_activations = {key: torch.cat(values, dim=0) for key, values in all_activations.items()}
        else:
            all_activations = {key: values[0] for key, values in all_activations.items()}

        # Compute importance for each layer
        metric_names = self.llm_config.alignment_methods

        for layer_name in self.wrapped_model._tracked_layers:
            logger.info(f"Computing scores for {layer_name}")

            layer_module = dict(self.wrapped_model._model.named_modules())[layer_name]
            layer_input_key = f"{layer_name}_input"

            if layer_input_key not in all_activations:
                logger.warning(f"No activations for {layer_name}")
                continue

            layer_inputs = all_activations[layer_input_key]

            # Get weight tensor (prefer gate_proj for MLP layers)
            weight = self._get_layer_weights(layer_module)
            if weight is None:
                continue

            # Compute scores with each metric
            layer_scores = {}
            for metric_name in metric_names:
                try:
                    # Use already-initialized metric from self.metrics if available
                    if metric_name in self.metrics:
                        metric = self.metrics[metric_name]
                    else:
                        # Otherwise get fresh from registry without extra params
                        metric = get_metric(metric_name)

                    scores = metric.compute(inputs=layer_inputs, weights=weight)
                    layer_scores[metric_name] = scores

                    logger.debug(f"  {metric_name}: " f"mean={scores.mean().item():.6f}, " f"std={scores.std().item():.6f}")
                except Exception as e:
                    logger.error(f"Error computing {metric_name} for {layer_name}: {e}")
                    continue

            self.importance_scores[layer_name] = layer_scores
        
        return self.importance_scores
    
    def _get_layer_weights(self, layer_module: nn.Module) -> Optional[torch.Tensor]:
        """Find the weight tensor to use for importance/pruning decisions."""
        # common MLP naming
        for attr in ("gate_proj", "up_proj", "fc1", "fc2", "lin", "weight"):
            if hasattr(layer_module, attr):
                w = getattr(layer_module, attr)
                # if attribute is a Parameter or Module, get .weight when needed
                if isinstance(w, torch.nn.Parameter):
                    return w
                if isinstance(w, torch.nn.Module) and hasattr(w, "weight"):
                    return getattr(w, "weight")
                # else maybe it's a tensor
                return w if isinstance(w, torch.Tensor) else None
        return None

    def run(self) -> Dict[str, Any]:
        """Run the full LLM experiment pipeline: compute importance, optionally prune, evaluate."""
        logger.info("Running LLMAlignmentExperiment...")

        results: Dict[str, Any] = {"config": self.llm_config.to_dict(), "importance_scores": {}, "pruning_results": {}, "evaluation": {}}

        scores = self.compute_importance_scores(
            num_samples=self.llm_config.importance_num_samples
        )

        for layer_name, layer_scores in scores.items():
            results["importance_scores"][layer_name] = {}
            for metric_name, vals in layer_scores.items():
                try:
                    results["importance_scores"][layer_name][metric_name] = {
                        "mean": float(vals.mean().item()),
                        "std": float(vals.std().item()),
                        "min": float(vals.min().item()),
                        "max": float(vals.max().item()),
                    }
                except Exception:
                    # if vals is non-tensor or empty
                    results["importance_scores"][layer_name][metric_name] = {"summary": "unavailable"}

        return results