"""
LLM-specific experiment classes for alignment analysis and pruning.

Refactored to use a typed dataclass config and to initialize models/tokenizers
in a way that's consistent with BaseExperiment / GeneralAlignmentExperiment.
"""

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
    alignment_metrics: List[str] = field(default_factory=lambda: ["rayleigh_quotient"])
    importance_computation_texts: List[str] = field(default_factory=list)
    importance_num_samples: int = 1

    # Pruning
    pruning_enabled: bool = False
    pruning_algorithms: List[str] = field(default_factory=lambda: ["alignment"])
    pruning_sparsity_levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    pruning_alignment_metric: str = "rayleigh_quotient"
    pruning_mode: str = "low"  # "low" or "high"
    pruning_structured: bool = True

    # Evaluation
    evaluation_compute_perplexity: bool = False
    evaluation_dataset: str = "wikitext"
    evaluation_split: str = "test"
    evaluation_num_samples: int = 100

    # Misc
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class LLMAlignmentExperiment(BaseExperiment):
    """
    Experiment for analyzing neuron alignment in Large Language Models.

    Usage:
        config = LLMAlignmentConfig(name="llm_exp", model_id="meta-llama/...")
        exp = LLMAlignmentExperiment(config)
        exp.setup()
        results = exp.run()
    """

    def __init__(self, config: LLMAlignmentConfig):
        if not isinstance(config, LLMAlignmentConfig):
            # allow passing plain dicts for convenience
            config = LLMAlignmentConfig.from_dict(config) if isinstance(config, dict) else config
        super().__init__(config)
        self.tokenizer = None
        self.importance_scores: Dict[str, Dict[str, torch.Tensor]] = {}
        self.pruning_masks: Dict[str, torch.Tensor] = {}
        self.evaluation_results: Dict[str, Any] = {}

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

        # Initialize metrics specifically for alignment metrics if not present
        for metric_name in self.llm_config.alignment_metrics:
            if metric_name not in self.metrics:
                MetricClass = get_metric(metric_name)
                self.metrics[metric_name] = MetricClass()

        # Setup pruning-related objects if enabled (no heavy instantiation here)
        if self.llm_config.pruning_enabled:
            logger.info("Pruning enabled; pruning will be available during run()")

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

    def compute_importance_scores(
        self, dataset: str = "wikitext", split: str = "test", num_samples: int = 1
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute importance scores for tracked layers using configured metrics.

        Returns mapping {layer_name: {metric_name: scores_tensor}}
        """
        logger.info("Computing importance scores for LLM tracked layers...")

        try:
            from alignment.dataops.datasets.text_datasets import load_text_dataset
        except Exception as e:
            logger.error(f"Could not import text dataset loader: {e}")
            raise

        dataset_obj = load_text_dataset(dataset, self.llm_config.model_id, split=split, max_samples=num_samples)
        calibration_texts = dataset_obj.texts
        num_samples = min(num_samples, len(calibration_texts))

        self.llm_config.importance_computation_texts = calibration_texts[:num_samples]
        
        self.model.eval()

        all_activations: Dict[str, List[torch.Tensor]] = {}

        for text in calibration_texts[:num_samples]:
            # tokenize and move to device
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                **getattr(self.llm_config, "tokenizer_kwargs", {})
            )
            inputs = {k: v.to(self.llm_config.device) for k, v in inputs.items()}

            with torch.no_grad():
                # prefer wrapper method if available
                if hasattr(self.wrapped_model, "forward_with_activations"):
                    outputs, activations = self.wrapped_model.forward_with_activations(inputs)
                elif hasattr(self.model, "forward_with_activations"):
                    outputs, activations = self.model.forward_with_activations(inputs)
                else:
                    # fallback
                    outputs = self.model(inputs)
                    activations = {}
                    logger.warning("No forward_with_activations available; activations may be empty")

            # accumulate activations
            for key, value in activations.items():
                all_activations.setdefault(key, []).append(value.detach().cpu())

        # collapse lists into tensors
        aggregated_activations = {
            key: torch.cat(values, dim=0) if len(values) > 1 else values[0]
            for key, values in all_activations.items()
        }

        metric_names = self.llm_config.alignment_metrics

        # expand tracked layers if needed
        tracked = getattr(self.wrapped_model, "tracked_layers", None) or getattr(
            self.llm_config, "tracked_layer_patterns", []
        )
        if tracked and isinstance(tracked[0], str) and any(("*" in x or "[" in x) for x in tracked):
            tracked = self._expand_layer_patterns(tracked, self._get_underlying_model())

        for layer_name in tracked:
            logger.info(f"Computing importance for layer: {layer_name}")
            layer_input_key = f"{layer_name}_input"
            if layer_input_key not in aggregated_activations:
                logger.warning(f"No activations available for {layer_name} (expected key {layer_input_key})")
                continue

            layer_inputs = aggregated_activations[layer_input_key].to(self.llm_config.device)

            # find module
            named_modules = dict(self._get_underlying_model().named_modules())
            if layer_name not in named_modules:
                logger.warning(f"Layer {layer_name} not found among model modules")
                continue

            layer_module = named_modules[layer_name]
            weight = self._get_layer_weights(layer_module)
            if weight is None:
                logger.warning(f"No suitable weight found for {layer_name}; skipping")
                continue

            layer_scores: Dict[str, torch.Tensor] = {}
            for metric_name in metric_names:
                try:
                    MetricClass = get_metric(metric_name)
                    metric = MetricClass()
                    scores = metric.compute(inputs=layer_inputs, weights=weight)
                    layer_scores[metric_name] = scores.detach().cpu()
                    logger.debug(
                        f"{layer_name} {metric_name}: mean={float(scores.mean()):.6f} "
                        f"std={float(scores.std()):.6f}"
                    )
                except Exception as e:
                    logger.error(f"Failed to compute metric {metric_name} on {layer_name}: {e}")

            self.importance_scores[layer_name] = layer_scores

        return self.importance_scores, calibration_texts[:num_samples]

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

    def apply_pruning(self, sparsity: float = 0.2, metric: str = "rayleigh_quotient", mode: str = "low") -> Dict[str, torch.Tensor]:
        """
        Apply pruning using AlignmentPruning for layers that have importance scores.
        Returns mapping {layer_name: mask_tensor}
        """
        logger.info(f"Applying pruning: sparsity={sparsity}, metric={metric}, mode={mode}")

        if not self.importance_scores:
            raise RuntimeError("Importance scores are empty. Call compute_importance_scores() first.")

        pruner = AlignmentPruning(metric=metric, config=PruningConfig(amount=sparsity, structured=self.llm_config.pruning_structured, pruning_mode=mode))

        masks: Dict[str, torch.Tensor] = {}
        # iterate layers for which we have scores
        for layer_name, metrics in self.importance_scores.items():
            if metric not in metrics:
                continue
            scores = metrics[metric].to(self.llm_config.device)

            # get module to prune
            named_modules = dict(self._get_underlying_model().named_modules())
            if layer_name not in named_modules:
                logger.warning(f"Layer {layer_name} not present when applying pruning")
                continue
            layer_module = named_modules[layer_name]

            # determine target parameter/module to prune
            target = None
            for candidate_name in ("gate_proj", "up_proj", "fc1", "lin", "weight"):
                if hasattr(layer_module, candidate_name):
                    candidate = getattr(layer_module, candidate_name)
                    # accept parameters or modules with .weight
                    if isinstance(candidate, torch.nn.Parameter):
                        target = candidate
                    elif isinstance(candidate, torch.nn.Module) and hasattr(candidate, "weight"):
                        target = getattr(candidate, "weight")
                    elif isinstance(candidate, torch.Tensor):
                        target = candidate
                    if target is not None:
                        break

            if target is None:
                logger.warning(f"No pruning target found for {layer_name}; skipping")
                continue

            try:
                mask = pruner.create_pruning_mask(scores)
                pruner.apply_pruning(target, mask)
                masks[layer_name] = mask.detach().cpu()
                sparsity_achieved = float((mask == 0).float().mean().item())
                logger.info(f"Pruned {layer_name}: achieved sparsity {sparsity_achieved:.2%}")
            except Exception as e:
                logger.error(f"Error pruning {layer_name}: {e}")

        self.pruning_masks = masks
        return masks

    def evaluate_perplexity(self, dataset: str = "wikitext", split: str = "test", num_samples: int = 100) -> float:
        """
        Evaluate model perplexity on a text dataset.
        Expects a helper load_text_dataset(dataset, tokenizer, split, max_samples).
        """
        logger.info(f"Evaluating perplexity on {dataset} ({split}) for {num_samples} samples")

        # lazy import dataset loader to avoid strict dependency
        try:
            from alignment.data.datasets.text_datasets import load_text_dataset
        except Exception as e:
            logger.error(f"Could not import text dataset loader: {e}")
            raise

        dataset_obj = load_text_dataset(dataset, self.tokenizer, split=split, max_samples=num_samples)

        self.model.eval()
        nlls = []
        total_length = 0

        with torch.no_grad():
            for i, batch in enumerate(dataset_obj):
                if i >= num_samples:
                    break
                # expecting each batch to be dict with input_ids and optionally labels
                input_ids = batch["input_ids"].unsqueeze(0).to(self.llm_config.device)
                labels = batch.get("labels", input_ids).to(self.llm_config.device)
                try:
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = getattr(outputs, "loss", None)
                    if loss is None:
                        # Some HF models return tuple
                        if isinstance(outputs, tuple) and len(outputs) > 0:
                            loss = outputs[0]
                        else:
                            logger.warning(f"No loss returned for sample {i}; skipping")
                            continue
                    nlls.append(loss * input_ids.size(1))
                    total_length += input_ids.size(1)
                except Exception as e:
                    logger.warning(f"Error evaluating sample {i}: {e}")
                    continue

        if total_length == 0 or len(nlls) == 0:
            logger.warning("No valid samples for perplexity computation; returning inf")
            return float("inf")

        ppl = torch.exp(torch.stack(nlls).sum() / total_length)
        perplexity = float(ppl.item())
        logger.info(f"Perplexity computed: {perplexity:.2f}")
        return perplexity

    def run(self) -> Dict[str, Any]:
        """Run the full LLM experiment pipeline: compute importance, optionally prune, evaluate."""
        logger.info("Running LLMAlignmentExperiment...")

        results: Dict[str, Any] = {"config": self.llm_config.to_dict(), "importance_scores": {}, "pruning_results": {}, "evaluation": {}}

        # compute importance scores
        scores, calibration_text = self.compute_importance_scores(
            dataset=self.llm_config.evaluation_dataset,
            split=self.llm_config.evaluation_split,
            num_samples=self.llm_config.importance_num_samples
        )
        results["importance_computation_texts"] = calibration_text

        # summarize scores
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

        # baseline evaluation
        # if self.llm_config.evaluation_compute_perplexity:
        #     try:
        #         baseline_ppl = self.evaluate_perplexity(
        #             dataset=self.llm_config.evaluation_dataset,
        #             split=self.llm_config.evaluation_split,
        #             num_samples=self.llm_config.evaluation_num_samples,
        #         )
        #         results["evaluation"]["baseline_perplexity"] = baseline_ppl
        #     except Exception as e:
        #         logger.error(f"Error computing baseline perplexity: {e}")
        #         results["evaluation"]["baseline_perplexity"] = None

        # pruning experiments
        if self.llm_config.pruning_enabled:
            for sparsity in self.llm_config.pruning_sparsity_levels:
                try:
                    masks = self.apply_pruning(sparsity=sparsity, metric=self.llm_config.pruning_alignment_metric, mode=self.llm_config.pruning_mode)
                    eval_result = {}
                    if self.llm_config.evaluation_compute_perplexity:
                        try:
                            pruned_ppl = self.evaluate_perplexity(
                                dataset=self.llm_config.evaluation_dataset,
                                split=self.llm_config.evaluation_split,
                                num_samples=self.llm_config.evaluation_num_samples,
                            )
                            eval_result["perplexity"] = pruned_ppl
                        except Exception as e:
                            logger.error(f"Error computing pruned perplexity for sparsity={sparsity}: {e}")
                            eval_result["perplexity"] = None

                    results["pruning_results"][f"sparsity_{sparsity}"] = {
                        "sparsity": sparsity,
                        "num_pruned_layers": len(masks),
                        "masks": {k: v.tolist() if isinstance(v, torch.Tensor) else str(type(v)) for k, v in masks.items()},
                        "evaluation": eval_result,
                    }
                except Exception as e:
                    logger.error(f"Pruning run failed for sparsity={sparsity}: {e}")

        return results
