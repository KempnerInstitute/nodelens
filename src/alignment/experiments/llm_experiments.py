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

    # Evaluation
    evaluation_compute_perplexity: bool = False
    evaluation_dataset: str = "wikitext"
    evaluation_split: str = "test"
    evaluation_num_samples: int = 100

    # Alignment
    importance_computation_texts: List[str] = field(default_factory=list)
    importance_num_samples: int = 1

    # Pruning
    pruning_enabled: bool = False
    pruning_algorithms: List[str] = field(default_factory=lambda: ["alignment"])
    pruning_sparsity_levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    pruning_alignment_metric: str = "activation_l2_norm"
    pruning_mode: str = "low"  # "low" or "high"

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

        logger.info(f"Evaluating perplexity on {dataset} ({split})...")

        # Load dataset
        try:
            from alignment.dataops.datasets.text_datasets import load_text_dataset
        except Exception as e:
            logger.error(f"Could not import text dataset loader: {e}")
            raise

        # Load calibration texts
        dataset_obj = load_text_dataset(dataset, self.llm_config.model_id, split=split, max_samples=num_samples)

        # Compute perplexity
        self.model.eval()
        nlls = []
        total_length = 0

        with torch.no_grad():
            for i, batch in enumerate(dataset_obj):
                if i >= num_samples:
                    break

                input_ids = batch["input_ids"].unsqueeze(0).to(self.llm_config.device)
                
                # Create labels and mask out padding tokens
                labels = input_ids.clone()
                
                # Get pad token id (usually 128001 for Llama 3)
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id
                
                # Set padding tokens to -100 (ignored in loss)
                labels[labels == pad_token_id] = -100
                
                # Also ignore BOS token (128000) at start
                if labels[0, 0] == 128000:
                    labels[0, 0] = -100

                try:
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs.loss
                    
                    # Count only non-ignored tokens
                    num_valid_tokens = (labels != -100).sum().item()
                    
                    if num_valid_tokens > 0:
                        nlls.append(loss * num_valid_tokens)
                        total_length += num_valid_tokens
                        logger.info(f"Sample {i}: loss={loss.item():.4f}, valid_tokens={num_valid_tokens}")
                    else:
                        logger.warning(f"Sample {i}: No valid tokens!")
                        
                except Exception as e:
                    logger.warning(f"Error on sample {i}: {e}")
                    continue

        if total_length == 0:
            logger.error("No valid tokens processed!")
            return float('inf')
        
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
    
    # def apply_pruning(self, sparsity: float = 0.2, metric: str = "activation_l2_norm", mode: str = "low") -> Dict[str, torch.Tensor]:
    #     """
    #     Apply pruning to the model based on importance scores.

    #     Args:
    #         sparsity: Fraction of neurons to prune
    #         metric: Which importance metric to use
    #         mode: 'low' to prune low-importance, 'high' for high-importance

    #     Returns:
    #         Dictionary of pruning masks
    #     """
    #     logger.info(f"Applying pruning: sparsity={sparsity}, metric={metric}, mode={mode}")

    #     if not self.importance_scores:
    #         raise ValueError("Must compute importance scores before pruning")

    #     config = PruningConfig(amount=sparsity, structured=True, pruning_mode=mode)

    #     pruner = AlignmentPruning(metric=metric, config=config)

    #     masks = {}
    #     for layer_name in self.importance_scores.keys():
    #         if metric not in self.importance_scores[layer_name]:
    #             continue

    #         scores = self.importance_scores[layer_name][metric]
    #         layer_module = dict(self.wrapped_model._model.named_modules())[layer_name]

    #         # Get target module for pruning
    #         if hasattr(layer_module, "gate_proj"):
    #             target = layer_module.gate_proj
    #         elif hasattr(layer_module, "up_proj"):
    #             target = layer_module.up_proj
    #         else:
    #             continue

    #         try:
    #             mask = pruner.create_pruning_mask(scores)
    #             pruner.apply_pruning(target, mask)
    #             masks[layer_name] = mask

    #             sparsity_achieved = (mask == 0).float().mean().item()
    #             logger.info(f"  {layer_name}: {sparsity_achieved:.2%} sparsity")
    #         except Exception as e:
    #             logger.error(f"Error pruning {layer_name}: {e}")

    #     self.pruning_masks = masks
    #     return masks

    def apply_pruning(self, sparsity: float = 0.2, metric: str = "activation_l2_norm", mode: str = "low") -> Dict[str, torch.Tensor]:
        """
        Apply structured pruning to MLP layers.
        Prunes gate_proj, up_proj (output dims), and down_proj (input dims) together.

        Args:
            sparsity: Fraction of neurons to prune
            metric: Which importance metric to use
            mode: 'low' to prune low-importance, 'high' for high-importance

        Returns:
            Dictionary of pruning masks
        """
        logger.info(f"Applying pruning: sparsity={sparsity}, metric={metric}, mode={mode}")

        if not self.importance_scores:
            raise ValueError("Must compute importance scores before pruning")

        config = PruningConfig(amount=sparsity, structured=True, pruning_mode=mode)
        pruner = AlignmentPruning(metric=metric, config=config)

        masks = {}
        processed_mlps = set()  # Track which MLPs we've already processed
        
        for layer_name in self.importance_scores.keys():
            if metric not in self.importance_scores[layer_name]:
                continue

            # Extract layer index (e.g., "model.layers.0.mlp.gate_proj" → 0)
            import re
            match = re.search(r'layers\.(\d+)\.mlp', layer_name)
            if not match:
                continue
            layer_idx = match.group(1)
            
            # Skip if we already processed this MLP
            if layer_idx in processed_mlps:
                continue
            processed_mlps.add(layer_idx)
            
            # Get importance scores (should be for gate_proj)
            scores = self.importance_scores[layer_name][metric]
            
            # Create mask based on importance scores
            mask = pruner.create_pruning_mask(scores)
            
            # Get the MLP module
            mlp_module = dict(self.wrapped_model._model.named_modules())[f"model.layers.{layer_idx}.mlp"]
            
            try:
                # Verify we have the right modules
                if not all(hasattr(mlp_module, attr) for attr in ['gate_proj', 'up_proj', 'down_proj']):
                    logger.warning(f"Layer {layer_idx} MLP missing expected projections")
                    continue
                
                # Verify mask shape matches intermediate dimension
                expected_dim = mlp_module.gate_proj.out_features

                if len(mask) != expected_dim:
                    logger.error(f"Mask size {len(mask)} doesn't match intermediate dim {expected_dim}")
                    continue
                
                # Prune gate_proj output dimension (rows of weight matrix)
                pruner.apply_pruning(mlp_module.gate_proj, mask, dim="output")
                masks[f"model.layers.{layer_idx}.mlp.gate_proj"] = mask
                
                # Prune up_proj output dimension (rows of weight matrix) - same mask
                pruner.apply_pruning(mlp_module.up_proj, mask, dim="output")
                masks[f"model.layers.{layer_idx}.mlp.up_proj"] = mask
                
                # Prune down_proj input dimension (columns of weight matrix)
                pruner.apply_pruning(mlp_module.down_proj, mask, dim="input")
                masks[f"model.layers.{layer_idx}.mlp.down_proj"] = mask
                
                sparsity_achieved = (mask == 0).float().mean().item()
                logger.info(f"  Layer {layer_idx} MLP: {sparsity_achieved:.2%} sparsity across all projections")
                
            except Exception as e:
                logger.error(f"Error pruning layer {layer_idx} MLP: {e}")
                import traceback
                logger.error(traceback.format_exc())

        self.pruning_masks = masks
        logger.info(f"Pruned {len(processed_mlps)} MLP layers with {sparsity:.1%} target sparsity")
        return masks

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

        if self.llm_config.evaluation_compute_perplexity:
            baseline_ppl = self.evaluate_perplexity(dataset=self.llm_config.evaluation_dataset, num_samples=self.llm_config.evaluation_num_samples)
            results["evaluation"]["baseline_perplexity"] = baseline_ppl


        if self.llm_config.pruning_enabled:
            sparsity_levels = self.llm_config.pruning_sparsity_levels
            metric = self.llm_config.pruning_alignment_metric

            for sparsity in sparsity_levels:
                masks = self.apply_pruning(sparsity=sparsity, metric=metric)

                # Evaluate pruned model
                if self.llm_config.evaluation_compute_perplexity:
                    pruned_ppl = self.evaluate_perplexity(
                        dataset=self.llm_config.evaluation_dataset, num_samples=self.llm_config.evaluation_num_samples
                    )

                    results["pruning_results"][f"sparsity_{sparsity}"] = {
                        "perplexity": pruned_ppl,
                        "sparsity": sparsity,
                        "num_pruned_layers": len(masks),
                    }

        return results