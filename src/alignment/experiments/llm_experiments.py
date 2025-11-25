import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from alignment.experiments.base import ExperimentConfig, BaseExperiment
from alignment.metrics import get_metric
from alignment.models.transformers import TransformerWrapperEnhanced as TransformerWrapper
from alignment.pruning import AlignmentPruning, PruningConfig
from alignment.services import MaskOperations
from alignment.training.base import BaseTrainer  # kept for compatibility if used elsewhere
from alignment.core.streaming import StreamingCovariance
from alignment.analysis.visualization import UnifiedVisualizer

logger = logging.getLogger(__name__)


class LLMAlignmentExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.importance_scores: Dict[str, Dict[str, torch.Tensor]] = {}

    def setup(self):
        """Setup LLM alignment experiment components."""
        logger.info("Setting up LLM alignment experiment...")

        # If using HuggingFace backend, (re)wrap the HF model and load tokenizer.
        # Prefer reusing an already-initialized registry model (hf_causal_lm) to
        # avoid double-loading large checkpoints.
        if self.config.model_config.get("model_backend") == "hf":
            if (
                getattr(self, "model", None) is not None
                and self.config.model_name.lower() == "hf_causal_lm"
            ):
                logger.info("Reusing existing 'hf_causal_lm' model from registry for LLMAlignmentExperiment.")
                self._wrap_existing_hf_model()
            else:
                self._load_hf_tokenizer_and_model()
        else:
            # If not HF, rely on BaseExperiment's initialization (already called in __init__).
            logger.info("Using registry or torchvision model; BaseExperiment has initialized it.")

        expanded = None

        # Expand tracked layer patterns into actual layer names for the wrapper
        if self.config.tracked_layers is not None:
            underlying_model = self._get_underlying_model()
            expanded = self._expand_layer_patterns(self.config.tracked_layers, underlying_model)

            if expanded:
                # Directly set the internal storage for tracked layers
                if hasattr(self.wrapped_model, "_tracked_layers"):
                    self.wrapped_model._tracked_layers = expanded
                else:
                    # fallback if internal attribute differs
                    setattr(self.wrapped_model, "_tracked_layers", expanded)

                logger.info(f"Tracked layers expanded to {len(expanded)} layers")

        if expanded is not None:
            # print("underlying_model: ", underlying_model)
            print("expanded: ", expanded)

        # Ensure we have a text dataset for importance computation in LLM experiments.
        # BaseExperiment may skip dataset initialization for LLM experiment types.
        if getattr(self, "dataset", None) is None:
            try:
                from alignment.dataops.datasets.text_datasets import load_text_dataset
            except ImportError as e:
                logger.error(f"Unable to import text datasets for LLMAlignmentExperiment: {e}")
                self.dataset = None
            else:
                # Use dataset_name if provided, otherwise fall back to evaluation_dataset.
                dataset_name = getattr(self.config, "dataset_name", None) or getattr(
                    self.config, "evaluation_dataset", "wikitext"
                )
                model_id = self.config.model_config.get("model_id")
                logger.info(
                    f"Creating text calibration dataset '{dataset_name}' for model '{model_id}' "
                    f"with up to {self.config.alignment_data_num_samples} samples."
                )
                # We intentionally load a Dataset object with a .texts list so we can reuse
                # the calibration texts for multiple metrics without repeatedly calling HF.
                try:
                    text_dataset = load_text_dataset(
                        dataset_name,
                        model_id,
                        split="train",
                        max_length=512,
                        max_samples=self.config.alignment_data_num_samples,
                    )
                    # Many of our text datasets expose a `.texts` attribute for raw strings.
                    if not hasattr(text_dataset, "texts"):
                        logger.warning(
                            f"Loaded text dataset '{dataset_name}' does not expose `.texts`; "
                            f"LLM importance scores will fall back to iterating the dataset."
                        )
                    self.dataset = text_dataset
                except Exception as e:
                    logger.error(f"Failed to create text dataset '{dataset_name}': {e}")
                    self.dataset = None

    def evaluate_perplexity(self, dataset: str = "wikitext", split: str = "test", num_samples: int = 100) -> float:
        """
        Evaluate model perplexity on a dataset (bfloat16-safe).

        Args:
            dataset: Dataset name
            split: Dataset split
            num_samples: Number of samples to evaluate

        Returns:
            Perplexity value
        """
        import torch
        from torch import autocast

        logger.info(f"Evaluating perplexity on {dataset} ({split})...")

        # Load dataset
        from alignment.dataops.datasets.text_datasets import load_text_dataset
        dataset_obj = load_text_dataset(dataset, self.config.model_config.get("model_id"), split=split, max_samples=num_samples)

        self.model.eval()
        nlls = []
        total_length = 0

        device = torch.device(self.config.device)
        model_dtype = getattr(torch, self.config.model_config.get("torch_dtype", "float32"))

        with torch.no_grad():
            for i, batch in enumerate(dataset_obj):
                if i >= num_samples:
                    break

                # Move input_ids to device (long, never bfloat16)
                input_ids = batch["input_ids"].unsqueeze(0).to(device, dtype=torch.long)

                # Prepare labels
                labels = input_ids.clone()
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
                labels[labels == pad_token_id] = -100
                if labels[0, 0] == 128000:  # ignore BOS token if needed
                    labels[0, 0] = -100

                try:
                    # Use autocast for bfloat16-safe forward
                    with autocast(device_type=self.config.device, dtype=model_dtype):
                        outputs = self.model(input_ids, labels=labels)
                        loss = outputs.loss

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
            return float("inf")

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

    def _wrap_existing_hf_model(self) -> None:
        """Reuse an HF Causal LM created via the model registry and wrap it."""
        from transformers import AutoTokenizer

        model_id = self.config.model_config.get("model_id")
        if not model_id:
            raise ValueError("LLMAlignmentExperiment requires config.model_id for HF backend")

        logger.info(f"Loading tokenizer for existing HF causal LM '{model_id}'")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **self.config.tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Unwrap underlying HF model if we're holding a small wrapper (e.g., HFCausalLM)
        hf_model = getattr(self.model, "model", self.model)

        # Wrap with TransformerWrapper (expects an nn.Module)
        wrapper_kwargs = {"tracked_layers": getattr(self.config, "tracked_layers", None)}
        try:
            wrapped = TransformerWrapper(hf_model, **wrapper_kwargs)
        except Exception:
            # Fallback to a minimal wrapper creation if signature differs
            wrapped = TransformerWrapper(hf_model)

        self.tokenizer = tokenizer
        self.model = hf_model
        self.wrapped_model = wrapped

        logger.info("Reused HF causal LM from registry and wrapped with TransformerWrapperEnhanced.")

    def _load_hf_tokenizer_and_model(self) -> None:
        """Load HuggingFace tokenizer + causal LM and wrap it."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = self.config.model_config.get("model_id")
        if not model_id:
            raise ValueError("LLMAlignmentExperiment requires config.model_id for HF backend")

        logger.info(f"Loading tokenizer for {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **self.config.tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load model config and model with dtype/device options
        model_kwargs = dict(self.config.model_kwargs or {})
        torch_dtype = None
        if self.config.model_config.get("torch_dtype"):
            # map string to torch dtype if possible
            try:
                torch_dtype = getattr(torch, self.config.model_config.get("torch_dtype"))
            except Exception:
                torch_dtype = None

        # Use device_map when provided; otherwise load to CPU/GPU according to config.device
        device_map = self.config.model_config.get("hf_device_map", self.config.model_config.get("device_map"))

        logger.info(f"Loading HF model {model_id} with dtype={torch_dtype} device_map={device_map}")
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device_map, **model_kwargs)

        # Move model to explicit device if device_map not used
        if device_map is None:
            device = torch.device(self.config.device)
            hf_model = hf_model.to(device)

        # Wrap with TransformerWrapper (expects an nn.Module)
        # Wrapper constructor signature may vary; try to pass tracked layers and other opts
        wrapper_kwargs = {"tracked_layers": getattr(self.config, "tracked_layers", None)}
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

    def _normalize_activation(self, tensor: torch.Tensor, hidden_dim: Optional[int] = None) -> torch.Tensor:
        """
        Convert LLM activations to [1, hidden_dim] by averaging over sequence.
        
        For LLM linear layers (up_proj, down_proj, etc.):
        - Raw activation might be flattened to [batch, seq*hidden] or [seq*hidden]
        - If hidden_dim is provided, we reshape and average properly
        - Otherwise we try to infer from tensor shape
        
        This ensures consistent feature dimensions regardless of input sequence length.
        """
        if tensor is None:
            return None

        tensor = tensor.detach()
        
        # If hidden_dim is provided, use it to properly reshape
        if hidden_dim is not None:
            # Flatten everything and reshape to [N, hidden_dim]
            flat = tensor.reshape(-1)
            num_elements = flat.numel()
            if num_elements % hidden_dim == 0:
                # Reshape to [seq_or_batch*seq, hidden_dim] and average
                reshaped = flat.reshape(-1, hidden_dim)
                result = reshaped.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                return result
        
        # Fallback: assume last dimension is hidden_dim (works for 3D tensors)
        if tensor.ndim >= 2:
            hidden_dim = tensor.shape[-1]
            flat = tensor.reshape(-1, hidden_dim)
            result = flat.mean(dim=0, keepdim=True)
            return result
        elif tensor.ndim == 1:
            return tensor.unsqueeze(0)
        
        return tensor.reshape(1, -1)


    def compute_importance_scores(self, num_samples: int = 1, dim="input") -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute importance scores for tracked layers using configured metrics.
        Returns mapping {layer_name: {metric_name: scores_tensor}}
        
        Supports two modes:
        1. Standard: Collect all activations, then compute metrics (fast for small models).
        2. Streaming: Compute metrics batch-by-batch (required for Llama-3 on 1M tokens).
        
        Also implements "Smart Redundancy": Only compute pairwise metrics for outlier candidates.
        """
        logger.info("Computing importance scores for LLM tracked layers...")

        # Build a list of calibration texts. Prefer a dataset with a `.texts` attribute;
        # otherwise, fall back to iterating the dataset or raise if no dataset is available.
        calibration_texts: List[str] = []
        if getattr(self, "dataset", None) is not None:
            if hasattr(self.dataset, "texts"):
                calibration_texts = list(self.dataset.texts)
            else:
                logger.warning(
                    "LLMAlignmentExperiment.dataset does not expose `.texts`; "
                    "falling back to iterating the dataset to extract raw text."
                )
                try:
                    for sample in self.dataset:
                        # Try common text fields
                        text = None
                        if isinstance(sample, dict):
                            for key in ("text", "raw_text", "input_text"):
                                if key in sample:
                                    text = sample[key]
                                    break
                        if isinstance(text, str) and text.strip():
                            calibration_texts.append(text)
                        if len(calibration_texts) >= num_samples:
                            break
                except Exception as e:
                    logger.error(f"Failed to iterate over dataset for calibration texts: {e}")
                    calibration_texts = []
        else:
            logger.error("No dataset available for LLM importance computation.")

        if not calibration_texts:
            raise RuntimeError(
                "Unable to obtain calibration texts for LLM importance computation. "
                "Ensure that `setup()` successfully created a text dataset with a `.texts` attribute "
                "or that the dataset yields samples containing a 'text' field."
            )

        num_samples = min(num_samples, len(calibration_texts))
        self.config.importance_computation_texts = calibration_texts[:num_samples]

        self.model.eval()
        
        # Check if we need streaming (heuristic: num_samples * context > 10k tokens for 8B model)
        # Actually, let's stick to standard accumulation for simplicity unless configured otherwise
        # But for Llama-3 SCAR, we usually run on ~500k tokens. That requires streaming for covariance.
        
        use_streaming = getattr(self.config, "use_streaming_metrics", False)
        
        # Initialize streaming objects if needed
        streaming_covs = {}
        if use_streaming:
            for layer_name in self.wrapped_model._tracked_layers:
                # We don't know dim yet, will init on first batch
                streaming_covs[f"{layer_name}_input"] = None 
        
        all_activations = {} # For non-streaming metrics (like OutlierIndex which needs quantiles)
        # Note: OutlierIndex usually needs full distribution. Streaming approx is hard. 
        # We'll assume we can fit sampled activations for OI, but use streaming for Covariance/RQ.

        # Pre-compute hidden dimensions for each tracked layer
        layer_dims = {}
        underlying_model = self._get_underlying_model()
        for layer_name in self.wrapped_model._tracked_layers:
            try:
                module = dict(underlying_model.named_modules()).get(layer_name)
                if module is not None and hasattr(module, 'weight'):
                    # For Linear: weight shape is [out_features, in_features]
                    in_dim = module.weight.shape[1]
                    out_dim = module.weight.shape[0]
                    layer_dims[f"{layer_name}_input"] = in_dim
                    layer_dims[f"{layer_name}_output"] = out_dim
                    layer_dims[layer_name] = out_dim  # Default for layer itself
                    logger.debug(f"Layer {layer_name}: in_dim={in_dim}, out_dim={out_dim}")
            except Exception as e:
                logger.warning(f"Could not get dims for {layer_name}: {e}")

        for text in calibration_texts[:num_samples]:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            outputs, activations = self.wrapped_model.forward_with_activations(inputs)

            # Process activations
            for key, value in activations.items():
                logger.debug(f"Raw activation {key}: shape={value.shape}, ndim={value.ndim}")
                # Get expected hidden_dim for this activation
                hidden_dim = layer_dims.get(key)
                normalized = self._normalize_activation(value, hidden_dim=hidden_dim)
                if normalized is None:
                    logger.warning(f"Normalization returned None for {key}")
                    continue
                logger.debug(f"Normalized activation {key}: shape={normalized.shape}")
                
                # Streaming Covariance Update
                if use_streaming and "input" in key:
                    if streaming_covs.get(key) is None:
                        streaming_covs[key] = StreamingCovariance(normalized.shape[1], device=self.config.device)
                    streaming_covs[key].update(normalized)
                
                # Store for other metrics (limit size if needed)
                if key not in all_activations:
                    all_activations[key] = []
                all_activations[key].append(normalized.cpu() if use_streaming else normalized)

        # Concatenate collected activations
        # Debug: check shapes before concatenation
        for key, values in all_activations.items():
            shapes = [v.shape for v in values[:5]]  # First 5 shapes
            logger.info(f"Before concat {key}: first 5 shapes = {shapes}, total = {len(values)} tensors")
        
        all_activations = {key: torch.cat(values, dim=0).to(self.config.device) for key, values in all_activations.items()}

        # Compute importance for each layer
        metric_names = self.config.alignment_methods

        for layer_name in self.wrapped_model._tracked_layers:
            logger.info(f"Computing scores for {layer_name}")

            layer_module = dict(self.wrapped_model._model.named_modules())[layer_name]

            layer_inputs = all_activations.get(f"{layer_name}_input")
            layer_outputs = all_activations.get(f"{layer_name}_output")

            if layer_inputs is None and layer_outputs is None and not use_streaming:
                logger.warning(f"No normalized activations for {layer_name}")
                continue

            # Get weight tensor (prefer gate_proj for MLP layers)
            weight = self._get_layer_weights(layer_module)
            if weight is None:
                continue

            default_activation = layer_inputs if dim == "input" else layer_outputs
            if default_activation is None:
                default_activation = layer_outputs if dim == "input" else layer_inputs

            # Compute scores with each metric
            layer_scores = {}
            
            # Candidates for redundancy (Smart Redundancy)
            redundancy_candidates = None
            
            # Pass 1: Compute independent metrics (RQ, OI, Magnitude)
            for metric_name in metric_names:
                # Skip pairwise for now
                if "redundancy" in metric_name or "synergy" in metric_name:
                    continue
                    
                try:
                    # Use already-initialized metric from self.metrics if available
                    if metric_name in self.metrics:
                        metric = self.metrics[metric_name]
                    else:
                        # Otherwise get fresh from registry without extra params
                        metric = get_metric(metric_name)

                    metric_args = {}

                    if getattr(metric, "requires_inputs", False):
                        if use_streaming and "rayleigh" in metric_name:
                            # Use streaming covariance for RQ
                            cov_key = f"{layer_name}_input"
                            if streaming_covs.get(cov_key):
                                metric_args["covariance"] = streaming_covs[cov_key].get_covariance()
                            else:
                                metric_args["inputs"] = layer_inputs
                        else:
                            metric_args["inputs"] = layer_inputs

                    if getattr(metric, "requires_outputs", False):
                        metric_args["outputs"] = layer_outputs

                    if getattr(metric, "requires_weights", False):
                        metric_args["weights"] = weight

                    if "inputs" not in metric_args and "outputs" not in metric_args and default_activation is not None:
                        metric_args["outputs"] = default_activation

                    scores = metric.compute(**metric_args)
                    layer_scores[metric_name] = scores

                    logger.debug(f"  {metric_name}: " f"mean={scores.mean().item():.6f}, " f"std={scores.std().item():.6f}")
                except Exception as e:
                    logger.error(f"Error computing {metric_name} for {layer_name}: {e}")
                    continue
            
            # Identify Supernode Candidates for Redundancy Reduction
            # We want to check redundancy mainly among high-activation nodes
            if "activation_outlier_index" in layer_scores:
                oi_scores = layer_scores["activation_outlier_index"]
                # Top 10% or threshold
                k = int(oi_scores.numel() * 0.1)
                _, redundancy_candidates = torch.topk(oi_scores, k)
            
            # Pass 2: Pairwise metrics (Redundancy/Synergy)
            for metric_name in metric_names:
                if "redundancy" not in metric_name and "synergy" not in metric_name:
                    continue
                    
                try:
                    if metric_name in self.metrics:
                        metric = self.metrics[metric_name]
                    else:
                        metric = get_metric(metric_name)
                        
                    metric_args = {}
                    
                    # Add inputs/weights/outputs
                    if getattr(metric, "requires_inputs", False):
                        metric_args["inputs"] = layer_inputs
                    if getattr(metric, "requires_outputs", False):
                        metric_args["outputs"] = layer_outputs
                    if getattr(metric, "requires_weights", False):
                        metric_args["weights"] = weight
                        
                    # SMART REDUNDANCY: Pass target indices
                    # Only compute redundancy for candidates
                    if redundancy_candidates is not None and "redundancy" in metric_name:
                        metric_args["target_indices"] = redundancy_candidates
                        # Also restrict partners to candidates? Or all?
                        # Usually we want to know if a candidate is redundant with ANYONE.
                        # But checking against all is slow. Checking against other candidates is O(K^2).
                        metric_args["allowed_partners"] = redundancy_candidates
                        logger.info(f"  Computing {metric_name} for {len(redundancy_candidates)} candidates only")

                    scores = metric.compute(**metric_args)
                    layer_scores[metric_name] = scores
                    
                except Exception as e:
                    logger.error(f"Error computing {metric_name} for {layer_name}: {e}")
                    continue

            composite_score = self._compute_composite_score(layer_scores)
            if composite_score is not None:
                layer_scores["composite"] = composite_score

            self._apply_supernode_selection(layer_scores, composite_score)

            self.importance_scores[layer_name] = layer_scores
        
        return self.importance_scores

    def compute_scar_supernode_metrics(
        self,
        num_samples: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute SCAR-style supernode metrics (activation power, first-order saliency, curvature, loss proxy)
        for FFN channels in transformer MLP layers.

        This routine performs a small number of full forward+backward passes on a calibration stream and uses
        lightweight hooks on the FFN down_proj modules:

        - u:     input to down_proj (post-gate FFN activations)
        - g_u:   gradient w.r.t. u
        - g_y:   gradient w.r.t. down_proj output y
        - W_down: down_proj weight

        Metrics per channel i:
            activation_power_i = E[u_i^2]
            taylor_i           = E[ | (g_u_i * u_i) | ]            (first-order saliency)
            curvature_i        = E[ (v_i^T g_y)^2 ]                (Rayleigh-style curvature along v_i)
            loss_proxy_i       = 0.5 * activation_power_i * curvature_i
        """
        if not getattr(self.config, "do_scar_metrics", False):
            logger.info("SCAR metrics disabled in config; skipping compute_scar_supernode_metrics.")
            return {}

        logger.info("Computing SCAR-style supernode metrics (T_i, R_i, L_i) for LLM FFN layers...")

        # Determine calibration texts
        # Prefer texts used for alignment importance if available
        calibration_texts: List[str] = []
        if getattr(self.config, "importance_computation_texts", None):
            calibration_texts = list(self.config.importance_computation_texts)
        else:
            # Fallback: rebuild from dataset if possible
            if getattr(self, "dataset", None) is not None:
                if hasattr(self.dataset, "texts"):
                    calibration_texts = list(self.dataset.texts)
                else:
                    logger.warning(
                        "SCAR metrics: dataset does not expose `.texts`; "
                        "falling back to iterating dataset for raw text."
                    )
                    try:
                        for sample in self.dataset:
                            text = None
                            if isinstance(sample, dict):
                                for key in ("text", "raw_text", "input_text"):
                                    if key in sample:
                                        text = sample[key]
                                        break
                            if isinstance(text, str) and text.strip():
                                calibration_texts.append(text)
                            if len(calibration_texts) >= (num_samples or self.config.alignment_data_num_samples):
                                break
                    except Exception as e:
                        logger.error(f"SCAR metrics: failed to iterate over dataset for texts: {e}")
                        calibration_texts = []

        if not calibration_texts:
            raise RuntimeError(
                "SCAR metrics: no calibration texts available. "
                "Run importance computation first or ensure the dataset provides raw texts."
            )

        # Limit number of samples and sequence length
        if num_samples is None or num_samples <= 0:
            num_samples = getattr(self.config, "scar_num_samples", 0) or self.config.alignment_data_num_samples
        max_length = max_length or getattr(self.config, "scar_max_length", 512)

        num_samples = min(num_samples, len(calibration_texts))
        logger.info(f"SCAR metrics will use {num_samples} calibration samples (max_length={max_length}).")

        device = torch.device(self.config.device)

        # Get underlying HF model (nn.Module with .named_modules())
        hf_model: nn.Module = self.model
        if hasattr(hf_model, "model"):
            hf_model = getattr(hf_model, "model")

        scar_state: Dict[str, Dict[str, Any]] = {}
        hooks: List[Any] = []

        # Create hooks on all FFN down_proj modules (LLaMA-style MLPs)
        for layer_name, module in hf_model.named_modules():
            if "mlp.down_proj" not in layer_name:
                continue

            scar_state[layer_name] = {
                "u_sqr_sum": None,  # sum over tokens of u^2
                "R_sum": None,      # sum over tokens of (v_i^T g_y)^2
                "T_sum": None,      # sum over tokens of |g_u_i * u_i|
                "count": 0,         # number of tokens seen
            }

            def make_hooks(name: str):
                def fwd_hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
                    # inputs[0] is u: post-gate FFN activations of shape [B, T, m] or [B*T, m]
                    if not inputs:
                        return
                    u = inputs[0]
                    if u is None:
                        return
                    # Ensure we track on the correct device/dtype
                    u_flat = u.detach()
                    if u_flat.ndim > 2:
                        u_flat = u_flat.reshape(-1, u_flat.shape[-1])  # [N_tokens, m]

                    state = scar_state[name]
                    m = u_flat.shape[-1]
                    if state["u_sqr_sum"] is None:
                        state["u_sqr_sum"] = torch.zeros(m, device=u_flat.device, dtype=u_flat.dtype)
                        state["R_sum"] = torch.zeros_like(state["u_sqr_sum"])
                        state["T_sum"] = torch.zeros_like(state["u_sqr_sum"])

                    state["u_sqr_sum"] += (u_flat * u_flat).sum(dim=0)
                    state["count"] += u_flat.shape[0]

                    # Store u for first-order saliency computation in backward
                    mod._scar_last_u = u.detach()

                def bwd_hook(mod: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
                    state = scar_state[name]

                    # Gradient w.r.t. module input (u)
                    if not grad_input or grad_input[0] is None:
                        return
                    if not grad_output or grad_output[0] is None:
                        return

                    g_u = grad_input[0]
                    g_y = grad_output[0]

                    if not hasattr(mod, "weight"):
                        return

                    weight = mod.weight  # [hidden_dim, m]

                    # Retrieve stored u from forward hook (if available)
                    if not hasattr(mod, "_scar_last_u"):
                        return

                    u = mod._scar_last_u
                    # Clean up to avoid holding onto large tensors longer than necessary
                    delattr(mod, "_scar_last_u")

                    # Flatten tensors to [N_tokens, *]
                    if u.ndim > 2:
                        u_flat = u.reshape(-1, u.shape[-1])
                    else:
                        u_flat = u.reshape(-1, u.shape[-1])

                    if g_u.ndim > 2:
                        g_u_flat = g_u.reshape(-1, g_u.shape[-1])
                    else:
                        g_u_flat = g_u.reshape(-1, g_u.shape[-1])

                    if g_y.ndim > 2:
                        g_y_flat = g_y.reshape(-1, g_y.shape[-1])
                    else:
                        g_y_flat = g_y.reshape(-1, g_y.shape[-1])

                    # Ensure shapes are consistent
                    if u_flat.shape != g_u_flat.shape:
                        logger.warning(
                            f"SCAR metrics: shape mismatch between u ({u_flat.shape}) and g_u ({g_u_flat.shape}) for layer {name}."
                        )
                        return

                    # Curvature: R_i = E[ (v_i^T g_y)^2 ]
                    # s = g_y * W_down  => [N_tokens, m]
                    try:
                        s_flat = torch.matmul(g_y_flat, weight)  # [N_tokens, m]
                    except Exception as e:
                        logger.error(f"SCAR metrics: failed to compute W_down^T g_y for layer {name}: {e}")
                        return

                    state["R_sum"] += (s_flat * s_flat).sum(dim=0)

                    # First-order Taylor saliency: E[ |g_u_i * u_i| ]
                    t_contrib = torch.abs(g_u_flat * u_flat).sum(dim=0)
                    state["T_sum"] += t_contrib

                return fwd_hook, bwd_hook

            fwd_hook, bwd_hook = make_hooks(layer_name)
            hooks.append(module.register_forward_hook(fwd_hook))
            hooks.append(module.register_full_backward_hook(bwd_hook))

        if not scar_state:
            logger.warning("SCAR metrics: no 'mlp.down_proj' modules found; skipping.")
            return {}

        # Calibration loop: forward + backward on a small number of samples
        self.model.eval()

        try:
            for idx, text in enumerate(calibration_texts[:num_samples]):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Create labels for language modeling loss (ignore padding)
                labels = inputs["input_ids"].clone()
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(
                    self.tokenizer, "eos_token_id", None
                )
                labels[labels == pad_token_id] = -100
                inputs["labels"] = labels

                self.model.zero_grad(set_to_none=True)

                outputs = self.model(**inputs)
                loss = outputs.loss

                loss.backward()

                logger.info(f"SCAR metrics: processed calibration sample {idx+1}/{num_samples}, loss={loss.item():.4f}")

        finally:
            # Always remove hooks, even if an error occurs
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # Aggregate metrics
        scar_scores: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_name, state in scar_state.items():
            count = state["count"]
            if count <= 0 or state["u_sqr_sum"] is None:
                continue

            u2_mean = state["u_sqr_sum"] / float(count)
            R_vals = state["R_sum"] / float(count)
            T_vals = state["T_sum"] / float(count)
            loss_proxy = 0.5 * u2_mean * R_vals

            scar_scores[layer_name] = {
                "scar_activation_power": u2_mean,
                "scar_taylor": T_vals,
                "scar_curvature": R_vals,
                "scar_loss_proxy": loss_proxy,
            }

            # Also attach these scores into importance_scores for later use in pruning
            layer_scores = self.importance_scores.get(layer_name, {})
            layer_scores["scar_activation_power"] = u2_mean
            layer_scores["scar_taylor"] = T_vals
            layer_scores["scar_curvature"] = R_vals
            layer_scores["scar_loss_proxy"] = loss_proxy
            self.importance_scores[layer_name] = layer_scores

        logger.info(f"SCAR metrics: computed metrics for {len(scar_scores)} FFN layers.")

        return scar_scores
    
    @staticmethod
    def _normalize_scores_tensor(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        min_val = torch.min(scores)
        max_val = torch.max(scores)
        if torch.isclose(max_val, min_val):
            return torch.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val + eps)

    def _compute_composite_score(self, layer_scores: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        weights = getattr(self.config, "alignment_composite_weights", {}) or {}
        mode = getattr(self.config, "score_composition_mode", "sum")  # "sum" or "product"
        
        if not weights:
            return None

        composite = None
        
        if mode == "product":
            # Start with 1.0
            composite = None
            for metric_name, weight in weights.items():
                if weight == 0:
                    continue
                
                metric_scores = layer_scores.get(metric_name)
                if metric_scores is None:
                    logger.debug(f"Composite score skipped metric '{metric_name}' (no data)")
                    continue
                
                # For product, we treat weight as exponent
                term = metric_scores.abs().pow(weight)
                
                if composite is None:
                    composite = term
                else:
                    composite = composite * term
                    
        else:
            # Sum mode (linear combination)
            for metric_name, weight in weights.items():
                if weight == 0:
                    continue
                metric_scores = layer_scores.get(metric_name)
                if metric_scores is None:
                    logger.debug(f"Composite score skipped metric '{metric_name}' (no data)")
                    continue
                
                normalized = self._normalize_scores_tensor(metric_scores)
                term = normalized * weight
                composite = term if composite is None else composite + term

        return composite

    def _apply_supernode_selection(self, layer_scores: Dict[str, torch.Tensor], composite: Optional[torch.Tensor]) -> None:
        config = getattr(self.config, "supernode_config", {}) or {}
        if not config.get("enabled"):
            return

        metric_name = config.get("score_metric", "composite")
        metric_scores = layer_scores.get(metric_name)
        if metric_scores is None and metric_name == "composite":
            metric_scores = composite

        if metric_scores is None:
            logger.warning(f"Supernode selection requested but metric '{metric_name}' is unavailable")
            return

        num_neurons = metric_scores.numel()
        if num_neurons == 0:
            return

        top_k = config.get("top_k")
        core_fraction = float(config.get("core_fraction", 0.1))
        min_core = max(1, int(config.get("min_core_neurons", 1)))

        if top_k is not None:
            num_core = min(num_neurons, int(top_k))
        else:
            num_core = max(1, int(round(core_fraction * num_neurons)))

        num_core = max(num_core, min_core)
        num_core = min(num_core, num_neurons)

        sorted_scores, sorted_indices = torch.sort(metric_scores, descending=True)
        top_indices = sorted_indices[:num_core]
        mask = torch.zeros_like(metric_scores, dtype=torch.bool)
        mask[top_indices] = True

        layer_scores["supernode_mask"] = mask
        layer_scores["supernode_core_size"] = num_core
        layer_scores["supernode_threshold"] = sorted_scores[min(num_core - 1, sorted_scores.shape[0] - 1)].item()

    def analyze_supernode_connections(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        follower_fraction: float = 0.10,
        plots_dir: Optional[Union[str, Path]] = None,
        supernode_metric: str = "scar_activation_power",
        cross_layer_analysis: bool = True,
        compute_metrics: Optional[List[str]] = None,
        compare_by_connection: bool = True,
        target_layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze supernode connections and their influence on downstream neurons.
        
        This analysis has two parts:
        
        1. **Same-Layer Analysis (down_proj):**
           - Identify supernodes in the INTERMEDIATE neurons (14336 dim) based on `supernode_metric`
           - These are the neurons INSIDE the FFN, before down_proj projects to hidden dim
           - Compute metrics (activation, RQ, MI, redundancy) for these intermediate neurons
           - Analyze outgoing weights from supernodes to the hidden dimension
        
        2. **Cross-Layer Analysis (optional, when cross_layer_analysis=True):**
           - Trace how supernodes influence the NEXT layer's input
           - The output of down_proj (4096 dim) feeds into the next transformer block
           - Identify "follower" neurons in the next layer's up_proj input
           - Compare metrics between high vs low supernode-connected neurons

        Args:
            scar_scores: SCAR metrics per layer (from compute_scar_supernode_metrics)
            supernode_fraction: Fraction of neurons to consider as supernodes (top by score)
            follower_fraction: Fraction of next-layer neurons to analyze by connection strength
            plots_dir: Directory to save plots
            supernode_metric: Metric to rank neurons for supernode identification
                Options: scar_activation_power, scar_taylor, scar_loss_proxy, 
                         rayleigh_quotient, mutual_information, activation_l2_norm
            cross_layer_analysis: Whether to analyze next layer's neurons
            compute_metrics: List of metrics to compute (activation, rayleigh_quotient, 
                           mutual_information, redundancy)
            compare_by_connection: Whether to compare high vs low connected neurons
            target_layers: List of layer names to analyze. If None or empty, analyzes all layers.
                         Can use patterns like "model.layers.10" or full names like 
                         "model.layers.10.mlp.down_proj"

        Returns:
            Dictionary with supernode analysis results
        """
        if compute_metrics is None:
            compute_metrics = ["activation", "rayleigh_quotient", "mutual_information", "redundancy"]
        
        logger.info(f"Analyzing supernode connections:")
        logger.info(f"  - Supernode metric: {supernode_metric}")
        logger.info(f"  - Supernode fraction: top {supernode_fraction*100:.1f}%")
        logger.info(f"  - Cross-layer analysis: {cross_layer_analysis}")
        if cross_layer_analysis:
            logger.info(f"  - Follower fraction: top {follower_fraction*100:.1f}%")
        if target_layers:
            logger.info(f"  - Target layers: {target_layers}")
        else:
            logger.info(f"  - Target layers: all layers with SCAR scores")

        if plots_dir is None:
            plots_dir = Path(getattr(self.config, "plots_dir", "./plots"))
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Get the underlying HF model
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        # Process each layer with SCAR scores
        for layer_name, layer_metrics in scar_scores.items():
            if "mlp.down_proj" not in layer_name:
                continue
            
            # Filter by target_layers if specified
            if target_layers:
                # Check if this layer matches any of the target patterns
                layer_matches = False
                for target in target_layers:
                    # Support both exact match and partial match (e.g., "model.layers.10" matches "model.layers.10.mlp.down_proj")
                    if target in layer_name or layer_name in target:
                        layer_matches = True
                        break
                if not layer_matches:
                    continue

            # Get the metric for supernode identification (configurable)
            supernode_scores = layer_metrics.get(supernode_metric)
            if supernode_scores is None:
                # Fallback to activation power if requested metric not available
                supernode_scores = layer_metrics.get("scar_activation_power")
                if supernode_scores is None:
                    logger.warning(f"  {layer_name}: No {supernode_metric} or fallback metric available, skipping")
                    continue
                logger.info(f"  {layer_name}: Using scar_activation_power as fallback (requested: {supernode_metric})")

            supernode_scores = supernode_scores.float().cpu()
            num_neurons = supernode_scores.numel()

            # Identify supernodes (top neurons by the selected metric)
            num_supernodes = max(1, int(supernode_fraction * num_neurons))
            sorted_vals, sorted_indices = torch.sort(supernode_scores, descending=True)
            supernode_indices = sorted_indices[:num_supernodes].numpy()
            supernode_scores_top = sorted_vals[:num_supernodes].numpy()

            logger.info(f"  {layer_name}: {num_supernodes} supernodes identified (by {supernode_metric})")

            # Get the down_proj weight matrix
            # down_proj has shape [hidden_dim, intermediate_dim] = [4096, 14336]
            # Each column corresponds to one intermediate neuron
            layer_idx = None
            for name, module in hf_model.named_modules():
                if name == layer_name or name.endswith(layer_name):
                    if hasattr(module, "weight"):
                        down_proj_weight = module.weight.detach().float().cpu()
                        # Extract layer index from name
                        import re
                        match = re.search(r"layers\.(\d+)", layer_name)
                        if match:
                            layer_idx = int(match.group(1))
                        break
            else:
                logger.warning(f"  Could not find weight for {layer_name}")
                continue

            # down_proj_weight: [hidden_dim=4096, intermediate_dim=14336]
            # Columns are the outgoing weights from each intermediate neuron

            # Get outgoing weights from supernodes
            supernode_weights = down_proj_weight[:, supernode_indices]  # [4096, num_supernodes]

            # Aggregate: for each output neuron, sum of absolute weights from supernodes
            supernode_influence = torch.abs(supernode_weights).sum(dim=1)  # [4096]

            # Identify "follower" neurons: those with highest total weight from supernodes
            num_followers = max(1, int(follower_fraction * supernode_influence.numel()))
            follower_vals, follower_indices = torch.sort(supernode_influence, descending=True)
            follower_indices = follower_indices[:num_followers].numpy()
            follower_weights = follower_vals[:num_followers].numpy()

            # Store results
            layer_results = {
                "num_supernodes": num_supernodes,
                "supernode_indices": supernode_indices.tolist(),
                "supernode_scores": supernode_scores_top.tolist(),
                "supernode_metric": supernode_metric,
                "num_followers": num_followers,
                "follower_indices": follower_indices.tolist(),
                "follower_weights": follower_weights.tolist(),
            }

            # Use UnifiedVisualizer for all plots
            viz = UnifiedVisualizer()
            layer_suffix = layer_name.replace('.', '_')

            # Plot 1: Distribution of supernode scores (based on selected metric)
            try:
                fig = viz.plot_supernode_activation_distribution(
                    activation_values=supernode_scores,
                    threshold_value=sorted_vals[num_supernodes-1].item(),
                    threshold_percentile=supernode_fraction,
                    layer_name=layer_name,
                    metric_name=supernode_metric,
                    save_path=plots_dir / f"supernode_score_dist_{layer_suffix}.png",
                )
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                logger.error(f"  Failed to plot supernode score distribution: {e}")

            # Plot 2: Histogram of outgoing weights from supernodes
            try:
                fig = viz.plot_outgoing_weights_distribution(
                    weights=supernode_weights,
                    layer_name=layer_name,
                    save_path=plots_dir / f"supernode_outgoing_weights_{layer_suffix}.png",
                )
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                logger.error(f"  Failed to plot outgoing weights: {e}")

            # Plot 3: Supernode influence on output neurons
            try:
                fig = viz.plot_supernode_influence(
                    influence_values=supernode_influence,
                    threshold_value=follower_vals[num_followers-1].item(),
                    threshold_percentile=follower_fraction,
                    layer_name=layer_name,
                    save_path=plots_dir / f"supernode_influence_{layer_suffix}.png",
                )
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                logger.error(f"  Failed to plot supernode influence: {e}")

            # =====================================================================
            # Cross-Layer Analysis (optional)
            # Analyze how supernodes in THIS layer influence NEXT layer's neurons
            # =====================================================================
            if cross_layer_analysis and layer_idx is not None and layer_idx < 31:
                next_layer_idx = layer_idx + 1
                next_layer_name = f"model.layers.{next_layer_idx}.mlp.up_proj"
                
                logger.info(f"  Cross-layer analysis: {layer_name} -> layer {next_layer_idx}")
                
                # Compute metrics for neurons in the NEXT layer, grouped by their
                # connection strength to supernodes in THIS layer
                try:
                    # follower_indices are indices into the hidden dimension (4096)
                    # These are the output positions of down_proj that have high weights from supernodes
                    # They become the INPUT to the next transformer block
                    next_layer_results = self._compute_next_layer_metrics(
                        follower_indices=follower_indices,
                        current_layer_name=layer_name,
                        next_layer_idx=next_layer_idx,
                        plots_dir=plots_dir,
                        compute_metrics=compute_metrics,
                    )
                    layer_results["next_layer_analysis"] = next_layer_results
                except Exception as e:
                    logger.error(f"  Failed to compute next layer metrics: {e}")
                
                # Compare metrics between high vs low supernode-connected neurons
                if compare_by_connection:
                    try:
                        comparison_results = self._compare_redundancy_by_supernode_connection(
                            supernode_influence=supernode_influence,
                            down_proj_weight=down_proj_weight,
                            layer_name=layer_name,
                            plots_dir=plots_dir,
                            follower_fraction=follower_fraction,
                        )
                        layer_results["connection_comparison"] = comparison_results
                    except Exception as e:
                        logger.error(f"  Failed to compute connection comparison: {e}")

            results[layer_name] = layer_results

        return results

    def _compute_next_layer_metrics(
        self,
        follower_indices: np.ndarray,
        current_layer_name: str,
        next_layer_idx: int,
        plots_dir: Path,
        compute_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics for neurons that receive high input from supernodes.
        
        Architecture context for LLaMA FFN:
        - Current layer: down_proj outputs to hidden dimension (4096)
        - These outputs are added to the residual stream
        - The residual feeds into the NEXT transformer block
        - Next block's up_proj receives the residual as input
        
        The `follower_indices` identify positions in the hidden dimension (4096)
        that have high total weight from supernodes in the intermediate dimension.
        We analyze how these positions behave as inputs to the next layer.
        
        Args:
            follower_indices: Indices into hidden dim with high supernode connection
            current_layer_name: Name of current layer (for logging/plotting)
            next_layer_idx: Index of the next transformer layer
            plots_dir: Directory to save plots
            compute_metrics: List of metrics to compute
            
        Returns:
            Dictionary with computed metrics and statistics
        """
        if compute_metrics is None:
            compute_metrics = ["activation", "rayleigh_quotient", "mutual_information", "redundancy"]
            
        logger.info(f"  Computing metrics for {len(follower_indices)} high-connection positions "
                    f"(inputs to layer {next_layer_idx})...")

        # We need to capture activations at the follower indices
        # These are the outputs of down_proj, which are inputs to the next transformer block
        
        # Get calibration texts
        calibration_texts = []
        if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
            calibration_texts = list(self.dataset.texts)[:8]
        
        if not calibration_texts:
            return {"error": "No calibration texts available"}

        # Capture activations at the residual stream (after down_proj output is added)
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        follower_activations = []
        input_activations = []  # For RQ computation (inputs to down_proj)
        
        # Hook to capture activations
        def capture_hook(module, inputs, outputs):
            # inputs[0] is the input to down_proj (intermediate activations)
            # outputs is the result after down_proj
            if inputs and inputs[0] is not None:
                inp = inputs[0].detach().float()
                if inp.ndim == 3:
                    inp = inp.reshape(-1, inp.shape[-1])
                input_activations.append(inp.cpu())
            
            if outputs is not None:
                out = outputs.detach().float()
                if out.ndim == 3:  # [B, T, D]
                    out = out.reshape(-1, out.shape[-1])  # [B*T, D]
                # Select only follower indices
                follower_acts = out[:, follower_indices]  # [B*T, num_followers]
                follower_activations.append(follower_acts.cpu())

        # Find the down_proj module
        hook_handle = None
        for name, module in hf_model.named_modules():
            if "mlp.down_proj" in name and name in current_layer_name:
                hook_handle = module.register_forward_hook(capture_hook)
                break

        if hook_handle is None:
            return {"error": f"Could not find module for {current_layer_name}"}

        # Run forward passes
        self.model.eval()
        with torch.no_grad():
            for text in calibration_texts[:4]:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=False,
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                try:
                    self.model(**inputs)
                except Exception:
                    pass

        hook_handle.remove()

        if not follower_activations:
            return {"error": "No activations captured"}

        # Concatenate all activations
        all_acts = torch.cat(follower_activations, dim=0)  # [total_tokens, num_followers]
        all_inputs = torch.cat(input_activations, dim=0) if input_activations else None  # [total_tokens, intermediate_dim]
        
        num_tokens = all_acts.shape[0]
        num_followers = all_acts.shape[1]
        
        # =====================================================================
        # Compute Covariance and Correlation matrices
        # =====================================================================
        acts_centered = all_acts - all_acts.mean(dim=0, keepdim=True)
        cov_matrix = (acts_centered.T @ acts_centered) / (num_tokens - 1)
        
        # Compute correlation matrix
        std = torch.sqrt(torch.diag(cov_matrix) + 1e-8)
        corr_matrix = cov_matrix / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
        corr_matrix = torch.clamp(corr_matrix, -1, 1)

        # Compute redundancy: average pairwise correlation (excluding diagonal)
        n = corr_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        pairwise_corr = corr_matrix[mask].abs()
        mean_redundancy = pairwise_corr.mean().item()
        max_redundancy = pairwise_corr.max().item()

        # =====================================================================
        # Compute Rayleigh Quotient (RQ) for each follower neuron
        # RQ_i = (w_i^T C_x w_i) / ||w_i||^2
        # where C_x is the input covariance and w_i is the weight vector for neuron i
        # =====================================================================
        rq_scores = torch.zeros(num_followers)
        
        # Get down_proj weights
        down_proj_weight = None
        for name, module in hf_model.named_modules():
            if "mlp.down_proj" in name and name in current_layer_name:
                if hasattr(module, "weight"):
                    down_proj_weight = module.weight.detach().float().cpu()
                    break
        
        if down_proj_weight is not None and all_inputs is not None:
            # down_proj_weight: [hidden_dim=4096, intermediate_dim=14336]
            # Each row is the weight vector for one output neuron
            
            # Compute input covariance
            inputs_centered = all_inputs - all_inputs.mean(dim=0, keepdim=True)
            input_cov = (inputs_centered.T @ inputs_centered) / (num_tokens - 1)
            
            # Regularize for numerical stability
            input_cov = input_cov + 1e-6 * torch.eye(input_cov.shape[0])
            
            # For each follower neuron, compute RQ
            for i, idx in enumerate(follower_indices):
                w = down_proj_weight[idx, :]  # [intermediate_dim]
                w_norm_sq = (w * w).sum() + 1e-8
                # RQ = w^T C_x w / ||w||^2
                wCw = w @ input_cov @ w
                rq_scores[i] = (wCw / w_norm_sq).item()
        
        # =====================================================================
        # Compute Gaussian Mutual Information (MI) for each follower neuron
        # MI_i = 0.5 * log(var(x_i) / var(x_i | others))
        # Approximated using correlation: MI ≈ -0.5 * log(1 - r^2)
        # =====================================================================
        mi_scores = torch.zeros(num_followers)
        
        # Compute variance of each follower
        variances = torch.var(all_acts, dim=0)
        
        # For MI, we compute how much each neuron's variance is explained by others
        # Using the average squared correlation as a proxy
        for i in range(num_followers):
            # Get correlations of neuron i with all others
            corr_with_others = corr_matrix[i, :].clone()
            corr_with_others[i] = 0  # Exclude self
            
            # Average squared correlation (R^2)
            r_squared = (corr_with_others ** 2).mean()
            
            # MI approximation: higher R^2 means more information shared
            # MI = -0.5 * log(1 - R^2) for Gaussian
            mi_scores[i] = -0.5 * torch.log(1 - r_squared.clamp(max=0.999) + 1e-8)
        
        # =====================================================================
        # Plot results
        # =====================================================================
        
        # Use UnifiedVisualizer for all plots
        viz = UnifiedVisualizer()
        layer_suffix = current_layer_name.replace('.', '_')
        import matplotlib.pyplot as plt
        
        # Create descriptive title prefix
        title_prefix = f"High-Connection Neurons (Layer {next_layer_idx} input)"

        # Plot correlation matrix
        try:
            fig = viz.plot_correlation_matrix(
                corr_matrix=corr_matrix,
                title=f"{title_prefix}\nPairwise Correlations (Mean |r|={mean_redundancy:.3f})",
                xlabel="Neuron Index (high supernode connection)",
                ylabel="Neuron Index (high supernode connection)",
                save_path=plots_dir / f"next_layer_correlation_{layer_suffix}.png",
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"  Failed to plot correlation matrix: {e}")

        # Plot histogram of pairwise correlations (redundancy)
        try:
            fig = viz.plot_1d_histogram(
                values=pairwise_corr,
                xlabel="Absolute Pairwise Correlation",
                ylabel="Count",
                title=f"{title_prefix}\nRedundancy Distribution",
                vline=mean_redundancy,
                vline_label=f"Mean: {mean_redundancy:.3f}",
                save_path=plots_dir / f"next_layer_redundancy_hist_{layer_suffix}.png",
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"  Failed to plot redundancy histogram: {e}")

        # Plot RQ distribution
        try:
            fig = viz.plot_1d_histogram(
                values=rq_scores,
                xlabel="Rayleigh Quotient",
                ylabel="Count",
                title=f"{title_prefix}\nRQ Distribution",
                vline=rq_scores.mean().item(),
                vline_label=f"Mean: {rq_scores.mean().item():.4f}",
                color='green',
                save_path=plots_dir / f"next_layer_rq_hist_{layer_suffix}.png",
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"  Failed to plot RQ histogram: {e}")

        # Plot MI distribution
        try:
            fig = viz.plot_1d_histogram(
                values=mi_scores,
                xlabel="Mutual Information (Gaussian approx)",
                ylabel="Count",
                title=f"{title_prefix}\nMI Distribution",
                vline=mi_scores.mean().item(),
                vline_label=f"Mean: {mi_scores.mean().item():.4f}",
                color='purple',
                save_path=plots_dir / f"next_layer_mi_hist_{layer_suffix}.png",
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"  Failed to plot MI histogram: {e}")

        # Plot combined metrics: RQ vs MI scatter
        try:
            redundancy_for_color = pairwise_corr[:num_followers] if len(pairwise_corr) >= num_followers else None
            fig = viz.plot_rq_vs_mi(
                rq_scores=rq_scores,
                mi_scores=mi_scores,
                redundancy_scores=redundancy_for_color,
                layer_name=f"{title_prefix}",
                save_path=plots_dir / f"next_layer_rq_vs_mi_{layer_suffix}.png",
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"  Failed to plot RQ vs MI: {e}")

        # Summary statistics
        results = {
            "description": f"Metrics for neurons with high supernode connection (layer {next_layer_idx} input)",
            "source_layer": current_layer_name,
            "target_layer_idx": next_layer_idx,
            "num_high_connection_neurons": len(follower_indices),
            "num_tokens_analyzed": num_tokens,
            "redundancy": {
                "mean": mean_redundancy,
                "max": max_redundancy,
                "std": pairwise_corr.std().item(),
            },
            "rayleigh_quotient": {
                "mean": rq_scores.mean().item(),
                "std": rq_scores.std().item(),
                "min": rq_scores.min().item(),
                "max": rq_scores.max().item(),
            },
            "mutual_information": {
                "mean": mi_scores.mean().item(),
                "std": mi_scores.std().item(),
                "min": mi_scores.min().item(),
                "max": mi_scores.max().item(),
            },
        }
        
        logger.info(f"    Metrics for high-connection neurons (next layer input):")
        logger.info(f"      Redundancy: mean={mean_redundancy:.4f}")
        logger.info(f"      RQ: mean={rq_scores.mean().item():.4f}, std={rq_scores.std().item():.4f}")
        logger.info(f"      MI: mean={mi_scores.mean().item():.4f}, std={mi_scores.std().item():.4f}")

        return results

    def _compare_redundancy_by_supernode_connection(
        self,
        supernode_influence: torch.Tensor,
        down_proj_weight: torch.Tensor,
        layer_name: str,
        plots_dir: Path,
        follower_fraction: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Compare redundancy between neurons with high vs low weight connections to supernodes.
        
        This analysis helps understand whether neurons strongly connected to supernodes
        exhibit different redundancy patterns compared to neurons weakly connected.
        
        Args:
            supernode_influence: Total absolute weight from supernodes for each output neuron [hidden_dim]
            down_proj_weight: Weight matrix of down_proj [hidden_dim, intermediate_dim]
            layer_name: Name of the layer for logging/plotting
            plots_dir: Directory to save plots
            follower_fraction: Fraction of neurons to consider as "high" or "low" connected
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"  Comparing redundancy: high vs low supernode-connected neurons...")
        
        hidden_dim = supernode_influence.numel()
        num_group = max(1, int(follower_fraction * hidden_dim))
        
        # Sort neurons by supernode influence
        sorted_influence, sorted_indices = torch.sort(supernode_influence, descending=True)
        
        # High-connected neurons (top follower_fraction)
        high_indices = sorted_indices[:num_group].numpy()
        high_influence_values = sorted_influence[:num_group].numpy()
        
        # Low-connected neurons (bottom follower_fraction)
        low_indices = sorted_indices[-num_group:].numpy()
        low_influence_values = sorted_influence[-num_group:].numpy()
        
        logger.info(f"    High-connected group: {num_group} neurons, influence range [{high_influence_values[-1]:.4f}, {high_influence_values[0]:.4f}]")
        logger.info(f"    Low-connected group: {num_group} neurons, influence range [{low_influence_values[-1]:.4f}, {low_influence_values[0]:.4f}]")
        
        # Capture activations for both groups
        calibration_texts = []
        if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
            calibration_texts = list(self.dataset.texts)[:8]
        
        if not calibration_texts:
            return {"error": "No calibration texts available"}
        
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model
        
        high_activations = []
        low_activations = []
        
        def capture_hook(module, inputs, outputs):
            if outputs is not None:
                out = outputs.detach().float()
                if out.ndim == 3:
                    out = out.reshape(-1, out.shape[-1])
                high_activations.append(out[:, high_indices].cpu())
                low_activations.append(out[:, low_indices].cpu())
        
        # Find and hook the down_proj module
        hook_handle = None
        for name, module in hf_model.named_modules():
            if "mlp.down_proj" in name and name in layer_name:
                hook_handle = module.register_forward_hook(capture_hook)
                break
        
        if hook_handle is None:
            return {"error": f"Could not find module for {layer_name}"}
        
        # Run forward passes
        self.model.eval()
        with torch.no_grad():
            for text in calibration_texts[:4]:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=False,
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                try:
                    self.model(**inputs)
                except Exception:
                    pass
        
        hook_handle.remove()
        
        if not high_activations or not low_activations:
            return {"error": "No activations captured"}
        
        # Concatenate activations
        high_acts = torch.cat(high_activations, dim=0)  # [total_tokens, num_group]
        low_acts = torch.cat(low_activations, dim=0)    # [total_tokens, num_group]
        
        num_tokens = high_acts.shape[0]
        
        # =====================================================================
        # Compute pairwise redundancy (correlation) for each group
        # =====================================================================
        def compute_group_redundancy(acts: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
            """Compute pairwise correlation stats for a group of neurons."""
            acts_centered = acts - acts.mean(dim=0, keepdim=True)
            cov = (acts_centered.T @ acts_centered) / (num_tokens - 1)
            std = torch.sqrt(torch.diag(cov) + 1e-8)
            corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
            corr = torch.clamp(corr, -1, 1)
            
            n = corr.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool)
            pairwise = corr[mask].abs()
            return pairwise, pairwise.mean().item(), pairwise.std().item()
        
        high_pairwise, high_mean_redundancy, high_std_redundancy = compute_group_redundancy(high_acts)
        low_pairwise, low_mean_redundancy, low_std_redundancy = compute_group_redundancy(low_acts)
        
        logger.info(f"    High-connected redundancy: mean={high_mean_redundancy:.4f}, std={high_std_redundancy:.4f}")
        logger.info(f"    Low-connected redundancy: mean={low_mean_redundancy:.4f}, std={low_std_redundancy:.4f}")
        
        # =====================================================================
        # Statistical comparison
        # =====================================================================
        redundancy_diff = high_mean_redundancy - low_mean_redundancy
        
        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt((high_std_redundancy**2 + low_std_redundancy**2) / 2)
        effect_size = redundancy_diff / (pooled_std + 1e-8)
        
        logger.info(f"    Redundancy difference (high - low): {redundancy_diff:.4f}")
        logger.info(f"    Effect size (Cohen's d): {effect_size:.4f}")
        
        # =====================================================================
        # Plot comparison using UnifiedVisualizer
        # =====================================================================
        viz = UnifiedVisualizer()
        import matplotlib.pyplot as plt
        
        # Plots 1-3: Redundancy comparison (side-by-side, overlay, boxplot)
        try:
            figs = viz.plot_redundancy_comparison(
                high_redundancy=high_pairwise,
                low_redundancy=low_pairwise,
                high_mean=high_mean_redundancy,
                low_mean=low_mean_redundancy,
                layer_name=layer_name,
                follower_fraction=follower_fraction,
                save_dir=plots_dir,
            )
            for fig in figs:
                plt.close(fig)
        except Exception as e:
            logger.error(f"  Failed to plot redundancy comparison: {e}")
        
        # Plot 4: Scatter plot - supernode influence vs mean redundancy per neuron
        try:
            # For each neuron, compute its mean correlation with others in its group
            # High group: per-neuron mean correlation
            high_centered = high_acts - high_acts.mean(dim=0, keepdim=True)
            high_cov = (high_centered.T @ high_centered) / (num_tokens - 1)
            high_std = torch.sqrt(torch.diag(high_cov) + 1e-8)
            high_corr = high_cov / (high_std.unsqueeze(0) * high_std.unsqueeze(1) + 1e-8)
            high_corr = torch.clamp(high_corr, -1, 1)
            high_corr.fill_diagonal_(0)  # Exclude self
            high_per_neuron_redundancy = high_corr.abs().mean(dim=1).numpy()
            
            # Low group: per-neuron mean correlation
            low_centered = low_acts - low_acts.mean(dim=0, keepdim=True)
            low_cov = (low_centered.T @ low_centered) / (num_tokens - 1)
            low_std = torch.sqrt(torch.diag(low_cov) + 1e-8)
            low_corr = low_cov / (low_std.unsqueeze(0) * low_std.unsqueeze(1) + 1e-8)
            low_corr = torch.clamp(low_corr, -1, 1)
            low_corr.fill_diagonal_(0)
            low_per_neuron_redundancy = low_corr.abs().mean(dim=1).numpy()
            
            # Combine data for grouped scatter
            all_influence = np.concatenate([high_influence_values, low_influence_values])
            all_redundancy = np.concatenate([high_per_neuron_redundancy, low_per_neuron_redundancy])
            all_labels = ['High'] * len(high_influence_values) + ['Low'] * len(low_influence_values)
            
            fig = viz.plot_metric_scatter_by_group(
                x_values=all_influence,
                y_values=all_redundancy,
                group_labels=all_labels,
                xlabel="Supernode Influence (Total Abs Weight)",
                ylabel="Mean Redundancy (Avg |Correlation| with Group)",
                title=f"Supernode Influence vs Redundancy per Neuron\n{layer_name}",
                save_path=plots_dir / f"redundancy_vs_influence_scatter_{layer_name.replace('.', '_')}.png",
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"  Failed to plot scatter comparison: {e}")
        
        # =====================================================================
        # Results
        # =====================================================================
        results = {
            "high_connected": {
                "num_neurons": num_group,
                "influence_range": [float(high_influence_values[-1]), float(high_influence_values[0])],
                "redundancy_mean": high_mean_redundancy,
                "redundancy_std": high_std_redundancy,
            },
            "low_connected": {
                "num_neurons": num_group,
                "influence_range": [float(low_influence_values[-1]), float(low_influence_values[0])],
                "redundancy_mean": low_mean_redundancy,
                "redundancy_std": low_std_redundancy,
            },
            "comparison": {
                "redundancy_difference": redundancy_diff,
                "effect_size_cohens_d": effect_size,
            },
        }
        
        return results

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
            
            # Get importance scores
            scores = self.importance_scores[layer_name][metric].clone()

            supernode_cfg = getattr(self.config, "supernode_config", {}) or {}
            core_mask = self.importance_scores[layer_name].get("supernode_mask")
            if supernode_cfg.get("enabled") and supernode_cfg.get("protect_core", True) and core_mask is not None:
                margin = torch.abs(scores).max().detach().item() + 1.0
                if mode == "low":
                    scores[core_mask] = scores.max() + margin
                elif mode == "high":
                    scores[core_mask] = scores.min() - margin

            # Create mask based on importance scores
            mask = pruner.create_pruning_mask(scores)
            
            # Get the MLP module - use underlying model to handle HFCausalLM wrapper
            underlying_model = self._get_underlying_model()
            module_dict = dict(underlying_model.named_modules())
            
            # Try different module path patterns for compatibility
            mlp_path = f"model.layers.{layer_idx}.mlp"
            if mlp_path not in module_dict:
                # Try without 'model.' prefix (for direct HF models)
                mlp_path = f"layers.{layer_idx}.mlp"
            if mlp_path not in module_dict:
                logger.warning(f"Could not find MLP module for layer {layer_idx}")
                continue
            
            mlp_module = module_dict[mlp_path]
            
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

        attention_masks, num_attention_layers = self._prune_attention_layers(
            pruner=pruner,
            metric=metric,
            mode=mode,
            sparsity=sparsity,
        )
        masks.update(attention_masks)

        self.pruning_masks = masks
        logger.info(f"Pruned {len(processed_mlps)} MLP layers with {sparsity:.1%} target sparsity")
        if num_attention_layers > 0:
            logger.info(f"Pruned {num_attention_layers} attention blocks with shared Q/K/V/O masks")
        return masks

    def _prune_attention_layers(
        self,
        pruner: AlignmentPruning,
        metric: str,
        mode: str,
        sparsity: float,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Apply shared pruning masks to attention Q/K/V/O projections so that entire heads
        are dropped consistently.
        """
        import re

        attention_masks: Dict[str, torch.Tensor] = {}
        processed_layers = set()
        successful_layers = 0

        named_modules = dict(self.wrapped_model._model.named_modules())
        pattern = re.compile(r"layers\.(\d+)\.self_attn")

        for layer_name, layer_scores in self.importance_scores.items():
            if metric not in layer_scores:
                continue

            match = pattern.search(layer_name)
            if not match:
                continue

            layer_idx = match.group(1)
            if layer_idx in processed_layers:
                continue
            processed_layers.add(layer_idx)

            base_name = f"model.layers.{layer_idx}.self_attn"
            attn_module = named_modules.get(base_name)
            if attn_module is None:
                logger.warning(f"Attention module '{base_name}' not found; skipping attention pruning for layer {layer_idx}")
                continue

            scores, ref_layer = self._select_attention_scores(base_name, metric)
            if scores is None or ref_layer is None:
                logger.warning(f"No attention scores found for {base_name} using metric '{metric}'")
                continue

            neuron_mask, heads_kept, total_heads = self._create_attention_neuron_mask(
                scores=scores,
                attn_module=attn_module,
                mode=mode,
                sparsity=sparsity,
                layer_key=ref_layer,
            )
            if neuron_mask is None:
                continue

            # Apply mask to Q/K/V outputs (rows) and O input (columns)
            devices = []
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj_module = getattr(attn_module, proj_name, None)
                if proj_module is None:
                    continue
                devices.append(proj_module.weight.device)
                pruner.apply_pruning(proj_module, neuron_mask.to(proj_module.weight.device), dim="output")
                attention_masks[f"{base_name}.{proj_name}"] = neuron_mask.detach().clone()

            o_proj = getattr(attn_module, "o_proj", None) or getattr(attn_module, "out_proj", None)
            if o_proj is not None:
                devices.append(o_proj.weight.device)
                pruner.apply_pruning(o_proj, neuron_mask.to(o_proj.weight.device), dim="input")
                attention_masks[f"{base_name}.o_proj"] = neuron_mask.detach().clone()

            if devices:
                successful_layers += 1
                pruned_fraction = float((neuron_mask == 0).sum().item()) / float(neuron_mask.numel())
                if heads_kept is not None and total_heads is not None:
                    logger.info(
                        f"  Layer {layer_idx} attention: kept {heads_kept}/{total_heads} heads "
                        f"({1 - pruned_fraction:.2%} of Q/K/V outputs retained)"
                    )
                else:
                    logger.info(
                        f"  Layer {layer_idx} attention: pruned {pruned_fraction:.2%} of Q/K/V outputs"
                    )

        return attention_masks, successful_layers

    def _select_attention_scores(self, base_name: str, metric: str) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Find the first projection within an attention block that has the requested metric."""
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj"):
            key = f"{base_name}.{proj}"
            layer_scores = self.importance_scores.get(key)
            if not layer_scores:
                continue
            metric_scores = layer_scores.get(metric)
            if metric_scores is None:
                continue
            return metric_scores.clone(), key
        return None, None

    def _create_attention_neuron_mask(
        self,
        scores: torch.Tensor,
        attn_module: nn.Module,
        mode: str,
        sparsity: float,
        layer_key: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[int]]:
        """
        Convert per-neuron attention scores into a shared mask aligned with heads.
        Returns (mask, heads_kept, total_heads).
        """
        scores = scores.flatten()
        device = scores.device

        supernode_cfg = getattr(self.config, "supernode_config", {}) or {}
        core_mask = self.importance_scores.get(layer_key, {}).get("supernode_mask")
        if supernode_cfg.get("enabled") and supernode_cfg.get("protect_core", True) and core_mask is not None:
            margin = torch.abs(scores).max().detach().item() + 1.0
            if mode == "low":
                scores[core_mask] = scores.max() + margin
            elif mode == "high":
                scores[core_mask] = scores.min() - margin

        num_heads = None
        for attr in ("num_heads", "n_heads", "num_attention_heads"):
            if hasattr(attn_module, attr):
                num_heads = int(getattr(attn_module, attr))
                break

        if num_heads is None or num_heads <= 0:
            logger.warning("Attention module missing head count; falling back to per-neuron mask")
            raw_mask = MaskOperations.create_structured_mask(scores, amount=sparsity, mode=mode)
            return raw_mask.float().to(device), None, None

        head_dim = getattr(attn_module, "head_dim", None)
        if head_dim is None and hasattr(attn_module, "hidden_size"):
            head_dim = getattr(attn_module, "hidden_size") // num_heads
        if head_dim is None and hasattr(attn_module, "embed_dim"):
            head_dim = getattr(attn_module, "embed_dim") // num_heads
        if head_dim is None and scores.numel() % num_heads == 0:
            head_dim = scores.numel() // num_heads

        if head_dim is None or head_dim <= 0 or scores.numel() != num_heads * head_dim:
            logger.warning(
                f"Attention score length {scores.numel()} is incompatible with num_heads={num_heads}; "
                f"falling back to per-neuron mask."
            )
            raw_mask = MaskOperations.create_structured_mask(scores, amount=sparsity, mode=mode)
            return raw_mask.float().to(device), None, None

        head_scores = scores.view(num_heads, head_dim).mean(dim=1)
        head_keep = MaskOperations.create_structured_mask(head_scores, amount=sparsity, mode=mode)

        # Ensure that any head containing a protected core neuron is always kept.
        if core_mask is not None and core_mask.numel() == scores.numel():
            core_heads = core_mask.view(num_heads, head_dim).any(dim=1)
            if core_heads.any():
                head_keep = head_keep | core_heads.to(head_keep.device)

        heads_kept = int(head_keep.sum().item())

        expanded = head_keep.unsqueeze(1).expand(-1, head_dim).reshape(-1).float()
        return expanded.to(device), heads_kept, num_heads
    
    def apply_minimal_repair(self, dataset_name: str = "wikitext", epochs: int = 1, lr: float = 1e-4) -> None:
        """
        Apply Minimal Repair (LoRA) to the pruned model.
        Target supernode-adjacent weights or all MLP weights.
        """
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            logger.error("PEFT library not installed. Cannot run minimal repair.")
            return

        logger.info(f"Applying Minimal Repair (LoRA) for {epochs} epochs...")

        # Configure LoRA
        # We target the projection layers in MLPs.
        target_modules = ["gate_proj", "up_proj", "down_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=target_modules
        )
        
        # Wrap model
        # Note: We are wrapping the HUGGINGFACE model, not our wrapper
        # Our wrapper wrapper_model.model or similar needs to be accessed
        hf_model = self.model # This is the AutoModelForCausalLM
        
        # Enable gradients for LoRA
        hf_model.enable_input_require_grads()
        
        model = get_peft_model(hf_model, peft_config)
        model.print_trainable_parameters()
        
        # Create trainer
        # Need a dataset loader
        from alignment.dataops.datasets.text_datasets import load_text_dataset
        from torch.utils.data import DataLoader
        
        # Minimal dataset for repair (calibration set)
        dataset = load_text_dataset(dataset_name, self.config.model_config.get("model_id"), split="train", max_samples=1000)
        
        # Create a simple collator if needed, or use default
        def collate_fn(batch):
            input_ids = [b['input_ids'] for b in batch]
            # Pad
            from torch.nn.utils.rnn import pad_sequence
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return input_ids, labels

        train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        # Simple training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        model.train()
        model.to(self.config.device)
        
        for epoch in range(epochs):
            total_loss = 0
            for step, (input_ids, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                if step % 10 == 0:
                    logger.info(f"Repair Epoch {epoch} Step {step}: Loss {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Repair Epoch {epoch} Average Loss: {avg_loss:.4f}")
            
        # Merge LoRA weights back if desired, or keep as adapter
        # For evaluation, we usually merge
        model = model.merge_and_unload()
        self.model = model # Update self.model to the repaired one
        
        # Update wrapper reference if needed (wrapper usually holds reference to self.model)
        # Check if wrapper needs update
        if hasattr(self.wrapped_model, "model"):
             self.wrapped_model.model = model
        elif hasattr(self.wrapped_model, "_model"):
             self.wrapped_model._model = model

        logger.info("Minimal Repair complete.")


    def run(self) -> Dict[str, Any]:
        """Run the full LLM experiment pipeline: compute importance, optionally prune, evaluate."""
        logger.info("Running LLMAlignmentExperiment...")

        self.setup()

        results: Dict[str, Any] = {"config": self.config.to_dict(), "importance_scores": {}, "pruning_results": {}, "evaluation": {}}

        scores = self.compute_importance_scores(
            num_samples=self.config.alignment_data_num_samples
        )

        # Optional: SCAR-style supernode metrics (T_i, R_i, loss proxy) for FFN layers
        scar_scores: Dict[str, Any] = {}
        if getattr(self.config, "do_scar_metrics", False):
            try:
                scar_scores = self.compute_scar_supernode_metrics()
            except Exception as e:
                logger.error(f"Error while computing SCAR supernode metrics: {e}")
            else:
                if scar_scores and getattr(self.config, "generate_plots", True):
                    try:
                        import matplotlib.pyplot as plt

                        plots_dir = Path(getattr(self.config, "plots_dir", Path(self.config.log_dir) / "plots"))
                        plots_dir.mkdir(parents=True, exist_ok=True)

                        # Convert bfloat16 tensors to float32 for matplotlib compatibility
                        scar_scores_float32 = {}
                        for layer_name, layer_metrics in scar_scores.items():
                            scar_scores_float32[layer_name] = {}
                            for metric_name, values in layer_metrics.items():
                                if torch.is_tensor(values):
                                    scar_scores_float32[layer_name][metric_name] = values.float()
                                else:
                                    scar_scores_float32[layer_name][metric_name] = values

                        viz = UnifiedVisualizer()

                        # Layer-wise SCAR loss proxy distributions
                        fig = viz.plot_scar_layer_scores(
                            scar_scores_float32,
                            metric_name="scar_loss_proxy",
                            plot_type="violin",
                            save_path=plots_dir / "scar_loss_proxy_layers.png",
                        )
                        plt.close(fig)

                        # Heatmap of SCAR metrics (activation power, curvature, loss proxy, etc.)
                        scar_metric_list = [
                            "scar_activation_power",
                            "scar_taylor",
                            "scar_curvature",
                            "scar_loss_proxy",
                        ]
                        fig = viz.plot_scar_heatmap(
                            scar_scores_float32,
                            metrics=scar_metric_list,
                            title="SCAR Metrics per Layer",
                            save_path=plots_dir / "scar_metrics_heatmap.png",
                        )
                        plt.close(fig)
                    except Exception as viz_err:
                        logger.error(f"Failed to generate SCAR visualizations: {viz_err}")

                    # Run supernode connection analysis
                    try:
                        supernode_config = getattr(self.config, "supernode_config", {}) or {}
                        supernode_fraction = supernode_config.get("core_fraction", 0.01)
                        follower_fraction = supernode_config.get("follower_fraction", 0.10)
                        supernode_metric = supernode_config.get("score_metric", "scar_activation_power")
                        cross_layer_analysis = supernode_config.get("cross_layer_analysis", True)
                        compute_metrics = supernode_config.get("compute_metrics", 
                            ["activation", "rayleigh_quotient", "mutual_information", "redundancy"])
                        compare_by_connection = supernode_config.get("compare_by_connection", True)
                        
                        # Get target layers from config - use tracked_layers if not specified
                        # If target_layers is empty list or None, analyze all layers
                        target_layers = supernode_config.get("target_layers", None)
                        if target_layers is None:
                            # Use tracked layers from config as default
                            target_layers = getattr(self.config, "tracked_layers", None)
                        
                        supernode_analysis = self.analyze_supernode_connections(
                            scar_scores=scar_scores,
                            supernode_fraction=supernode_fraction,
                            follower_fraction=follower_fraction,
                            plots_dir=plots_dir,
                            supernode_metric=supernode_metric,
                            cross_layer_analysis=cross_layer_analysis,
                            compute_metrics=compute_metrics,
                            compare_by_connection=compare_by_connection,
                            target_layers=target_layers,
                        )
                        results["supernode_analysis"] = supernode_analysis
                        logger.info("Supernode connection analysis complete")
                    except Exception as sn_err:
                        logger.error(f"Failed supernode connection analysis: {sn_err}")

        # Example: per-layer histogram with top-5 annotations
        # self.plot_layer_importance_histogram(
        #     layer_name="model.layers.1.mlp.up_proj",
        #     metric="activation_l2_norm",
        #     importance_scores=scores,
        #     plots_dir=self.config.plots_dir,
        # )

        for layer_name, layer_scores in scores.items():
            results["importance_scores"][layer_name] = {}
            for metric_name, vals in layer_scores.items():
                if torch.is_tensor(vals):
                    try:
                        results["importance_scores"][layer_name][metric_name] = {
                            "mean": float(vals.mean().item()),
                            "std": float(vals.std().item()),
                            "min": float(vals.min().item()),
                            "max": float(vals.max().item()),
                        }
                    except Exception:
                        results["importance_scores"][layer_name][metric_name] = {"summary": "unavailable"}
                else:
                    results["importance_scores"][layer_name][metric_name] = vals

        # Add SCAR metrics summaries (if any)
        if scar_scores:
            results["scar_scores"] = {}
            for layer_name, scar_layer_scores in scar_scores.items():
                results["scar_scores"][layer_name] = {}
                for metric_name, vals in scar_layer_scores.items():
                    if torch.is_tensor(vals):
                        try:
                            results["scar_scores"][layer_name][metric_name] = {
                                "mean": float(vals.mean().item()),
                                "std": float(vals.std().item()),
                                "min": float(vals.min().item()),
                                "max": float(vals.max().item()),
                            }
                        except Exception:
                            results["scar_scores"][layer_name][metric_name] = {"summary": "unavailable"}
                    else:
                        results["scar_scores"][layer_name][metric_name] = vals

        if self.config.do_perplexity_computation:
            baseline_ppl = self.evaluate_perplexity(dataset=self.config.evaluation_dataset, num_samples=self.config.evaluation_num_samples)
            results["evaluation"]["baseline_perplexity"] = baseline_ppl


        if self.config.do_pruning_experiments:
            sparsity_levels = self.config.pruning_amounts
            metric = self.config.pruning_alignment_metric
            mode = self.config.pruning_selection_mode
            if isinstance(mode, list):
                mode = mode[0]

            # Collect pruning results for visualization
            pruning_data = {
                "sparsity_levels": [],
                "perplexities": [],
                "baseline_perplexity": results.get("evaluation", {}).get("baseline_perplexity", None),
            }

            for sparsity in sparsity_levels:
                logger.info(f"Applying pruning: sparsity={sparsity}, metric={metric}, mode={mode}")
                masks = self.apply_pruning(sparsity=sparsity, mode=mode, metric=metric)

                pruning_data["sparsity_levels"].append(sparsity)

                # Evaluate pruned model
                if self.config.do_perplexity_computation:
                    pruned_ppl = self.evaluate_perplexity(
                        dataset=self.config.evaluation_dataset, num_samples=self.config.evaluation_num_samples
                    )
                    pruning_data["perplexities"].append(pruned_ppl)

                    results["pruning_results"][f"sparsity_{sparsity}"] = {
                        "perplexity": pruned_ppl,
                        "sparsity": sparsity,
                        "num_pruned_layers": len(masks),
                    }
                else:
                    pruning_data["perplexities"].append(None)

            # Generate pruning visualization
            if getattr(self.config, "generate_plots", True) and pruning_data["perplexities"]:
                try:
                    import matplotlib.pyplot as plt
                    plots_dir = Path(getattr(self.config, "plots_dir", Path(self.config.log_dir) / "plots"))
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    
                    viz = UnifiedVisualizer()
                    
                    # Filter out None values for plotting
                    valid_data = [(s, p) for s, p in zip(pruning_data["sparsity_levels"], pruning_data["perplexities"]) if p is not None]
                    if valid_data:
                        sparsities, perplexities = zip(*valid_data)
                        
                        # Plot perplexity vs sparsity curve
                        save_path = plots_dir / f"pruning_curve_{metric}_{mode}.png"
                        fig = viz.plot_sparsity_performance(
                            sparsities=list(sparsities),
                            perplexities=list(perplexities),
                            strategy_name=f"{metric} ({mode})",
                            baseline_ppl=pruning_data["baseline_perplexity"],
                            title=f"Pruning Performance: Perplexity vs Sparsity",
                            save_path=save_path,
                        )
                        plt.close(fig)
                        logger.info(f"Saved pruning curve to {save_path}")
                except Exception as e:
                    logger.error(f"Failed to generate pruning visualization: {e}")

        return results
    

    def plot_layer_importance_histogram(
        self,
        layer_name: str,
        metric: str,
        importance_scores: Dict[str, Dict[str, torch.Tensor]],
        plots_dir: Union[str, Path],
    ):
        """
        Create a histogram of importance scores for a specific layer/metric and
        annotate the top-5 most important neurons.

        Args:
            layer_name: Layer name as used in importance_scores.
            metric: Metric name within importance_scores[layer_name].
            importance_scores: Nested mapping {layer_name: {metric: scores_tensor}}.
            plots_dir: Directory to save the figure.
        """

        if layer_name not in importance_scores or metric not in importance_scores[layer_name]:
            logger.warning(f"plot_layer_importance_histogram: missing scores for {layer_name}/{metric}")
            return

        raw_tensor = importance_scores[layer_name][metric]
        if not torch.is_tensor(raw_tensor) or raw_tensor.numel() == 0:
            logger.warning(f"plot_layer_importance_histogram: empty or non-tensor scores for {layer_name}/{metric}")
            return

        viz = UnifiedVisualizer()
        save_path = viz.plot_importance_histogram(
            scores=raw_tensor,
            layer_name=layer_name,
            metric_name=metric,
            plots_dir=plots_dir,
            top_k=5,
        )
        logger.info(f"[Saved] Histogram with top-5 annotations for {layer_name}/{metric}: {save_path}")

    def plot_neuron_output_weights_histogram(
        self,
        layer_name: str,
        neuron_index: int,
        plots_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Create a histogram of the outgoing weights of a specific neuron and
        highlight the top-5 largest-magnitude outgoing weights.

        Args:
            layer_name: Name of the layer (for labeling and lookup).
            neuron_index: Index of the neuron within the layer.
            plots_dir: Directory to save the figure.
        """

        # Look up the layer module and its weight tensor
        layer_module = dict(self.wrapped_model._model.named_modules()).get(layer_name)
        if layer_module is None:
            logger.warning(f"plot_neuron_output_weights_histogram: layer '{layer_name}' not found")
            return {}

        weight_tensor = self._get_layer_weights(layer_module)
        if weight_tensor is None:
            logger.warning(f"plot_neuron_output_weights_histogram: no weight tensor for layer '{layer_name}'")
            return {}

        W = weight_tensor.detach().cpu().to(torch.float32)

        if neuron_index < 0 or neuron_index >= W.shape[1]:
            logger.warning(
                f"plot_neuron_output_weights_histogram: neuron_index {neuron_index} "
                f"out of range for layer '{layer_name}' with width {W.shape[1]}"
            )
            return {}

        outgoing = W[:, neuron_index]
        magnitudes = outgoing.abs()
        k = min(5, magnitudes.numel())
        top_idxs, _ = torch.topk(magnitudes, k=k)
        top_vals = outgoing[top_idxs]

        viz = UnifiedVisualizer()
        save_path = viz.plot_neuron_outgoing_weights(
            weights=W,
            layer_name=layer_name,
            neuron_index=neuron_index,
            plots_dir=plots_dir,
            top_k=5,
        )

        logger.info(f"[Saved] Outgoing weights histogram for {layer_name} neuron {neuron_index}: {save_path}")

        return {
            "layer": layer_name,
            "neuron_index": neuron_index,
            "top5_output_indices": top_idxs.tolist(),
            "top5_values": [outgoing[i].item() for i in top_idxs],
            "plot_path": str(save_path),
        }
