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
from alignment.core.streaming import StreamingCovariance

import matplotlib.pyplot as plt
import numpy as np

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

    @staticmethod
    def _normalize_activation(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert activations with arbitrary shape to [batch, features].
        Handles variable sequence lengths by averaging across the sequence axis
        and flattens higher dimensional tensors.
        """
        if tensor is None:
            return None

        tensor = tensor.detach()

        if tensor.ndim == 3:
            tensor = tensor.mean(dim=1)
        elif tensor.ndim > 3:
            tensor = tensor.view(tensor.shape[0], -1)

        return tensor


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

        for text in calibration_texts[:num_samples]:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            outputs, activations = self.wrapped_model.forward_with_activations(inputs)

            # Process activations
            for key, value in activations.items():
                normalized = self._normalize_activation(value)
                if normalized is None:
                    continue
                
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

        # self.plot_layer_importance_histogram(
        #     layer_name="model.layers.1.mlp.up_proj",
        #     importance_scores=scores,
        #     plots_dir=self.config.plots_dir
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

            for sparsity in sparsity_levels:
                masks = self.apply_pruning(sparsity=sparsity, mode=mode, metric=metric)

                # Optional: Minimal Repair
                # if self.config.do_minimal_repair:
                #     self.apply_minimal_repair()

                # Evaluate pruned model
                if self.config.do_perplexity_computation:
                    pruned_ppl = self.evaluate_perplexity(
                        dataset=self.config.evaluation_dataset, num_samples=self.config.evaluation_num_samples
                    )

                    results["pruning_results"][f"sparsity_{sparsity}"] = {
                        "perplexity": pruned_ppl,
                        "sparsity": sparsity,
                        "num_pruned_layers": len(masks),
                    }

        return results
    

    def plot_layer_importance_histogram(self, layer_name, importance_scores, plots_dir):
        """
        Creates a histogram of importance scores for a specific layer.

        Parameters:
            layer_name (str): Name of the layer.
            importance_scores (array-like): List or numpy array of importance values.
            plots_dir (str or Path): Directory where the plot will be saved.
        """

        # Convert to numpy (if needed)
        scores = np.array(importance_scores)

        # Make sure directory exists
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create histogram
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=50, edgecolor="black")
        plt.xlabel("Importance Score")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Importance Scores — {layer_name}")
        plt.tight_layout()

        # Save plot
        save_path = plots_dir / f"{layer_name}_importance_histogram.png"
        plt.savefig(save_path)
        plt.close()

        print(f"[Saved] Histogram for {layer_name}: {save_path}")
