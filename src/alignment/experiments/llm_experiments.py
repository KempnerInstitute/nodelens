import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from alignment.analysis.visualization import UnifiedVisualizer
from alignment.core.streaming import StreamingCovariance
from alignment.experiments.base import BaseExperiment
from alignment.metrics import get_metric
from alignment.models.transformers import TransformerWrapperEnhanced as TransformerWrapper
from alignment.pruning import AlignmentPruning, PruningConfig
from alignment.pruning.pipeline import PruningPipelineOptions
from alignment.pruning.strategies.llm_baselines import SparseGPTPruning, WandaPruning
from alignment.services import MaskOperations

logger = logging.getLogger(__name__)


class LLMAlignmentExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.importance_scores: Dict[str, Dict[str, torch.Tensor]] = {}
        # Cache for expensive perplexity tokenization (e.g., full WikiText-2 test set).
        # Keyed by (dataset, subset, split, seqlen, add_bos_token flag).
        self._ppl_token_cache: Dict[Tuple[str, str, str, int, bool], torch.Tensor] = {}

    def setup(self):
        """Setup LLM alignment experiment components."""
        logger.info("Setting up LLM alignment experiment...")

        # If using HuggingFace backend, (re)wrap the HF model and load tokenizer.
        # Prefer reusing an already-initialized registry model (hf_causal_lm) to
        # avoid double-loading large checkpoints.
        if self.config.model_config.get("model_backend") == "hf":
            if getattr(self, "model", None) is not None and self.config.model_name.lower() == "hf_causal_lm":
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
                # Clear any existing hooks (they were registered with unexpanded patterns)
                if hasattr(self.wrapped_model, "_clear_hooks"):
                    self.wrapped_model._clear_hooks()

                # Directly set the internal storage for tracked layers
                if hasattr(self.wrapped_model, "_tracked_layers"):
                    self.wrapped_model._tracked_layers = expanded
                else:
                    # fallback if internal attribute differs
                    setattr(self.wrapped_model, "_tracked_layers", expanded)

                # Re-register hooks with the expanded (actual) layer names
                if hasattr(self.wrapped_model, "_register_hooks"):
                    self.wrapped_model._register_hooks()
                    logger.info(f"Re-registered hooks for {len(expanded)} expanded layers")

                logger.info(f"Tracked layers expanded to {len(expanded)} layers")

        # Ensure we have a *text* dataset for importance computation in LLM experiments.
        # BaseExperiment may skip dataset initialization for LLM experiment types, but even when it
        # does initialize a dataset (e.g., `c4` via registry), it may be a streaming dataset without
        # a materialized `.texts` list. For SCAR calibration we require raw strings, so we rebuild
        # the dataset when `.texts` is missing or None.
        needs_text_dataset = (
            getattr(self, "dataset", None) is None or not hasattr(self.dataset, "texts") or getattr(self.dataset, "texts", None) is None
        )
        if needs_text_dataset:
            try:
                from alignment.dataops.datasets.text_datasets import load_text_dataset
            except ImportError as e:
                logger.error(f"Unable to import text datasets for LLMAlignmentExperiment: {e}")
                self.dataset = None
            else:
                # Use dataset_name if provided, otherwise fall back to evaluation_dataset.
                dataset_name = getattr(self.config, "dataset_name", None) or getattr(self.config, "evaluation_dataset", "wikitext")
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

                    # Optional: deterministically shuffle the calibration text *pool* so that
                    # different seeds correspond to different calibration subsets (when the
                    # pool size exceeds the sample budget used by SCAR / robustness analyses).
                    #
                    # Enable by setting:
                    #   llm.shuffle_calibration_texts=true
                    #   llm.calibration_seed=<int>   (defaults to ExperimentConfig.seed)
                    try:
                        llm_cfg = getattr(self.config, "llm", {}) or {}
                        if isinstance(llm_cfg, dict) and bool(llm_cfg.get("shuffle_calibration_texts", False)):
                            import numpy as np

                            seed = llm_cfg.get("calibration_seed", getattr(self.config, "seed", 0))
                            try:
                                seed_i = int(seed)
                            except Exception:
                                seed_i = int(getattr(self.config, "seed", 0))

                            if hasattr(self.dataset, "texts") and isinstance(getattr(self.dataset, "texts", None), list):
                                rng = np.random.default_rng(seed_i)
                                rng.shuffle(self.dataset.texts)
                                logger.info(f"Shuffled calibration texts: seed={seed_i}, pool={len(self.dataset.texts)}")
                    except Exception as e:
                        logger.warning(f"Could not shuffle calibration texts (continuing without shuffle): {e}")
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

        llm_cfg = getattr(self.config, "llm", {}) or {}
        protocol = str(llm_cfg.get("perplexity_protocol", "legacy")).lower()

        logger.info(f"Evaluating perplexity on {dataset} ({split}) [protocol={protocol}]...")

        self.model.eval()
        device = torch.device(self.config.device)
        model_dtype = getattr(torch, self.config.model_config.get("torch_dtype", "float32"))

        # ------------------------------------------------------------------
        # OATS/SparseGPT-style WikiText-2 perplexity:
        # - concatenate full test set
        # - evaluate in contiguous blocks (default: 2048 tokens)
        #
        # This matches the common protocol used in pruning papers (and OATS Table 19),
        # and avoids padding artifacts from per-line evaluation.
        # ------------------------------------------------------------------
        if protocol in {"oats", "sparsegpt", "block"} and str(dataset).lower() in {"wikitext", "wikitext2", "wikitext-2"}:
            try:
                from datasets import load_dataset
            except Exception as e:
                logger.error(f"datasets library not available; cannot run OATS-style perplexity: {e}")
                return float("inf")

            subset = str(llm_cfg.get("wikitext_subset", "wikitext-2-raw-v1"))
            seqlen = int(llm_cfg.get("perplexity_seq_len", 2048))
            # HuggingFace tokenizers may or may not add a BOS token by default; we store the flag for caching.
            add_bos = bool(getattr(self.tokenizer, "add_bos_token", False))

            cache_key = (str(dataset).lower(), subset, str(split), seqlen, add_bos)
            input_ids = self._ppl_token_cache.get(cache_key)
            if input_ids is None:
                logger.info(f"Tokenizing WikiText for OATS-style PPL: subset={subset}, split={split}, seqlen={seqlen}")
                ds = load_dataset("wikitext", subset, split=split)
                texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
                joined = "\n\n".join(texts)
                enc = self.tokenizer(joined, return_tensors="pt")
                input_ids = enc["input_ids"].to(dtype=torch.long, device="cpu")
                self._ppl_token_cache[cache_key] = input_ids

            nlls: List[torch.Tensor] = []
            total_tokens = 0

            with torch.no_grad():
                # Iterate blocks without overlap (standard blockwise perplexity protocol).
                # If the last block is too short to have any targets, skip it.
                for bi, start in enumerate(range(0, int(input_ids.size(1)), seqlen)):
                    end = min(start + seqlen, int(input_ids.size(1)))
                    if end - start < 2:
                        continue
                    block = input_ids[:, start:end].to(device=device, dtype=torch.long)
                    labels = block.clone()
                    # Ensure token counting matches HF causal LM loss normalization (shifted by 1).
                    labels[:, 0] = -100
                    num_valid_tokens = int((labels != -100).sum().item())
                    if num_valid_tokens <= 0:
                        continue

                    with autocast(device_type=self.config.device, dtype=model_dtype):
                        outputs = self.model(block, labels=labels)
                        loss = outputs.loss
                    # HF causal LM loss is mean over valid tokens; weight by token count to
                    # aggregate correctly across variable-length blocks.
                    nlls.append(loss * num_valid_tokens)
                    total_tokens += num_valid_tokens

                    # Optional: allow partial evaluation for debugging
                    max_blocks = llm_cfg.get("perplexity_max_blocks")
                    if max_blocks is not None and bi + 1 >= int(max_blocks):
                        break

            if total_tokens <= 0 or not nlls:
                logger.error("No valid tokens processed for OATS-style perplexity!")
                return float("inf")

            mean_loss = torch.stack(nlls).sum() / total_tokens
            ppl = torch.exp(mean_loss)
            perplexity = float(ppl.item())
            logger.info(f"OATS-style WikiText PPL: {perplexity:.4f}")
            return perplexity

        # ------------------------------------------------------------------
        # Legacy per-sample perplexity (kept for backwards compatibility).
        # WARNING: this is sensitive to padding/truncation and is not a standard protocol for fair perplexity reporting.
        # ------------------------------------------------------------------
        from alignment.dataops.datasets.text_datasets import load_text_dataset

        dataset_obj = load_text_dataset(
            dataset,
            self.config.model_config.get("model_id"),
            split=split,
            max_samples=num_samples,
        )

        nlls = []
        total_length = 0

        with torch.no_grad():
            for i, batch in enumerate(dataset_obj):
                if i >= num_samples:
                    break

                input_ids = batch["input_ids"].unsqueeze(0).to(device, dtype=torch.long)

                labels = input_ids.clone()
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
                labels[labels == pad_token_id] = -100
                if labels[0, 0] == 128000:  # ignore BOS token if needed
                    labels[0, 0] = -100

                try:
                    with autocast(device_type=self.config.device, dtype=model_dtype):
                        outputs = self.model(input_ids, labels=labels)
                        loss = outputs.loss

                    num_valid_tokens = (labels != -100).sum().item()
                    if num_valid_tokens > 0:
                        # HF causal LM loss is mean over valid tokens; weight by token count.
                        nlls.append(loss * num_valid_tokens)
                        total_length += num_valid_tokens
                    else:
                        logger.warning(f"Sample {i}: No valid tokens!")
                except Exception as e:
                    logger.warning(f"Error on sample {i}: {e}")
                    continue

        if total_length == 0:
            logger.error("No valid tokens processed!")
            return float("inf")

        mean_loss = torch.stack(nlls).sum() / total_length
        ppl = torch.exp(mean_loss)
        perplexity = ppl.item()
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    # =========================================================================
    # NVIDIA MINITRON-COMPATIBLE FEW-SHOT SETTINGS
    # Reference: https://arxiv.org/abs/2408.11796
    # =========================================================================
    # These are the official few-shot settings used by NVIDIA Minitron:
    NVIDIA_FEWSHOT_SETTINGS = {
        "accuracy_mmlu": 5,  # MMLU: 5-shot
        "accuracy_hellaswag": 10,  # HellaSwag: 10-shot
        "accuracy_arc_challenge": 25,  # ARC-Challenge: 25-shot
        "accuracy_arc_easy": 25,  # ARC-Easy: 25-shot (same as Challenge)
        "accuracy_winogrande": 5,  # WinoGrande: 5-shot
        "accuracy_gsm8k": 5,  # GSM8k: 5-shot with chain-of-thought
        "accuracy_truthfulqa": 0,  # TruthfulQA: 0-shot
        "accuracy_mbpp": 0,  # MBPP: 0-shot
        "accuracy_humaneval": 0,  # HumanEval: 0-shot
        "accuracy_piqa": 0,  # PIQA: 0-shot
        "accuracy_boolq": 0,  # BoolQ: 0-shot
    }

    def evaluate_multiple_metrics(
        self,
        metrics: List[str] = None,
        num_samples: int = 50,
        fewshot_settings: Dict[str, int] = None,
        use_nvidia_settings: bool = False,
        use_chain_of_thought: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on multiple metrics with configurable few-shot settings.

        Supported metrics:
        - perplexity: Language modeling perplexity on WikiText (lower is better)
        - bits_per_byte: Bits per byte on WikiText (lower is better)
        - accuracy_hellaswag: HellaSwag commonsense (higher is better)
        - accuracy_arc_easy: ARC-Easy science (higher is better)
        - accuracy_arc_challenge: ARC-Challenge science (higher is better)
        - accuracy_piqa: PIQA physical intuition (higher is better)
        - accuracy_boolq: BoolQ boolean questions (higher is better)
        - accuracy_winogrande: WinoGrande commonsense (higher is better)
        - accuracy_truthfulqa: TruthfulQA truthfulness (higher is better)
        - accuracy_mmlu: MMLU across 57 subjects (higher is better)
        - accuracy_gsm8k: GSM8k math problems (higher is better)
        - accuracy_mbpp: MBPP code generation (higher is better)
        - accuracy_humaneval: HumanEval code generation (higher is better)

        NVIDIA Minitron benchmarks (https://arxiv.org/abs/2408.11796):
        - MMLU (5-shot), HellaSwag (10-shot), ARC-Challenge (25-shot),
        - Winogrande (5-shot), GSM8k (5-shot+CoT), TruthfulQA (0-shot),
        - MBPP (0-shot), HumanEval (0-shot)

        Args:
            metrics: List of metrics to evaluate. If None, uses config.
            num_samples: Number of samples for evaluation
            fewshot_settings: Dict mapping metric name to num_fewshot. If None, uses 0-shot.
            use_nvidia_settings: If True, use NVIDIA Minitron official few-shot settings
            use_chain_of_thought: If True, use chain-of-thought prompting for GSM8k

        Returns:
            Dict mapping metric name to value
        """
        if metrics is None:
            # Check both self.config and self.config.llm for evaluation_metrics
            llm_cfg = getattr(self.config, "llm", {}) or {}
            metrics = llm_cfg.get("evaluation_metrics") or getattr(self.config, "evaluation_metrics", ["perplexity"])

        # Get few-shot settings from config if not provided
        if fewshot_settings is None:
            llm_cfg = getattr(self.config, "llm", {}) or {}
            fewshot_settings = llm_cfg.get("fewshot_settings", {})

        # Check for NVIDIA mode from config
        llm_cfg = getattr(self.config, "llm", {}) or {}
        if llm_cfg.get("use_nvidia_fewshot", False):
            use_nvidia_settings = True
        if llm_cfg.get("use_chain_of_thought", False):
            use_chain_of_thought = True

        # Apply NVIDIA settings if requested
        if use_nvidia_settings:
            fewshot_settings = {**self.NVIDIA_FEWSHOT_SETTINGS, **fewshot_settings}
            use_chain_of_thought = True  # GSM8k uses CoT in Minitron
            logger.info("Using NVIDIA Minitron few-shot settings")

        results: Dict[str, Any] = {}

        # Avoid recomputing perplexity multiple times (loss/bpb derive from it).
        need_ppl = any(m in metrics for m in ["perplexity", "loss", "bits_per_byte", "normalized_perplexity"])
        ppl_cached: Optional[float] = None
        if need_ppl:
            try:
                ppl_cached = self.evaluate_perplexity(
                    dataset=getattr(self.config, "evaluation_dataset", "wikitext"),
                    num_samples=num_samples,
                )
            except Exception as e:
                logger.error(f"Failed to evaluate perplexity (shared): {e}")
                ppl_cached = None

        for metric in metrics:
            num_fewshot = fewshot_settings.get(metric, 0)
            try:
                if metric == "perplexity":
                    results["perplexity"] = ppl_cached
                elif metric == "loss":
                    results["loss"] = None if ppl_cached is None else float(np.log(ppl_cached))
                elif metric == "bits_per_byte":
                    # Bits per byte = log2(perplexity) / avg_chars_per_token
                    # Approximate: assume ~4 characters per token on average
                    results["bits_per_byte"] = None if ppl_cached is None else float(np.log2(ppl_cached) / 4.0)
                elif metric == "accuracy_hellaswag":
                    results["accuracy_hellaswag"] = self._evaluate_hellaswag(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_arc_easy":
                    results["accuracy_arc_easy"] = self._evaluate_arc_easy(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_arc_challenge":
                    results["accuracy_arc_challenge"] = self._evaluate_arc_challenge(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_openbookqa":
                    results["accuracy_openbookqa"] = self._evaluate_openbookqa(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_piqa":
                    results["accuracy_piqa"] = self._evaluate_piqa(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_boolq":
                    results["accuracy_boolq"] = self._evaluate_boolq(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_winogrande":
                    results["accuracy_winogrande"] = self._evaluate_winogrande(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_truthfulqa":
                    results["accuracy_truthfulqa"] = self._evaluate_truthfulqa(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_mmlu":
                    results["accuracy_mmlu"] = self._evaluate_mmlu(num_samples=num_samples, num_fewshot=num_fewshot)
                elif metric == "accuracy_gsm8k":
                    results["accuracy_gsm8k"] = self._evaluate_gsm8k(
                        num_samples=num_samples, num_fewshot=num_fewshot, use_chain_of_thought=use_chain_of_thought
                    )
                elif metric == "accuracy_mbpp":
                    results["accuracy_mbpp"] = self._evaluate_mbpp(num_samples=num_samples)
                elif metric == "accuracy_humaneval":
                    results["accuracy_humaneval"] = self._evaluate_humaneval(num_samples=num_samples)
                elif metric == "normalized_perplexity":
                    # Normalized to 0-100 scale (100 = best = PPL of 1)
                    results["normalized_perplexity"] = None if ppl_cached is None else float(100 * np.exp(-0.01 * (ppl_cached - 1)))
                else:
                    logger.warning(f"Unknown evaluation metric: {metric}")
            except Exception as e:
                logger.error(f"Failed to evaluate metric '{metric}': {e}")
                results[metric] = None

        return results

    def _score_continuations_conditional_logprob(
        self,
        prompt: str,
        continuations: List[str],
        *,
        max_length: int = 2048,
    ) -> List[float]:
        """
        Score each continuation by its conditional log-probability given the prompt.

        Implementation detail:
        We compute the model loss on *only the continuation tokens* (masking prompt tokens
        with -100) and return the mean log-prob per continuation token (higher is better).
        """
        # Defensive: handle empty candidate lists
        if not continuations:
            return []

        device = torch.device(self.config.device)
        model = self.model
        tok = self.tokenizer

        # Encode prompt once (no special tokens), then add BOS if the tokenizer defines one.
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        bos_id = getattr(tok, "bos_token_id", None)
        prefix_ids = ([bos_id] if bos_id is not None else []) + prompt_ids
        prefix_len_full = len(prefix_ids)

        scores: List[float] = []
        model.eval()
        with torch.no_grad():
            for cont in continuations:
                cont_ids = tok(cont, add_special_tokens=False).input_ids
                input_ids = prefix_ids + cont_ids

                # Truncate from the left if needed (keep most recent context).
                prefix_len = prefix_len_full
                if len(input_ids) > max_length:
                    drop = len(input_ids) - max_length
                    input_ids = input_ids[drop:]
                    prefix_len = max(0, prefix_len_full - drop)
                    # If we truncated away the entire prompt context, the score becomes meaningless.
                    if prefix_len <= 0:
                        scores.append(float("-inf"))
                        continue

                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
                attn = torch.ones_like(input_ids_t, dtype=torch.long, device=device)

                labels = input_ids_t.clone()
                labels[:, :prefix_len] = -100  # only score continuation tokens

                out = model(input_ids=input_ids_t, attention_mask=attn, labels=labels)
                loss = getattr(out, "loss", None)
                if loss is None:
                    scores.append(float("-inf"))
                else:
                    scores.append(float(-loss.item()))

        return scores

    def _evaluate_mmlu(self, num_samples: int = 100, subjects: List[str] = None, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on MMLU (Massive Multitask Language Understanding).
        Tests knowledge across 57 subjects.
        Returns accuracy (higher is better).

        Used in NVIDIA Minitron (https://arxiv.org/abs/2407.14679) with 5-shot.

        Args:
            num_samples: Number of samples per subject (or total if subjects is None)
            subjects: List of subjects to evaluate. If None, samples from all subjects.
            num_fewshot: Number of few-shot examples (NVIDIA Minitron uses 5)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate MMLU")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on MMLU (~{num_samples} samples total)...")

        # Default subjects for quick evaluation (covers different domains)
        if subjects is None:
            subjects = [
                "abstract_algebra",
                "anatomy",
                "astronomy",
                "business_ethics",
                "clinical_knowledge",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "econometrics",
                "electrical_engineering",
                "elementary_mathematics",
                "formal_logic",
                "global_facts",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_european_history",
                "high_school_geography",
                "high_school_government_and_politics",
                "high_school_macroeconomics",
                "high_school_mathematics",
                "high_school_microeconomics",
                "high_school_physics",
                "high_school_psychology",
                "high_school_statistics",
                "high_school_us_history",
                "high_school_world_history",
                "human_aging",
                "human_sexuality",
                "international_law",
                "jurisprudence",
                "logical_fallacies",
                "machine_learning",
                "management",
                "marketing",
                "medical_genetics",
                "miscellaneous",
                "moral_disputes",
                "moral_scenarios",
                "nutrition",
                "philosophy",
                "prehistory",
                "professional_accounting",
                "professional_law",
                "professional_medicine",
                "professional_psychology",
                "public_relations",
                "security_studies",
                "sociology",
                "us_foreign_policy",
                "virology",
                "world_religions",
            ]

        self.model.eval()
        correct = 0
        total = 0
        torch.device(self.config.device)
        choice_labels = ["A", "B", "C", "D"]

        # Deterministic sampling (avoid label/order bias from always taking the first example).
        # Interpret num_samples as a TOTAL budget across subjects.
        import zlib

        subjects = list(subjects)
        seed = int(getattr(self.config, "seed", 0) or 0)

        # Shuffle subject order to avoid any systematic ordering artifacts.
        rng_subj = np.random.default_rng(seed)
        rng_subj.shuffle(subjects)

        # Allocate an exact per-subject quota summing to num_samples.
        base = int(num_samples) // max(1, len(subjects))
        rem = int(num_samples) % max(1, len(subjects))
        quotas = [base + (1 if i < rem else 0) for i in range(len(subjects))]

        with torch.no_grad():
            for subject, quota in zip(subjects, quotas):
                if quota <= 0:
                    continue
                try:
                    dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
                    # Load dev split for few-shot examples
                    if num_fewshot > 0:
                        dev_dataset = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)
                except Exception as e:
                    logger.warning(f"Failed to load MMLU subject '{subject}': {e}")
                    continue

                # Build few-shot prompt for this subject
                fewshot_prompt = ""
                if num_fewshot > 0:
                    # Sample few-shot examples from dev split.
                    try:
                        dev_n = len(dev_dataset)
                        dev_seed = seed + int(zlib.adler32(f"{subject}:dev".encode()))
                        rng_dev = np.random.default_rng(dev_seed)
                        dev_idxs = rng_dev.choice(dev_n, size=min(int(num_fewshot), dev_n), replace=False).tolist()
                    except Exception:
                        dev_idxs = list(range(int(num_fewshot)))

                    for ex_idx in dev_idxs:
                        ex = dev_dataset[int(ex_idx)]
                        q = ex["question"]
                        choices = ex["choices"]
                        answer_idx = ex["answer"]
                        # Format: The following are multiple choice questions...
                        choices_str = "\n".join([f"{choice_labels[j]}) {c}" for j, c in enumerate(choices)])
                        correct_label = choice_labels[answer_idx]
                        fewshot_prompt += f"Question: {q}\n{choices_str}\nAnswer: {correct_label}\n\n"

                subject_correct = 0
                subject_total = 0

                # Sample test examples for this subject (without replacement).
                try:
                    n_test = len(dataset)
                    test_seed = seed + int(zlib.adler32(f"{subject}:test".encode()))
                    rng_test = np.random.default_rng(test_seed)
                    test_idxs = rng_test.choice(n_test, size=min(int(quota), n_test), replace=False).tolist()
                except Exception:
                    test_idxs = list(range(int(quota)))

                for ex_i, ex_idx in enumerate(test_idxs):
                    if total >= num_samples:
                        break

                    try:
                        example = dataset[int(ex_idx)]
                        question = example["question"]
                        choices = example["choices"]
                        answer_idx = example["answer"]  # 0-indexed

                        # Score each answer label by conditional log-probability (standard MCQ protocol):
                        # prompt includes all choices; continuation is just the option label.
                        choices_str = "\n".join([f"{choice_labels[k]}) {c}" for k, c in enumerate(choices)])
                        prompt = (
                            f"{fewshot_prompt}Question: {question}\n{choices_str}\nAnswer:"
                            if num_fewshot > 0
                            else (f"Question: {question}\n{choices_str}\nAnswer:")
                        )
                        continuations = [f" {choice_labels[j]}" for j in range(len(choices))]
                        scores = self._score_continuations_conditional_logprob(prompt, continuations, max_length=2048)

                        predicted = np.argmax(scores)
                        if predicted == answer_idx:
                            correct += 1
                            subject_correct += 1
                        total += 1
                        subject_total += 1

                    except Exception as e:
                        logger.warning(f"Error on MMLU {subject} sample {ex_i}: {e}")
                        continue

                if subject_total > 0:
                    logger.debug(f"MMLU {subject}: {100*subject_correct/subject_total:.1f}% ({subject_correct}/{subject_total})")

                if total >= num_samples:
                    break

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"MMLU accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_hellaswag(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on HellaSwag (commonsense reasoning).
        Returns accuracy (higher is better).

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples (NVIDIA Minitron uses 10)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate HellaSwag")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on HellaSwag ({num_samples} samples)...")

        try:
            dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load HellaSwag dataset: {e}")
            return 0.0

        # Build few-shot examples from the beginning of the dataset
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(dataset):
                if i >= num_fewshot:
                    break
                ctx = ex["ctx"]
                endings = ex["endings"]
                label = int(ex["label"])
                correct_ending = endings[label]
                fewshot_prompt += f"Context: {ctx}\nEnding: {correct_ending}\n\n"
            # Skip the few-shot examples when evaluating
            eval_start_idx = num_fewshot
        else:
            eval_start_idx = 0

        self.model.eval()
        correct = 0
        total = 0
        torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i < eval_start_idx:
                    continue
                if total >= num_samples:
                    break

                try:
                    ctx = example["ctx"]
                    endings = example["endings"]
                    label = int(example["label"])

                    # Score endings by conditional log-probability (context is prompt, ending is continuation).
                    prompt = f"{fewshot_prompt}Context: {ctx}\nEnding:" if num_fewshot > 0 else f"Context: {ctx}\nEnding:"
                    continuations = [f" {ending}" for ending in endings]
                    scores = self._score_continuations_conditional_logprob(prompt, continuations, max_length=2048)

                    predicted = np.argmax(scores)
                    if predicted == label:
                        correct += 1
                    total += 1

                except Exception as e:
                    logger.warning(f"Error on HellaSwag sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"HellaSwag accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_arc_easy(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on ARC-Easy (science questions).
        Returns accuracy (higher is better).

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples (NVIDIA Minitron uses 25)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate ARC")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on ARC-Easy ({num_samples} samples)...")

        try:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", trust_remote_code=True)
            # Load train split for few-shot examples
            if num_fewshot > 0:
                train_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load ARC-Easy dataset: {e}")
            return 0.0

        # Build few-shot examples
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(train_dataset):
                if i >= num_fewshot:
                    break
                q = ex["question"]
                choices = ex["choices"]
                answer_key = ex["answerKey"]
                choice_texts = choices["text"]
                choice_labels = choices["label"]
                answer_idx = choice_labels.index(answer_key)
                correct_answer = choice_texts[answer_idx]
                fewshot_prompt += f"Question: {q}\nAnswer: {correct_answer}\n\n"

        self.model.eval()
        correct = 0
        total = 0
        torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break

                try:
                    question = example["question"]
                    choices = example["choices"]
                    answer_key = example["answerKey"]

                    choice_texts = choices["text"]
                    choice_labels = choices["label"]
                    answer_idx = choice_labels.index(answer_key)

                    # Score candidate answers by conditional log-probability (prompt excludes answer tokens).
                    prompt = f"{fewshot_prompt}Question: {question}\nAnswer:" if num_fewshot > 0 else f"Question: {question}\nAnswer:"
                    continuations = [f" {ct}" for ct in choice_texts]
                    scores = self._score_continuations_conditional_logprob(prompt, continuations, max_length=2048)

                    predicted = np.argmax(scores)
                    if predicted == answer_idx:
                        correct += 1
                    total += 1

                except Exception as e:
                    logger.warning(f"Error on ARC-Easy sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"ARC-Easy accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_piqa(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on PIQA (physical intuition).
        Returns accuracy (higher is better).

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate PIQA")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on PIQA ({num_samples} samples)...")

        try:
            dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
            if num_fewshot > 0:
                train_dataset = load_dataset("piqa", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load PIQA dataset: {e}")
            return 0.0

        # Build few-shot examples
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(train_dataset):
                if i >= num_fewshot:
                    break
                goal = ex["goal"]
                sol1 = ex["sol1"]
                sol2 = ex["sol2"]
                label = ex["label"]
                correct_sol = sol1 if label == 0 else sol2
                fewshot_prompt += f"Goal: {goal}\nSolution: {correct_sol}\n\n"

        self.model.eval()
        correct = 0
        total = 0
        torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break

                try:
                    goal = example["goal"]
                    sol1 = example["sol1"]
                    sol2 = example["sol2"]
                    label = example["label"]  # 0 or 1

                    # Score solutions by conditional log-probability (goal is prompt, solution is continuation).
                    prompt = f"{fewshot_prompt}Goal: {goal}\nSolution:" if num_fewshot > 0 else f"Goal: {goal}\nSolution:"
                    continuations = [f" {sol1}", f" {sol2}"]
                    scores = self._score_continuations_conditional_logprob(prompt, continuations, max_length=2048)

                    predicted = np.argmax(scores)
                    if predicted == label:
                        correct += 1
                    total += 1

                except Exception as e:
                    logger.warning(f"Error on PIQA sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"PIQA accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_boolq(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on BoolQ (boolean questions).
        Returns accuracy (higher is better).

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate BoolQ")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on BoolQ ({num_samples} samples)...")

        try:
            dataset = load_dataset("google/boolq", split="validation", trust_remote_code=True)
            if num_fewshot > 0:
                train_dataset = load_dataset("google/boolq", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load BoolQ dataset: {e}")
            return 0.0

        # Build few-shot examples
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(train_dataset):
                if i >= num_fewshot:
                    break
                q = ex["question"]
                p = ex["passage"][:200]  # Truncate passage for few-shot
                a = "Yes" if ex["answer"] else "No"
                fewshot_prompt += f"Passage: {p}...\nQuestion: {q}\nAnswer: {a}\n\n"

        self.model.eval()
        correct = 0
        total = 0
        torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break

                try:
                    question = example["question"]
                    passage = example["passage"]
                    answer = example["answer"]  # True or False

                    # Score "Yes" vs "No" by conditional log-probability of the answer token(s).
                    prompt = (
                        f"{fewshot_prompt}Passage: {passage}\nQuestion: {question}\nAnswer:"
                        if num_fewshot > 0
                        else f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                    )
                    scores = self._score_continuations_conditional_logprob(prompt, [" Yes", " No"], max_length=2048)

                    # 0 = Yes (True), 1 = No (False)
                    predicted = np.argmax(scores) == 0  # True if "Yes" has higher score
                    if predicted == answer:
                        correct += 1
                    total += 1

                except Exception as e:
                    logger.warning(f"Error on BoolQ sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"BoolQ accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_winogrande(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on WinoGrande (commonsense reasoning with Winograd schemas).
        Returns accuracy (higher is better).

        WinoGrande is a large-scale dataset of Winograd Schema Challenge problems.
        Each example has a sentence with a blank and two options to fill it.

        Used in NVIDIA Minitron (https://arxiv.org/abs/2407.14679) with 5-shot.

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples (NVIDIA Minitron uses 5)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate WinoGrande")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on WinoGrande ({num_samples} samples)...")

        try:
            dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
            if num_fewshot > 0:
                train_dataset = load_dataset("winogrande", "winogrande_xl", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load WinoGrande dataset: {e}")
            return 0.0

        # Build few-shot examples
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(train_dataset):
                if i >= num_fewshot:
                    break
                sentence = ex["sentence"]
                option1 = ex["option1"]
                option2 = ex["option2"]
                answer = int(ex["answer"]) - 1
                correct_option = option1 if answer == 0 else option2
                completed = sentence.replace("_", correct_option)
                fewshot_prompt += f"Sentence: {completed}\n\n"

        self.model.eval()
        correct = 0
        total = 0
        torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break

                try:
                    sentence = example["sentence"]
                    option1 = example["option1"]
                    option2 = example["option2"]
                    answer = int(example["answer"]) - 1  # Convert 1/2 to 0/1

                    # Score each option by conditional log-probability of the completion.
                    # We split at the blank so only the option + suffix is scored (prefix is prompt).
                    if "_" in sentence:
                        prefix, suffix = sentence.split("_", 1)
                    else:
                        prefix, suffix = sentence, ""
                    prompt = f"{fewshot_prompt}Sentence: {prefix}" if num_fewshot > 0 else f"Sentence: {prefix}"
                    continuations = [f"{option1}{suffix}", f"{option2}{suffix}"]
                    scores = self._score_continuations_conditional_logprob(prompt, continuations, max_length=2048)

                    predicted = np.argmax(scores)
                    if predicted == answer:
                        correct += 1
                    total += 1

                    if (i + 1) % 50 == 0:
                        logger.info(f"WinoGrande: {total}/{num_samples}, accuracy so far: {100*correct/total:.1f}%")

                except Exception as e:
                    logger.warning(f"Error on WinoGrande sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"WinoGrande accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_arc_challenge(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on ARC-Challenge (harder science questions).
        Returns accuracy (higher is better).

        ARC-Challenge is the harder subset of the AI2 Reasoning Challenge,
        containing questions that require more complex reasoning.
        Used in NVIDIA Minitron (https://arxiv.org/abs/2407.14679) with 25-shot.

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples (NVIDIA Minitron uses 25)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate ARC-Challenge")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on ARC-Challenge ({num_samples} samples)...")

        try:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
            if num_fewshot > 0:
                train_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load ARC-Challenge dataset: {e}")
            return 0.0

        # Build few-shot examples
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(train_dataset):
                if i >= num_fewshot:
                    break
                q = ex["question"]
                choices = ex["choices"]
                answer_key = ex["answerKey"]
                choice_texts = choices["text"]
                choice_labels = choices["label"]
                answer_idx = choice_labels.index(answer_key)
                correct_answer = choice_texts[answer_idx]
                fewshot_prompt += f"Question: {q}\nAnswer: {correct_answer}\n\n"

        self.model.eval()
        correct = 0
        total = 0
        torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break

                try:
                    question = example["question"]
                    choices = example["choices"]
                    answer_key = example["answerKey"]

                    choice_texts = choices["text"]
                    choice_labels = choices["label"]
                    answer_idx = choice_labels.index(answer_key)

                    # Score candidate answers by conditional log-probability (prompt excludes answer tokens).
                    prompt = f"{fewshot_prompt}Question: {question}\nAnswer:" if num_fewshot > 0 else f"Question: {question}\nAnswer:"
                    continuations = [f" {ct}" for ct in choice_texts]
                    scores = self._score_continuations_conditional_logprob(prompt, continuations, max_length=2048)

                    predicted = np.argmax(scores)
                    if predicted == answer_idx:
                        correct += 1
                    total += 1

                    if (i + 1) % 50 == 0:
                        logger.info(f"ARC-Challenge: {total}/{num_samples}, accuracy so far: {100*correct/total:.1f}%")

                except Exception as e:
                    logger.warning(f"Error on ARC-Challenge sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"ARC-Challenge accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_openbookqa(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Zero-/few-shot evaluation on OpenBookQA (4-way MCQ).

        We score options using conditional log-probability of the *option label* continuation,
        with the full question + choices included in the prompt (standard MCQ protocol).

        Returns accuracy in percent (higher is better).
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate OpenBookQA")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on OpenBookQA ({num_samples} samples)...")

        # Dataset schema varies a bit across versions; handle both common shapes.
        # HF dataset: openbookqa, config \"main\".
        try:
            dataset = load_dataset("openbookqa", "main", split="test", trust_remote_code=True)
            if num_fewshot > 0:
                train_dataset = load_dataset("openbookqa", "main", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load OpenBookQA dataset: {e}")
            return 0.0

        def _get_question(ex: Dict[str, Any]) -> str:
            if isinstance(ex.get("question_stem"), str):
                return ex["question_stem"]
            q = ex.get("question")
            if isinstance(q, dict) and isinstance(q.get("stem"), str):
                return q["stem"]
            return str(ex.get("question", ""))

        def _get_choices(ex: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            ch = ex.get("choices")
            if isinstance(ch, dict):
                texts = ch.get("text") or ch.get("texts") or []
                labels = ch.get("label") or ch.get("labels") or []
                return list(texts), list(labels)
            # Some variants store as list of dicts
            if isinstance(ch, list):
                texts = [c.get("text", "") for c in ch if isinstance(c, dict)]
                labels = [c.get("label", "") for c in ch if isinstance(c, dict)]
                return texts, labels
            return [], []

        # Build few-shot prompt in the same MCQ format.
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(train_dataset):
                if i >= num_fewshot:
                    break
                q = _get_question(ex)
                choice_texts, choice_labels = _get_choices(ex)
                answer_key = ex.get("answerKey")
                if not choice_texts or not choice_labels or answer_key not in choice_labels:
                    continue
                choices_str = "\n".join([f"{choice_labels[j]}) {choice_texts[j]}" for j in range(len(choice_texts))])
                fewshot_prompt += f"Question: {q}\n{choices_str}\nAnswer: {answer_key}\n\n"

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break
                try:
                    question = _get_question(example)
                    choice_texts, choice_labels = _get_choices(example)
                    answer_key = example.get("answerKey")
                    if not choice_texts or not choice_labels or answer_key not in choice_labels:
                        continue

                    # Prompt includes choices; continuation is the option label.
                    choices_str = "\n".join([f"{choice_labels[j]}) {choice_texts[j]}" for j in range(len(choice_texts))])
                    prompt = (
                        f"{fewshot_prompt}Question: {question}\n{choices_str}\nAnswer:"
                        if num_fewshot > 0
                        else f"Question: {question}\n{choices_str}\nAnswer:"
                    )
                    continuations = [f" {lab}" for lab in choice_labels]
                    scores = self._score_continuations_conditional_logprob(prompt, continuations, max_length=2048)

                    predicted = int(np.argmax(scores))
                    if choice_labels[predicted] == answer_key:
                        correct += 1
                    total += 1
                except Exception as e:
                    logger.warning(f"Error on OpenBookQA sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"OpenBookQA accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_truthfulqa(self, num_samples: int = 100, num_fewshot: int = 0) -> float:
        """
        Few-shot evaluation on TruthfulQA (truthfulness in answers).
        Returns accuracy (higher is better).

        TruthfulQA measures whether models generate truthful answers to questions
        that humans might answer incorrectly due to false beliefs or misconceptions.
        Used in NVIDIA Minitron (https://arxiv.org/abs/2407.14679) with 0-shot.

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples (NVIDIA Minitron uses 0)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate TruthfulQA")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        logger.info(f"Evaluating {shot_str} accuracy on TruthfulQA ({num_samples} samples)...")

        try:
            # TruthfulQA multiple choice format
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load TruthfulQA dataset: {e}")
            return 0.0

        # Build few-shot examples (though NVIDIA uses 0-shot)
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(dataset):
                if i >= num_fewshot:
                    break
                q = ex["question"]
                mc1_targets = ex["mc1_targets"]
                choices = mc1_targets["choices"]
                labels = mc1_targets["labels"]
                correct_idx = labels.index(1)
                correct_answer = choices[correct_idx]
                fewshot_prompt += f"Question: {q}\nAnswer: {correct_answer}\n\n"
            eval_start_idx = num_fewshot
        else:
            eval_start_idx = 0

        self.model.eval()
        correct = 0
        total = 0
        device = torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i < eval_start_idx:
                    continue
                if total >= num_samples:
                    break

                try:
                    question = example["question"]
                    mc1_targets = example["mc1_targets"]

                    choices = mc1_targets["choices"]
                    labels = mc1_targets["labels"]  # List of 0s and 1s (1 = correct)

                    # Find correct answer index
                    correct_idx = labels.index(1)

                    # Score each choice
                    scores = []
                    for choice in choices:
                        if num_fewshot > 0:
                            text = f"{fewshot_prompt}Question: {question}\nAnswer: {choice}"
                        else:
                            text = f"Question: {question}\nAnswer: {choice}"
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()

                        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scores.append(-loss.item())

                    predicted = np.argmax(scores)
                    if predicted == correct_idx:
                        correct += 1
                    total += 1

                    if (i + 1) % 50 == 0:
                        logger.info(f"TruthfulQA: {total}/{num_samples}, accuracy so far: {100*correct/total:.1f}%")

                except Exception as e:
                    logger.warning(f"Error on TruthfulQA sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"TruthfulQA accuracy ({shot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_gsm8k(self, num_samples: int = 100, num_fewshot: int = 0, use_chain_of_thought: bool = False) -> float:
        """
        Few-shot evaluation on GSM8k (grade school math word problems) with chain-of-thought.
        Returns accuracy (higher is better).

        GSM8k tests mathematical reasoning with grade school level word problems.
        Used in NVIDIA Minitron (https://arxiv.org/abs/2408.11796) with 5-shot + CoT.

        Args:
            num_samples: Number of samples to evaluate
            num_fewshot: Number of few-shot examples (NVIDIA Minitron uses 5)
            use_chain_of_thought: If True, include step-by-step reasoning in few-shot examples
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate GSM8k")
            return 0.0

        shot_str = f"{num_fewshot}-shot" if num_fewshot > 0 else "zero-shot"
        cot_str = " + CoT" if use_chain_of_thought else ""
        logger.info(f"Evaluating {shot_str}{cot_str} accuracy on GSM8k ({num_samples} samples)...")

        try:
            dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
            if num_fewshot > 0:
                train_dataset = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load GSM8k dataset: {e}")
            return 0.0

        def extract_answer(text: str) -> str:
            """Extract the final numerical answer from GSM8k format."""
            import re

            # GSM8k answers are in format "#### NUMBER"
            match = re.search(r"####\s*([0-9,\-\.]+)", text)
            if match:
                return match.group(1).replace(",", "")
            # Try to find last number in text
            numbers = re.findall(r"[\-]?[0-9,]+\.?[0-9]*", text)
            if numbers:
                return numbers[-1].replace(",", "")
            return ""

        # Build few-shot prompt with chain-of-thought
        fewshot_prompt = ""
        if num_fewshot > 0:
            for i, ex in enumerate(train_dataset):
                if i >= num_fewshot:
                    break
                q = ex["question"]
                a = ex["answer"]
                gold = extract_answer(a)

                if use_chain_of_thought:
                    # Include the full reasoning chain from the dataset
                    # GSM8k answers contain step-by-step reasoning before ####
                    reasoning = a.split("####")[0].strip() if "####" in a else a
                    fewshot_prompt += f"Question: {q}\n\nLet's solve this step by step:\n{reasoning}\n\nThe answer is #### {gold}\n\n"
                else:
                    fewshot_prompt += f"Question: {q}\nAnswer: {gold}\n\n"

        self.model.eval()
        correct = 0
        total = 0
        device = torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break

                try:
                    question = example["question"]
                    answer = example["answer"]
                    gold_answer = extract_answer(answer)

                    # Generate answer using the model
                    if num_fewshot > 0 and use_chain_of_thought:
                        prompt = f"{fewshot_prompt}Question: {question}\n\nLet's solve this step by step:\n"
                    elif num_fewshot > 0:
                        prompt = f"{fewshot_prompt}Question: {question}\nAnswer:"
                    else:
                        prompt = f"Question: {question}\n\nLet's solve this step by step:\n"

                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=512 if use_chain_of_thought else 256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                    pred_answer = extract_answer(generated)

                    # Compare answers (allow for minor formatting differences)
                    try:
                        if pred_answer and gold_answer:
                            if float(pred_answer) == float(gold_answer):
                                correct += 1
                    except ValueError:
                        if pred_answer == gold_answer:
                            correct += 1

                    total += 1

                    if (i + 1) % 20 == 0:
                        logger.info(f"GSM8k: {total}/{num_samples}, accuracy so far: {100*correct/total:.1f}%")

                except Exception as e:
                    logger.warning(f"Error on GSM8k sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"GSM8k accuracy ({shot_str}{cot_str}): {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_mbpp(self, num_samples: int = 100) -> float:
        """
        Zero-shot evaluation on MBPP (Mostly Basic Python Problems).
        Returns accuracy based on pass@1 (higher is better).

        MBPP tests code generation with simple Python problems.
        Used in NVIDIA Minitron (https://arxiv.org/abs/2408.11796) for LLM evaluation.

        Note: This is a simplified evaluation that checks syntax and basic execution.
        Full evaluation would require running test cases in a sandbox.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate MBPP")
            return 0.0

        logger.info(f"Evaluating zero-shot accuracy on MBPP ({num_samples} samples)...")

        try:
            dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load MBPP dataset: {e}")
            return 0.0

        self.model.eval()
        correct = 0
        total = 0
        device = torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break

                try:
                    prompt_text = example["prompt"]
                    test_list = example["test_list"]

                    # Format prompt for code generation
                    prompt = f"Write a Python function to solve the following problem:\n\n{prompt_text}\n\n```python\n"
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Generate code
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.encode("```")[0] if "```" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
                        )

                    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

                    # Extract code (up to closing ```)
                    code = generated.split("```")[0].strip()

                    # Try to execute the code with test cases
                    try:
                        # Create a namespace for execution
                        namespace = {}
                        exec(code, namespace)

                        # Run test cases
                        passed = 0
                        for test in test_list[:3]:  # Limit to first 3 tests for speed
                            try:
                                exec(test, namespace)
                                passed += 1
                            except Exception:
                                pass

                        if passed == len(test_list[:3]):
                            correct += 1

                    except Exception:
                        pass  # Code didn't compile/run

                    total += 1

                    if (i + 1) % 20 == 0:
                        logger.info(f"MBPP: {i+1}/{num_samples}, accuracy so far: {100*correct/total:.1f}%")

                except Exception as e:
                    logger.warning(f"Error on MBPP sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"MBPP accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def _evaluate_humaneval(self, num_samples: int = 100) -> float:
        """
        Zero-shot evaluation on HumanEval (code generation).
        Returns accuracy based on pass@1 (higher is better).

        HumanEval tests code generation with Python programming problems.
        Used in NVIDIA Minitron (https://arxiv.org/abs/2408.11796) for LLM evaluation.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed, cannot evaluate HumanEval")
            return 0.0

        logger.info(f"Evaluating zero-shot accuracy on HumanEval ({num_samples} samples)...")

        try:
            dataset = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load HumanEval dataset: {e}")
            return 0.0

        self.model.eval()
        correct = 0
        total = 0
        device = torch.device(self.config.device)

        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break

                try:
                    prompt = example["prompt"]
                    test = example["test"]
                    entry_point = example["entry_point"]

                    # Generate code completion
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

                    # Combine prompt and generated code
                    # Stop at first function definition that's not the target
                    lines = generated.split("\n")
                    code_lines = []
                    for line in lines:
                        if line.strip().startswith("def ") and entry_point not in line:
                            break
                        code_lines.append(line)

                    full_code = prompt + "\n".join(code_lines)

                    # Try to execute with test
                    try:
                        namespace = {}
                        exec(full_code, namespace)
                        exec(test, namespace)
                        # If we get here, tests passed
                        correct += 1
                    except Exception:
                        pass

                    total += 1

                    if (i + 1) % 20 == 0:
                        logger.info(f"HumanEval: {i+1}/{num_samples}, accuracy so far: {100*correct/total:.1f}%")

                except Exception as e:
                    logger.warning(f"Error on HumanEval sample {i}: {e}")
                    continue

        accuracy = 100 * correct / total if total > 0 else 0.0
        logger.info(f"HumanEval accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

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
        # Ensure the tokenizer can pad batched inputs (needed by several analysis utilities).
        # For causal LMs, padding with EOS is standard; fall back to BOS/UNK if EOS is unavailable.
        added_special = 0
        if getattr(tokenizer, "pad_token", None) is None or getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "bos_token", None) is not None:
                tokenizer.pad_token = tokenizer.bos_token
            elif getattr(tokenizer, "unk_token", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token
        if getattr(tokenizer, "pad_token", None) is None or getattr(tokenizer, "pad_token_id", None) is None:
            # Last resort: add a PAD token (should almost never trigger for Llama-family tokenizers).
            added_special = tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Unwrap underlying HF model if we're holding a small wrapper (e.g., HFCausalLM)
        hf_model = getattr(self.model, "model", self.model)
        if added_special > 0:
            try:
                hf_model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass

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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config.model_config.get("model_id")
        if not model_id:
            raise ValueError("LLMAlignmentExperiment requires config.model_id for HF backend")

        logger.info(f"Loading tokenizer for {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **self.config.tokenizer_kwargs)
        # Ensure the tokenizer can pad batched inputs (needed by several analysis utilities).
        # For causal LMs, padding with EOS is standard; fall back to BOS/UNK if EOS is unavailable.
        added_special = 0
        if getattr(tokenizer, "pad_token", None) is None or getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "bos_token", None) is not None:
                tokenizer.pad_token = tokenizer.bos_token
            elif getattr(tokenizer, "unk_token", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token
        if getattr(tokenizer, "pad_token", None) is None or getattr(tokenizer, "pad_token_id", None) is None:
            # Last resort: add a PAD token (should almost never trigger for Llama-family tokenizers).
            added_special = tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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
        if added_special > 0:
            try:
                hf_model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass

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
            if hasattr(self.dataset, "texts") and getattr(self.dataset, "texts", None) is not None:
                calibration_texts = list(self.dataset.texts)
            else:
                logger.warning(
                    "LLMAlignmentExperiment.dataset does not expose `.texts`; " "falling back to iterating the dataset to extract raw text."
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

        all_activations = {}  # For non-streaming metrics (like OutlierIndex which needs quantiles)
        # Note: OutlierIndex usually needs full distribution. Streaming approx is hard.
        # We'll assume we can fit sampled activations for OI, but use streaming for Covariance/RQ.

        # Pre-compute hidden dimensions for each tracked layer
        layer_dims = {}
        underlying_model = self._get_underlying_model()
        for layer_name in self.wrapped_model._tracked_layers:
            try:
                module = dict(underlying_model.named_modules()).get(layer_name)
                if module is not None and hasattr(module, "weight"):
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
                # TODO: Add an efficient pairwise-metric path (redundancy/synergy) for LLM layers.
                # Current implementation computes independent metrics only.
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
                        # Move weight to same device as activations to handle multi-GPU models
                        metric_args["weights"] = weight.to(self.config.device)

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
                        # Move weight to same device as activations to handle multi-GPU models
                        metric_args["weights"] = weight.to(self.config.device)

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

        # For FFN channel comparisons and structured pruning, down_proj should use
        # the FFN-channel-side activation score (intermediate width), not the hidden
        # output width of down_proj itself. Reuse sibling gate/up activation scores
        # when they match the down_proj input width.
        try:
            module_map = dict(self.wrapped_model._model.named_modules())
            for layer_name, layer_scores in list(self.importance_scores.items()):
                if "mlp.down_proj" not in layer_name:
                    continue

                layer_module = module_map.get(layer_name)
                if layer_module is None or not hasattr(layer_module, "weight"):
                    continue
                target_dim = int(layer_module.weight.shape[1])

                act_scores = layer_scores.get("activation_l2_norm")
                if torch.is_tensor(act_scores) and int(act_scores.numel()) == target_dim:
                    continue

                replacement = None
                replacement_src = None
                for sibling_proj in ("gate_proj", "up_proj"):
                    sibling_name = layer_name.replace("down_proj", sibling_proj)
                    sibling_scores = (self.importance_scores.get(sibling_name) or {}).get("activation_l2_norm")
                    if torch.is_tensor(sibling_scores) and int(sibling_scores.numel()) == target_dim:
                        replacement = sibling_scores.detach().clone()
                        replacement_src = sibling_name
                        break

                if replacement is not None:
                    layer_scores["activation_l2_norm"] = replacement
                    if getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}):
                        try:
                            composite_score = layer_scores.get("composite")
                            self._apply_supernode_selection(layer_scores, composite_score if torch.is_tensor(composite_score) else None)
                        except Exception:
                            pass
                    self.importance_scores[layer_name] = layer_scores
                    logger.info(
                        "Aligned activation_l2_norm for %s using FFN-channel activations from %s",
                        layer_name,
                        replacement_src,
                    )
        except Exception as align_err:
            logger.debug(f"Failed to align down_proj activation scores: {align_err}")

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
            loss_proxy_i       = 0.5 * E[(u_i * (v_i^T g_y))^2]     (joint second moment; matches the documented loss-proxy definition)

        Notes:
        - We also compute a factored approximation (0.5 * E[u_i^2] * E[(v_i^T g_y)^2]) for diagnostics.
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
                    logger.warning("SCAR metrics: dataset does not expose `.texts`; " "falling back to iterating dataset for raw text.")
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
                "SCAR metrics: no calibration texts available. " "Run importance computation first or ensure the dataset provides raw texts."
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
                "R_sum": None,  # sum over tokens of (v_i^T g_y)^2
                "T_sum": None,  # sum over tokens of |g_u_i * u_i|
                "loss_proxy_sum": None,  # sum over tokens of (u_i * (v_i^T g_y))^2
                "count": 0,  # number of tokens seen
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
                        # Accumulate in float32 for numerical stability (bfloat16 accumulation is too lossy)
                        state["u_sqr_sum"] = torch.zeros(m, device=u_flat.device, dtype=torch.float32)
                        state["R_sum"] = torch.zeros_like(state["u_sqr_sum"])
                        state["T_sum"] = torch.zeros_like(state["u_sqr_sum"])
                        state["loss_proxy_sum"] = torch.zeros_like(state["u_sqr_sum"])

                    u_flat_f = u_flat.float()
                    state["u_sqr_sum"] += (u_flat_f * u_flat_f).sum(dim=0)
                    state["count"] += u_flat.shape[0]

                    # Store u for first-order saliency computation in backward
                    mod._scar_last_u = u.detach()

                def bwd_hook(mod: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
                    state = scar_state[name]

                    # Gradient w.r.t. module input (u)
                    if not grad_input or grad_input[0] is None:
                        return

                    g_u = grad_input[0]

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

                    # Ensure shapes are consistent
                    if u_flat.shape != g_u_flat.shape:
                        logger.warning(f"SCAR metrics: shape mismatch between u ({u_flat.shape}) and g_u ({g_u_flat.shape}) for layer {name}.")
                        return

                    # NOTE: In backprop through y=W_down u, PyTorch already computes:
                    #   g_u = dL/du = W_down^T * dL/dy
                    # So s_i := (v_i^T g_y) is exactly g_u_i. No extra GEMM needed.
                    s_flat = g_u_flat.float()

                    s2 = (s_flat * s_flat).sum(dim=0)
                    state["R_sum"] += s2

                    # First-order Taylor saliency: E[ |g_u_i * u_i| ]
                    u_flat_f = u_flat.float()
                    g_u_flat_f = g_u_flat.float()
                    t_contrib = torch.abs(g_u_flat_f * u_flat_f).sum(dim=0)
                    state["T_sum"] += t_contrib

                    # Loss proxy: 0.5 * E[(u_i * (v_i^T g_y))^2] (joint moment)
                    q = u_flat_f * s_flat
                    state["loss_proxy_sum"] += (q * q).sum(dim=0)

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
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
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
            # Exact joint estimator used by the default definition
            loss_proxy_joint = 0.5 * (state["loss_proxy_sum"] / float(count))
            # Diagnostic: separable approximation (can diverge if u^2 and (v^T g)^2 correlate)
            loss_proxy_factored = 0.5 * u2_mean * R_vals

            scar_scores[layer_name] = {
                "scar_activation_power": u2_mean,
                "scar_taylor": T_vals,
                "scar_curvature": R_vals,
                "scar_loss_proxy": loss_proxy_joint,
                "scar_loss_proxy_factored": loss_proxy_factored,
            }

            # Also attach these scores into importance_scores for later use in pruning
            layer_scores = self.importance_scores.get(layer_name, {})
            layer_scores["scar_activation_power"] = u2_mean
            layer_scores["scar_taylor"] = T_vals
            layer_scores["scar_curvature"] = R_vals
            layer_scores["scar_loss_proxy"] = loss_proxy_joint
            layer_scores["scar_loss_proxy_factored"] = loss_proxy_factored
            # Now that scar_loss_proxy exists, we can compute the configured supernode mask on this layer.
            # This ensures 'protect_core' works during pruning even when score_metric='scar_loss_proxy'.
            self._apply_supernode_selection(layer_scores, composite=None)
            # Propagate the supernode mask to sibling MLP projections so that channel-level protection
            # works regardless of which projection holds the pruning scores (e.g., Wanda stores channel
            # scores on gate/up/down; alignment metrics often live on gate/up).
            try:
                mask = layer_scores.get("supernode_mask")
                if mask is not None and isinstance(layer_name, str) and "down_proj" in layer_name:
                    for sibling_proj in ("gate_proj", "up_proj"):
                        sibling_name = layer_name.replace("down_proj", sibling_proj)
                        if sibling_name in self.importance_scores:
                            sib_scores = self.importance_scores.get(sibling_name, {})
                            sib_scores["supernode_mask"] = mask
                            if "supernode_core_size" in layer_scores:
                                sib_scores["supernode_core_size"] = layer_scores["supernode_core_size"]
                            if "supernode_threshold" in layer_scores:
                                sib_scores["supernode_threshold"] = layer_scores["supernode_threshold"]
                            self.importance_scores[sibling_name] = sib_scores
            except Exception as _prop_err:
                logger.debug(f"Failed to propagate supernode mask for {layer_name}: {_prop_err}")
            self.importance_scores[layer_name] = layer_scores

        logger.info(f"SCAR metrics: computed metrics for {len(scar_scores)} FFN layers.")

        return scar_scores

    def compute_attention_scar_metrics(
        self,
        num_samples: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute SCAR-style supernode metrics for attention heads in transformer layers.

        This routine performs forward+backward passes on calibration data and uses hooks on
        attention output projection modules (o_proj) to compute per-head loss sensitivity.

        For each attention head h:
            - o_h: output of head h (portion of o_proj input corresponding to head h)
            - g_o_h: gradient w.r.t. head h's output

        Metrics per head h:
            activation_power_h = E[||o_h||^2]
            gradient_power_h   = E[||g_o_h||^2]
            taylor_h           = E[|<g_o_h, o_h>|]  (first-order saliency)
            loss_proxy_h       = 0.5 * E[(||o_h|| * ||g_o_h||)^2]  (joint second moment)

        This is analogous to FFN channel analysis but applied to attention heads.
        """
        if not getattr(self.config, "do_attention_scar_metrics", False):
            logger.info("Attention SCAR metrics disabled in config; skipping compute_attention_scar_metrics.")
            return {}

        logger.info("Computing SCAR-style supernode metrics for attention heads...")

        # Determine calibration texts (same logic as FFN SCAR)
        calibration_texts: List[str] = []
        if getattr(self.config, "importance_computation_texts", None):
            calibration_texts = list(self.config.importance_computation_texts)
        else:
            if getattr(self, "dataset", None) is not None:
                if hasattr(self.dataset, "texts"):
                    calibration_texts = list(self.dataset.texts)
                else:
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
                        logger.error(f"Attention SCAR metrics: failed to iterate over dataset: {e}")
                        calibration_texts = []

        if not calibration_texts:
            raise RuntimeError(
                "Attention SCAR metrics: no calibration texts available. "
                "Run importance computation first or ensure the dataset provides raw texts."
            )

        if num_samples is None or num_samples <= 0:
            num_samples = getattr(self.config, "scar_num_samples", 0) or self.config.alignment_data_num_samples
        max_length = max_length or getattr(self.config, "scar_max_length", 512)

        num_samples = min(num_samples, len(calibration_texts))
        logger.info(f"Attention SCAR metrics will use {num_samples} calibration samples (max_length={max_length}).")

        device = torch.device(self.config.device)

        # Get underlying HF model
        hf_model: nn.Module = self.model
        if hasattr(hf_model, "model"):
            hf_model = getattr(hf_model, "model")

        # Detect model architecture parameters
        num_heads = None
        head_dim = None
        if hasattr(hf_model, "config"):
            config = hf_model.config
            num_heads = getattr(config, "num_attention_heads", None)
            hidden_size = getattr(config, "hidden_size", None)
            if num_heads and hidden_size:
                head_dim = hidden_size // num_heads

        if num_heads is None or head_dim is None:
            logger.warning("Could not detect num_heads/head_dim from model config, trying to infer...")
            # Try to infer from first attention layer
            for name, module in hf_model.named_modules():
                if "self_attn" in name and hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                    head_dim = module.head_dim if hasattr(module, "head_dim") else None
                    break

        if num_heads is None:
            raise RuntimeError("Could not determine number of attention heads from model.")

        logger.info(f"Detected {num_heads} attention heads, head_dim={head_dim}")

        attn_scar_state: Dict[str, Dict[str, Any]] = {}
        hooks: List[Any] = []

        # Create hooks on all attention o_proj modules (output projection)
        for layer_name, module in hf_model.named_modules():
            # Match patterns like: model.layers.X.self_attn.o_proj
            if not ("self_attn" in layer_name and "o_proj" in layer_name):
                continue
            if not isinstance(module, nn.Linear):
                continue

            # Extract layer index for grouping
            import re

            layer_match = re.search(r"layers\.(\d+)", layer_name)
            layer_idx = layer_match.group(1) if layer_match else layer_name

            attn_scar_state[layer_name] = {
                "layer_idx": layer_idx,
                "num_heads": num_heads,
                "head_dim": head_dim,
                # Per-head accumulators [num_heads]
                "head_act_power_sum": None,  # sum of ||o_h||^2
                "head_grad_power_sum": None,  # sum of ||g_o_h||^2
                "head_taylor_sum": None,  # sum of |<g_o_h, o_h>|
                "head_loss_proxy_sum": None,  # sum of (||o_h|| * ||g_o_h||)^2
                "token_count": 0,
            }

            def make_hooks(name: str, n_heads: int, h_dim: int):
                def fwd_hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
                    # inputs[0] is the concatenated head outputs before o_proj: [B, T, num_heads * head_dim]
                    if not inputs:
                        return
                    x = inputs[0]
                    if x is None:
                        return

                    x_flat = x.detach()
                    # Reshape to [B*T, num_heads, head_dim]
                    if x_flat.ndim == 3:
                        B, T, D = x_flat.shape
                        x_flat = x_flat.reshape(B * T, n_heads, h_dim)
                    elif x_flat.ndim == 2:
                        N, D = x_flat.shape
                        x_flat = x_flat.reshape(N, n_heads, h_dim)
                    else:
                        return

                    state = attn_scar_state[name]

                    if state["head_act_power_sum"] is None:
                        state["head_act_power_sum"] = torch.zeros(n_heads, device=x_flat.device, dtype=torch.float32)
                        state["head_grad_power_sum"] = torch.zeros_like(state["head_act_power_sum"])
                        state["head_taylor_sum"] = torch.zeros_like(state["head_act_power_sum"])
                        state["head_loss_proxy_sum"] = torch.zeros_like(state["head_act_power_sum"])

                    # Compute per-head activation power: ||o_h||^2 for each head
                    # x_flat: [N_tokens, num_heads, head_dim]
                    head_norms_sq = (x_flat.float() ** 2).sum(dim=-1)  # [N_tokens, num_heads]
                    state["head_act_power_sum"] += head_norms_sq.sum(dim=0)  # [num_heads]
                    state["token_count"] += x_flat.shape[0]

                    # Store for backward hook
                    mod._attn_scar_last_input = x.detach()

                def bwd_hook(mod: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
                    state = attn_scar_state[name]

                    # grad_input[0] is gradient w.r.t. the input to o_proj (the concatenated heads)
                    if not grad_input or grad_input[0] is None:
                        return

                    g_x = grad_input[0]

                    if not hasattr(mod, "_attn_scar_last_input"):
                        return

                    x = mod._attn_scar_last_input
                    delattr(mod, "_attn_scar_last_input")

                    # Reshape both to [N_tokens, num_heads, head_dim]
                    if x.ndim == 3:
                        B, T, D = x.shape
                        x_flat = x.reshape(B * T, n_heads, h_dim)
                        g_flat = g_x.reshape(B * T, n_heads, h_dim)
                    elif x.ndim == 2:
                        N, D = x.shape
                        x_flat = x.reshape(N, n_heads, h_dim)
                        g_flat = g_x.reshape(N, n_heads, h_dim)
                    else:
                        return

                    x_f = x_flat.float()
                    g_f = g_flat.float()

                    # Per-head gradient power: ||g_o_h||^2
                    head_grad_norms_sq = (g_f**2).sum(dim=-1)  # [N_tokens, num_heads]
                    state["head_grad_power_sum"] += head_grad_norms_sq.sum(dim=0)

                    # Per-head Taylor saliency: |<g_o_h, o_h>|
                    head_inner = (g_f * x_f).sum(dim=-1)  # [N_tokens, num_heads]
                    state["head_taylor_sum"] += head_inner.abs().sum(dim=0)

                    # Per-head loss proxy: (||o_h|| * ||g_o_h||)^2
                    head_act_norms = (x_f**2).sum(dim=-1).sqrt()  # [N_tokens, num_heads]
                    head_grad_norms = head_grad_norms_sq.sqrt()
                    head_proxy_contrib = (head_act_norms * head_grad_norms) ** 2
                    state["head_loss_proxy_sum"] += head_proxy_contrib.sum(dim=0)

                return fwd_hook, bwd_hook

            if head_dim is not None:
                fwd_hook, bwd_hook = make_hooks(layer_name, num_heads, head_dim)
                hooks.append(module.register_forward_hook(fwd_hook))
                hooks.append(module.register_full_backward_hook(bwd_hook))

        if not attn_scar_state:
            logger.warning("Attention SCAR metrics: no attention o_proj modules found; skipping.")
            return {}

        # Calibration loop
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

                labels = inputs["input_ids"].clone()
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
                labels[labels == pad_token_id] = -100
                inputs["labels"] = labels

                self.model.zero_grad(set_to_none=True)
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()

                if (idx + 1) % 10 == 0:
                    logger.info(f"Attention SCAR: processed sample {idx+1}/{num_samples}, loss={loss.item():.4f}")

        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # Aggregate metrics per layer
        attn_scar_scores: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_name, state in attn_scar_state.items():
            count = state["token_count"]
            if count <= 0 or state["head_act_power_sum"] is None:
                continue

            n_tokens = float(count)

            head_act_power = state["head_act_power_sum"] / n_tokens
            head_grad_power = state["head_grad_power_sum"] / n_tokens
            head_taylor = state["head_taylor_sum"] / n_tokens
            head_loss_proxy = 0.5 * (state["head_loss_proxy_sum"] / n_tokens)

            attn_scar_scores[layer_name] = {
                "attn_activation_power": head_act_power,
                "attn_gradient_power": head_grad_power,
                "attn_taylor": head_taylor,
                "attn_loss_proxy": head_loss_proxy,
                "layer_idx": state["layer_idx"],
                "num_heads": state["num_heads"],
            }

            # Store in importance_scores for later use
            layer_scores = self.importance_scores.get(layer_name, {})
            layer_scores["attn_activation_power"] = head_act_power
            layer_scores["attn_gradient_power"] = head_grad_power
            layer_scores["attn_taylor"] = head_taylor
            layer_scores["attn_loss_proxy"] = head_loss_proxy
            self.importance_scores[layer_name] = layer_scores

        logger.info(f"Attention SCAR metrics: computed metrics for {len(attn_scar_scores)} attention layers.")

        # Compute summary statistics for comparison with FFN
        if attn_scar_scores:
            all_lp = torch.cat([s["attn_loss_proxy"] for s in attn_scar_scores.values()])
            top_k = max(1, int(0.1 * len(all_lp)))  # Top 10% for attention (vs 1% for FFN)
            sorted_lp, _ = torch.sort(all_lp, descending=True)
            top_mass = sorted_lp[:top_k].sum() / (all_lp.sum() + 1e-8)
            cv = all_lp.std() / (all_lp.mean() + 1e-8)

            logger.info(f"Attention SCAR summary: top-10% heads capture {top_mass:.1%} of total loss proxy mass")
            logger.info(f"Attention SCAR summary: coefficient of variation = {cv:.2f}")

        return attn_scar_scores

    def compute_baseline_pruning_scores(
        self,
        strategies: Optional[List[str]] = None,
        num_calibration_samples: int = 128,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute importance scores for baseline pruning methods (Wanda, SparseGPT).

        These methods require calibration data to compute activation-aware scores.
        Scores are stored in self.importance_scores for use in pruning experiments.

        Args:
            strategies: List of baseline strategies to compute. Default: ["wanda", "sparsegpt", "owl", "llm_pruner"]
            num_calibration_samples: Number of samples for calibration

        Returns:
            Dict mapping layer names to {strategy_name: scores_tensor}
        """
        if strategies is None:
            # Check which baseline strategies are configured
            pruning_strategies = getattr(self.config, "pruning_strategies", [])
            strategies = [
                s for s in pruning_strategies if s in ["wanda", "sparsegpt", "owl", "llm_pruner", "flap", "ria", "slimllm", "flap", "ria", "slimllm"]
            ]

        if not strategies:
            logger.info("No baseline pruning strategies (wanda/sparsegpt) configured, skipping.")
            return {}

        logger.info(f"Computing baseline pruning scores for: {strategies}")

        # Get calibration dataloader
        try:
            from torch.utils.data import DataLoader

            from alignment.dataops.datasets.text_datasets import WikiTextDataset

            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                logger.error("Tokenizer not available for baseline score calibration")
                return {}

            # Create calibration dataset and dataloader
            calib_dataset = WikiTextDataset(
                tokenizer=tokenizer,
                split="train",
                max_length=getattr(self.config, "scar_max_length", 512),
            )
            # Limit samples
            if len(calib_dataset) > num_calibration_samples:
                from torch.utils.data import Subset

                indices = list(range(min(num_calibration_samples, len(calib_dataset))))
                calib_dataset = Subset(calib_dataset, indices)

            calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=False)
            logger.info(f"Created calibration dataloader with {len(calib_dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create calibration dataloader: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {}

        results = {}
        model = self.wrapped_model._model
        device = next(model.parameters()).device

        # ---------------------------------------------------------------------
        # IMPORTANT: Channel/group adaptation for structured FFN pruning
        #
        # A "channel" corresponds to:
        # - row i of gate_proj and up_proj (out_features = intermediate_dim)
        # - column i of down_proj (in_features = intermediate_dim)
        #
        # So for baseline methods (Wanda/SparseGPT) we compute a *group score* per channel:
        #   score_i = score_gate_row_i + score_up_row_i + score_down_col_i
        # and store that 1D score (length = intermediate_dim) for pruning.
        # ---------------------------------------------------------------------
        import re

        layer_indices = set()
        for k in self.importance_scores.keys():
            m = re.search(r"layers\.(\d+)\.mlp", k)
            if m:
                layer_indices.add(int(m.group(1)))

        if not layer_indices:
            logger.warning("No MLP layers found in importance_scores; cannot compute baseline channel scores.")
            return {}

        underlying_model = self._get_underlying_model()
        module_dict = dict(underlying_model.named_modules())

        def _resolve_mlp_path(layer_idx: int) -> Optional[str]:
            candidates = [
                f"model.model.layers.{layer_idx}.mlp",
                f"model.layers.{layer_idx}.mlp",
                f"layers.{layer_idx}.mlp",
            ]
            for p in candidates:
                if p in module_dict:
                    return p
            return None

        logger.info(f"Computing baseline channel scores for {len(layer_indices)} MLP layers")

        # Compute Wanda scores
        if "wanda" in strategies:
            logger.info("Calibrating Wanda pruning strategy...")
            try:
                wanda = WandaPruning(num_calibration_samples=num_calibration_samples)
                wanda.calibrate(model, calib_dataloader, device=str(device))
                # Keep the calibrated object for optional unstructured reproduction baselines.
                self._wanda_baseline = wanda

                for layer_idx in sorted(layer_indices):
                    mlp_path = _resolve_mlp_path(layer_idx)
                    if mlp_path is None:
                        logger.warning(f"Wanda: could not resolve MLP path for layer {layer_idx}")
                        continue

                    gate_name = f"{mlp_path}.gate_proj"
                    up_name = f"{mlp_path}.up_proj"
                    down_name = f"{mlp_path}.down_proj"

                    if gate_name not in module_dict or up_name not in module_dict or down_name not in module_dict:
                        logger.warning(f"Wanda: missing projections for {mlp_path}")
                        continue

                    gate = module_dict[gate_name]
                    up = module_dict[up_name]
                    down = module_dict[down_name]

                    if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
                        logger.warning(f"Wanda: projections for {mlp_path} are not all nn.Linear; skipping")
                        continue

                    try:
                        # gate/up: per output channel (rows) => dim=0
                        gate_scores = wanda.get_structured_scores(gate, layer_name=gate_name, dim=0)
                        up_scores = wanda.get_structured_scores(up, layer_name=up_name, dim=0)
                        # down: per input channel (columns) => dim=1
                        down_scores = wanda.get_structured_scores(down, layer_name=down_name, dim=1)

                        channel_scores = (gate_scores + up_scores + down_scores).detach()

                        # Store the channel-group score for pruning under all three projection names
                        for store_name in (gate_name, up_name, down_name):
                            if store_name not in self.importance_scores:
                                self.importance_scores[store_name] = {}
                            self.importance_scores[store_name]["wanda"] = channel_scores

                            if store_name not in results:
                                results[store_name] = {}
                            results[store_name]["wanda"] = channel_scores

                        logger.debug(
                            f"Wanda channel scores for {mlp_path}: shape={tuple(channel_scores.shape)}, mean={channel_scores.mean().item():.4f}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute Wanda channel scores for {mlp_path}: {e}")
                        continue

                logger.info(f"Wanda: computed channel scores for {len(layer_indices)} MLP layers")
            except Exception as e:
                logger.error(f"Wanda calibration failed: {e}")
                import traceback

                logger.error(traceback.format_exc())

        # Compute SparseGPT scores
        if "sparsegpt" in strategies:
            logger.info("Calibrating SparseGPT pruning strategy...")
            try:
                sparsegpt = SparseGPTPruning(num_calibration_samples=num_calibration_samples)
                sparsegpt.calibrate(model, calib_dataloader, device=str(device))
                # Keep the calibrated object for optional unstructured reproduction baselines.
                self._sparsegpt_baseline = sparsegpt

                for layer_idx in sorted(layer_indices):
                    mlp_path = _resolve_mlp_path(layer_idx)
                    if mlp_path is None:
                        logger.warning(f"SparseGPT: could not resolve MLP path for layer {layer_idx}")
                        continue

                    gate_name = f"{mlp_path}.gate_proj"
                    up_name = f"{mlp_path}.up_proj"
                    down_name = f"{mlp_path}.down_proj"

                    if gate_name not in module_dict or up_name not in module_dict or down_name not in module_dict:
                        logger.warning(f"SparseGPT: missing projections for {mlp_path}")
                        continue

                    gate = module_dict[gate_name]
                    up = module_dict[up_name]
                    down = module_dict[down_name]

                    if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
                        logger.warning(f"SparseGPT: projections for {mlp_path} are not all nn.Linear; skipping")
                        continue

                    try:
                        gate_scores = sparsegpt.get_structured_scores(gate, layer_name=gate_name, dim=0)
                        up_scores = sparsegpt.get_structured_scores(up, layer_name=up_name, dim=0)
                        down_scores = sparsegpt.get_structured_scores(down, layer_name=down_name, dim=1)

                        channel_scores = (gate_scores + up_scores + down_scores).detach()

                        for store_name in (gate_name, up_name, down_name):
                            if store_name not in self.importance_scores:
                                self.importance_scores[store_name] = {}
                            self.importance_scores[store_name]["sparsegpt"] = channel_scores

                            if store_name not in results:
                                results[store_name] = {}
                            results[store_name]["sparsegpt"] = channel_scores

                        logger.debug(
                            f"SparseGPT channel scores for {mlp_path}: shape={tuple(channel_scores.shape)}, mean={channel_scores.mean().item():.4f}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute SparseGPT channel scores for {mlp_path}: {e}")
                        continue

                logger.info(f"SparseGPT: computed channel scores for {len(layer_indices)} MLP layers")
            except Exception as e:
                logger.error(f"SparseGPT calibration failed: {e}")
                import traceback

                logger.error(traceback.format_exc())

        # Compute OWL scores (outlier-aware Wanda)
        if "owl" in strategies:
            logger.info("Calibrating OWL (Outlier-aware Wanda) pruning strategy...")
            try:
                from alignment.pruning.strategies.llm_baselines import OWLPruning

                owl = OWLPruning(num_calibration_samples=num_calibration_samples)
                owl.calibrate(model, calib_dataloader, device=str(device))
                self._owl_baseline = owl

                for layer_idx in sorted(layer_indices):
                    mlp_path = _resolve_mlp_path(layer_idx)
                    if mlp_path is None:
                        continue

                    gate_name = f"{mlp_path}.gate_proj"
                    up_name = f"{mlp_path}.up_proj"
                    down_name = f"{mlp_path}.down_proj"

                    if gate_name not in module_dict or up_name not in module_dict or down_name not in module_dict:
                        continue

                    gate = module_dict[gate_name]
                    up = module_dict[up_name]
                    down = module_dict[down_name]

                    if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
                        continue

                    try:
                        gate_scores = owl.get_structured_scores(gate, layer_name=gate_name, dim=0)
                        up_scores = owl.get_structured_scores(up, layer_name=up_name, dim=0)
                        down_scores = owl.get_structured_scores(down, layer_name=down_name, dim=1)

                        channel_scores = (gate_scores + up_scores + down_scores).detach()

                        for store_name in (gate_name, up_name, down_name):
                            if store_name not in self.importance_scores:
                                self.importance_scores[store_name] = {}
                            self.importance_scores[store_name]["owl"] = channel_scores

                            if store_name not in results:
                                results[store_name] = {}
                            results[store_name]["owl"] = channel_scores
                    except Exception as e:
                        logger.warning(f"Failed to compute OWL channel scores for {mlp_path}: {e}")
                        continue

                logger.info(f"OWL: computed channel scores for {len(layer_indices)} MLP layers")
            except Exception as e:
                logger.error(f"OWL calibration failed: {e}")
                import traceback

                logger.error(traceback.format_exc())

        # Compute LLM-Pruner scores (Taylor-based)
        if "llm_pruner" in strategies:
            logger.info("Calibrating LLM-Pruner pruning strategy...")
            try:
                from alignment.pruning.strategies.llm_baselines import LLMPrunerChannelMode

                llm_pruner = LLMPrunerChannelMode(num_calibration_samples=num_calibration_samples)
                llm_pruner.calibrate(model, calib_dataloader, device=str(device))
                self._llmpruner_baseline = llm_pruner

                for layer_idx in sorted(layer_indices):
                    mlp_path = _resolve_mlp_path(layer_idx)
                    if mlp_path is None:
                        continue

                    gate_name = f"{mlp_path}.gate_proj"
                    up_name = f"{mlp_path}.up_proj"
                    down_name = f"{mlp_path}.down_proj"

                    if gate_name not in module_dict or up_name not in module_dict or down_name not in module_dict:
                        continue

                    gate = module_dict[gate_name]
                    up = module_dict[up_name]
                    down = module_dict[down_name]

                    if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
                        continue

                    try:
                        gate_scores = llm_pruner.get_structured_scores(gate, layer_name=gate_name, dim=0)
                        up_scores = llm_pruner.get_structured_scores(up, layer_name=up_name, dim=0)
                        down_scores = llm_pruner.get_structured_scores(down, layer_name=down_name, dim=1)

                        channel_scores = (gate_scores + up_scores + down_scores).detach()

                        for store_name in (gate_name, up_name, down_name):
                            if store_name not in self.importance_scores:
                                self.importance_scores[store_name] = {}
                            self.importance_scores[store_name]["llm_pruner"] = channel_scores

                            if store_name not in results:
                                results[store_name] = {}
                            results[store_name]["llm_pruner"] = channel_scores
                    except Exception as e:
                        logger.warning(f"Failed to compute LLM-Pruner channel scores for {mlp_path}: {e}")
                        continue

                logger.info(f"LLM-Pruner: computed channel scores for {len(layer_indices)} MLP layers")
            except Exception as e:
                logger.error(f"LLM-Pruner calibration failed: {e}")
                import traceback

                logger.error(traceback.format_exc())

        # Compute FLAP scores (Fluctuation-based)
        if "flap" in strategies:
            logger.info("Calibrating FLAP pruning strategy...")
            try:
                from alignment.pruning.strategies.llm_baselines import FLAPPruning

                flap = FLAPPruning(num_calibration_samples=num_calibration_samples)
                flap.calibrate(model, calib_dataloader, device=str(device))
                self._flap_baseline = flap

                for layer_idx in sorted(layer_indices):
                    mlp_path = _resolve_mlp_path(layer_idx)
                    if mlp_path is None:
                        continue
                    gate_name = f"{mlp_path}.gate_proj"
                    up_name = f"{mlp_path}.up_proj"
                    down_name = f"{mlp_path}.down_proj"
                    if gate_name not in module_dict:
                        continue
                    gate, up, down = module_dict[gate_name], module_dict[up_name], module_dict[down_name]
                    try:
                        g_s = flap.get_structured_scores(gate, layer_name=gate_name, dim=0)
                        u_s = flap.get_structured_scores(up, layer_name=up_name, dim=0)
                        d_s = flap.get_structured_scores(down, layer_name=down_name, dim=1)
                        ch_sc = (g_s + u_s + d_s).detach()
                        for sn in (gate_name, up_name, down_name):
                            if sn not in self.importance_scores:
                                self.importance_scores[sn] = {}
                            self.importance_scores[sn]["flap"] = ch_sc
                            if sn not in results:
                                results[sn] = {}
                            results[sn]["flap"] = ch_sc
                    except Exception as e:
                        logger.warning(f"FLAP failed for {mlp_path}: {e}")
                logger.info(f"FLAP: computed for {len(layer_indices)} layers")
            except Exception as e:
                logger.error(f"FLAP calibration failed: {e}")

        # Compute RIA scores (Relative Importance × Activation)
        if "ria" in strategies:
            logger.info("Calibrating RIA pruning strategy...")
            try:
                from alignment.pruning.strategies.llm_baselines import RIAPruning

                ria = RIAPruning(num_calibration_samples=num_calibration_samples)
                ria.calibrate(model, calib_dataloader, device=str(device))
                self._ria_baseline = ria

                for layer_idx in sorted(layer_indices):
                    mlp_path = _resolve_mlp_path(layer_idx)
                    if mlp_path is None:
                        continue
                    gate_name = f"{mlp_path}.gate_proj"
                    up_name = f"{mlp_path}.up_proj"
                    down_name = f"{mlp_path}.down_proj"
                    if gate_name not in module_dict:
                        continue
                    gate, up, down = module_dict[gate_name], module_dict[up_name], module_dict[down_name]
                    try:
                        g_s = ria.get_structured_scores(gate, layer_name=gate_name, dim=0)
                        u_s = ria.get_structured_scores(up, layer_name=up_name, dim=0)
                        d_s = ria.get_structured_scores(down, layer_name=down_name, dim=1)
                        ch_sc = (g_s + u_s + d_s).detach()
                        for sn in (gate_name, up_name, down_name):
                            if sn not in self.importance_scores:
                                self.importance_scores[sn] = {}
                            self.importance_scores[sn]["ria"] = ch_sc
                            if sn not in results:
                                results[sn] = {}
                            results[sn]["ria"] = ch_sc
                    except Exception as e:
                        logger.warning(f"RIA failed for {mlp_path}: {e}")
                logger.info(f"RIA: computed for {len(layer_indices)} layers")
            except Exception as e:
                logger.error(f"RIA calibration failed: {e}")

        # Compute SlimLLM scores (holistic channel importance)
        if "slimllm" in strategies:
            logger.info("Calibrating SlimLLM pruning strategy...")
            try:
                from alignment.pruning.strategies.llm_baselines import SlimLLMPruning

                slimllm = SlimLLMPruning(num_calibration_samples=num_calibration_samples)
                slimllm.calibrate(model, calib_dataloader, device=str(device))
                self._slimllm_baseline = slimllm

                for layer_idx in sorted(layer_indices):
                    mlp_path = _resolve_mlp_path(layer_idx)
                    if mlp_path is None:
                        continue
                    gate_name = f"{mlp_path}.gate_proj"
                    up_name = f"{mlp_path}.up_proj"
                    down_name = f"{mlp_path}.down_proj"
                    if gate_name not in module_dict:
                        continue
                    gate, up, down = module_dict[gate_name], module_dict[up_name], module_dict[down_name]
                    try:
                        g_s = slimllm.get_structured_scores(gate, layer_name=gate_name, dim=0)
                        u_s = slimllm.get_structured_scores(up, layer_name=up_name, dim=0)
                        d_s = slimllm.get_structured_scores(down, layer_name=down_name, dim=1)
                        ch_sc = (g_s + u_s + d_s).detach()
                        for sn in (gate_name, up_name, down_name):
                            if sn not in self.importance_scores:
                                self.importance_scores[sn] = {}
                            self.importance_scores[sn]["slimllm"] = ch_sc
                            if sn not in results:
                                results[sn] = {}
                            results[sn]["slimllm"] = ch_sc
                    except Exception as e:
                        logger.warning(f"SlimLLM failed for {mlp_path}: {e}")
                logger.info(f"SlimLLM: computed for {len(layer_indices)} layers")
            except Exception as e:
                logger.error(f"SlimLLM calibration failed: {e}")

        return results

    def compute_weight_magnitude_channel_scores(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute a fast, calibration-free structured *channel* baseline using weight magnitudes.

        For each MLP layer and intermediate channel i:
          score_i = ||W_gate[i,:]||_2 + ||W_up[i,:]||_2 + ||W_down[:,i]||_2

        This matches the "Magnitude (channel)" baseline commonly used in structured pruning comparisons.

        Returns:
            Dict mapping module_name -> {"weight_magnitude": score_tensor}
        """
        import re

        underlying_model = self._get_underlying_model()
        module_dict = dict(underlying_model.named_modules())

        # Identify MLP layer indices based on already-tracked layer names
        layer_indices = set()
        for k in self.importance_scores.keys():
            m = re.search(r"layers\.(\d+)\.mlp", k)
            if m:
                layer_indices.add(int(m.group(1)))

        if not layer_indices:
            logger.warning("weight_magnitude: no MLP layers found in importance_scores; skipping")
            return {}

        def _resolve_mlp_path(layer_idx: int) -> Optional[str]:
            candidates = [
                f"model.model.layers.{layer_idx}.mlp",
                f"model.layers.{layer_idx}.mlp",
                f"layers.{layer_idx}.mlp",
            ]
            for p in candidates:
                if p in module_dict:
                    return p
            return None

        results: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_idx in sorted(layer_indices):
            mlp_path = _resolve_mlp_path(layer_idx)
            if mlp_path is None:
                logger.warning(f"weight_magnitude: could not resolve MLP path for layer {layer_idx}")
                continue

            gate_name = f"{mlp_path}.gate_proj"
            up_name = f"{mlp_path}.up_proj"
            down_name = f"{mlp_path}.down_proj"

            if gate_name not in module_dict or up_name not in module_dict or down_name not in module_dict:
                logger.warning(f"weight_magnitude: missing projections for {mlp_path}")
                continue

            gate = module_dict[gate_name]
            up = module_dict[up_name]
            down = module_dict[down_name]

            if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
                logger.warning(f"weight_magnitude: projections for {mlp_path} are not all nn.Linear; skipping")
                continue

            # gate/up: row norms (out_features = intermediate_dim)
            gate_score = torch.norm(gate.weight.detach().float(), p=2, dim=1)
            up_score = torch.norm(up.weight.detach().float(), p=2, dim=1)
            # down: column norms (in_features = intermediate_dim)
            down_score = torch.norm(down.weight.detach().float(), p=2, dim=0)

            channel_scores = (gate_score + up_score + down_score).detach()

            for store_name in (gate_name, up_name, down_name):
                if store_name not in self.importance_scores:
                    self.importance_scores[store_name] = {}
                self.importance_scores[store_name]["weight_magnitude"] = channel_scores

                if store_name not in results:
                    results[store_name] = {}
                results[store_name]["weight_magnitude"] = channel_scores

        logger.info(f"Computed weight_magnitude channel scores for {len(layer_indices)} MLP layers")
        return results

    def compute_random_channel_scores(
        self,
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Structured *channel* random baseline.

        We generate one random score per intermediate FFN channel (shared across gate/up/down
        projections) and store it under metric name "random" in `self.importance_scores`.

        Note: If pruning_mode == "random", the pruning mask creation ignores score values and
        uses uniform random selection; we still store scores to provide consistent shapes and
        to make this baseline explicit in saved artifacts.
        """
        import re

        if seed is None:
            seed = int(getattr(self.config, "seed", 0) or 0)

        underlying_model = self._get_underlying_model()
        module_dict = dict(underlying_model.named_modules())

        # Identify MLP layer indices by scanning module names (robust even if no other
        # importance scores were computed).
        layer_indices = set()
        for name in module_dict.keys():
            m = re.search(r"layers\.(\d+)\.mlp\.gate_proj$", name)
            if m:
                layer_indices.add(int(m.group(1)))

        if not layer_indices:
            logger.warning("random: no MLP layers found; skipping random channel baseline")
            return {}

        # Use a dedicated generator for determinism.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        def _resolve_mlp_path(layer_idx: int) -> Optional[str]:
            candidates = [
                f"model.model.layers.{layer_idx}.mlp",
                f"model.layers.{layer_idx}.mlp",
                f"layers.{layer_idx}.mlp",
            ]
            for c in candidates:
                if c in module_dict:
                    return c
            return None

        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for layer_idx in sorted(layer_indices):
            mlp_path = _resolve_mlp_path(layer_idx)
            if mlp_path is None:
                logger.warning(f"random: could not resolve MLP path for layer {layer_idx}")
                continue

            gate_name = f"{mlp_path}.gate_proj"
            up_name = f"{mlp_path}.up_proj"
            down_name = f"{mlp_path}.down_proj"
            if gate_name not in module_dict or up_name not in module_dict or down_name not in module_dict:
                logger.warning(f"random: missing projections for {mlp_path}")
                continue

            gate = module_dict[gate_name]
            up = module_dict[up_name]
            down = module_dict[down_name]
            if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
                logger.warning(f"random: projections for {mlp_path} are not all nn.Linear; skipping")
                continue

            n = int(gate.out_features)
            if n <= 0:
                continue

            # One score per intermediate channel.
            scores = torch.rand((n,), generator=gen, dtype=torch.float32)

            for store_name in (gate_name, up_name, down_name):
                if store_name not in self.importance_scores:
                    self.importance_scores[store_name] = {}
                self.importance_scores[store_name]["random"] = scores
                if store_name not in results:
                    results[store_name] = {}
                results[store_name]["random"] = scores

        logger.info(f"Computed random channel scores for {len(layer_indices)} MLP layers (seed={seed})")
        return results

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
        config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
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

    def _should_protect_supernodes_for_metric(self, metric: str) -> bool:
        """
        Decide whether supernode protection (i.e., forcing core channels to be kept) should be applied
        for a given pruning metric.

        Backward-compatible behavior:
        - If `supernode.protect_core_metrics` is NOT set, protection applies to *all* metrics
          (matching the legacy behavior when `protect_core: true`).
        """
        cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
        if not cfg.get("enabled", False):
            return False
        if not cfg.get("protect_core", True):
            return False

        # Internal ablation metrics construct their own "protected set" (LP vs random).
        # Applying *additional* LP-core protection here would contaminate the control
        # (random-supernode metrics would still protect true LP supernodes).
        if isinstance(metric, str) and metric.startswith("random_supernode_ablation_"):
            return False

        # Hit-rate sweep metrics intentionally control how many supernodes are pruned.
        # Applying protection would defeat the purpose of the experiment.
        if isinstance(metric, str) and metric.startswith("supernode_hit_rate_sweep_"):
            return False

        protect_metrics = cfg.get("protect_core_metrics", None)
        if protect_metrics is None:
            return True

        # Accept a few convenient string shorthands.
        if isinstance(protect_metrics, str):
            token = protect_metrics.strip().lower()
            if token in {"all", "true", "yes", "1"}:
                return True
            if token in {"none", "false", "no", "0", ""}:
                return False
            # comma-separated list
            protect_metrics = [m.strip() for m in protect_metrics.split(",") if m.strip()]

        try:
            return metric in set(protect_metrics)
        except TypeError:
            # If the config value is malformed, fall back to "protect everything" (safer).
            return True

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

        logger.info("Analyzing supernode connections:")
        logger.info(f"  - Supernode metric: {supernode_metric}")
        logger.info(f"  - Supernode fraction: top {supernode_fraction*100:.1f}%")
        logger.info(f"  - Cross-layer analysis: {cross_layer_analysis}")
        if cross_layer_analysis:
            logger.info(f"  - Follower fraction: top {follower_fraction*100:.1f}%")
        if target_layers:
            logger.info(f"  - Target layers: {target_layers}")
        else:
            logger.info("  - Target layers: all layers with SCAR scores")

        if plots_dir is None:
            plots_dir = Path(getattr(self.config, "plots_dir", "./plots"))
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create subfolders for organized plots
        supernode_plots_dir = plots_dir / "supernode"
        supernode_plots_dir.mkdir(parents=True, exist_ok=True)
        layer_analysis_dir = plots_dir / "layer_analysis"
        layer_analysis_dir.mkdir(parents=True, exist_ok=True)

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
                    # Skip wildcard patterns - they mean "all layers"
                    if "*" in target:
                        layer_matches = True
                        break
                    # Support both exact match and partial match (e.g., "model.layers.10" matches "model.layers.10.mlp.down_proj")
                    # Also handle prefix variations (model.model.layers vs model.layers)
                    target_normalized = target.replace("model.model.", "model.")
                    layer_normalized = layer_name.replace("model.model.", "model.")
                    if target_normalized in layer_normalized or layer_normalized in target_normalized:
                        layer_matches = True
                        break
                if not layer_matches:
                    continue

            # Get the metric for supernode identification (configurable).
            # SCAR metrics live in `scar_scores`, while baseline metrics such as
            # activation_l2_norm live in `self.importance_scores`.
            def _get_importance_metric(name: str, metric_name: str) -> Optional[torch.Tensor]:
                candidates = [name]
                normalized = name.replace("model.model.", "model.")
                denormalized = name.replace("model.", "model.model.", 1) if name.startswith("model.") else name
                for cand in (normalized, denormalized):
                    if cand not in candidates:
                        candidates.append(cand)
                for cand in candidates:
                    metric_scores = (self.importance_scores.get(cand) or {}).get(metric_name)
                    if metric_scores is not None:
                        return metric_scores
                return None

            supernode_scores = layer_metrics.get(supernode_metric)
            if supernode_scores is None:
                supernode_scores = _get_importance_metric(layer_name, supernode_metric)
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
            layer_suffix = layer_name.replace(".", "_")

            # Plot 1: Distribution of supernode scores (based on selected metric)
            try:
                fig = viz.plot_supernode_activation_distribution(
                    activation_values=supernode_scores,
                    threshold_value=sorted_vals[num_supernodes - 1].item(),
                    threshold_percentile=supernode_fraction,
                    layer_name=layer_name,
                    metric_name=supernode_metric,
                    save_path=supernode_plots_dir / f"supernode_score_dist_{layer_suffix}.png",
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
                    save_path=supernode_plots_dir / f"supernode_outgoing_weights_{layer_suffix}.png",
                )
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as e:
                logger.error(f"  Failed to plot outgoing weights: {e}")

            # Plot 3: Supernode influence on output neurons
            try:
                fig = viz.plot_supernode_influence(
                    influence_values=supernode_influence,
                    threshold_value=follower_vals[num_followers - 1].item(),
                    threshold_percentile=follower_fraction,
                    layer_name=layer_name,
                    save_path=supernode_plots_dir / f"supernode_influence_{layer_suffix}.png",
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

                    # Optional: cross-layer "read-halo" diagnostic.
                    # This does NOT affect pruning; it is an analysis-only probe.
                    try:
                        supernode_cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
                        rh_cfg = (
                            supernode_cfg.get("read_halo", {})
                            or supernode_cfg.get("read_halo_analysis", {})
                            or getattr(self.config, "read_halo_analysis", {})
                            or {}
                        )
                        if isinstance(rh_cfg, dict) and bool(rh_cfg.get("enabled", False)):
                            from alignment.analysis.read_halo_llm import ReadHaloConfig, compute_next_layer_read_halo

                            cfg = ReadHaloConfig(
                                enabled=True,
                                read_halo_fraction=float(rh_cfg.get("read_halo_fraction", rh_cfg.get("fraction", 0.10))),
                                num_texts=int(rh_cfg.get("num_texts", 4)),
                                max_length=int(rh_cfg.get("max_length", 256)),
                                random_seed=int(rh_cfg.get("random_seed", 0)),
                                compute_dependence=bool(rh_cfg.get("compute_dependence", False)),
                                dependence_max_points=int(rh_cfg.get("dependence_max_points", 20000)),
                            )

                            _m = self.model
                            if hasattr(_m, "model"):
                                _m = _m.model

                            calibration_texts: List[str] = []
                            if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
                                calibration_texts = list(self.dataset.texts)

                            read_halo_res = compute_next_layer_read_halo(
                                model=_m,
                                tokenizer=self.tokenizer,
                                device=torch.device(self.config.device),
                                source_layer_name=layer_name,
                                next_layer_idx=next_layer_idx,
                                follower_indices=follower_indices,
                                calibration_texts=calibration_texts,
                                cfg=cfg,
                                plots_dir=plots_dir,
                            )
                            layer_results["next_layer_read_halo"] = read_halo_res
                    except Exception as e:
                        logger.error(f"  Failed read-halo analysis: {e}")
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

    def analyze_supernode_robustness(
        self,
        supernode_fraction: float = 0.01,
        num_bootstrap_samples: int = 10,
        batch_size: int = 32,
        max_samples: int = 256,
        metrics: Optional[List[str]] = None,
        target_layers: Optional[List[str]] = None,
        plots_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the robustness and consistency of supernode identification.

        This analysis quantifies how stable supernode identification is across:
        1. **Different metrics** - Do different metrics identify the same neurons as supernodes?
        2. **Different data batches** - Are supernodes consistent across different input samples?

        Key outputs:
        - Jaccard similarity between supernode sets from different metrics
        - Rank correlation (Spearman) between importance scores from different metrics
        - Bootstrap stability: fraction of times each neuron is identified as supernode
        - Consistency heatmaps across metrics and batches

        Args:
            supernode_fraction: Fraction of neurons to consider as supernodes (top by score)
            num_bootstrap_samples: Number of bootstrap samples for stability analysis
            batch_size: Batch size for forward passes
            max_samples: Maximum number of samples per bootstrap
            metrics: List of metrics to compare. If None, uses:
                     ['scar_activation_power', 'scar_loss_proxy', 'rayleigh_quotient',
                      'gaussian_mi_analytic', 'activation_l2_norm']
            target_layers: Layer patterns to analyze (e.g., ['model.layers.15', 'model.layers.20'])
            plots_dir: Directory to save visualizations

        Returns:
            Dictionary with robustness analysis results including:
            - metric_jaccard: Jaccard similarity matrix between metrics
            - metric_spearman: Spearman correlation matrix between metrics
            - bootstrap_stability: Per-neuron stability scores
            - consistent_supernodes: Neurons that are supernodes across all metrics
        """
        import matplotlib.pyplot as plt
        from scipy import stats

        if metrics is None:
            metrics = ["scar_activation_power", "scar_loss_proxy", "scar_taylor", "rayleigh_quotient", "gaussian_mi_analytic", "activation_l2_norm"]

        logger.info("Analyzing supernode robustness:")
        logger.info(f"  - Supernode fraction: top {supernode_fraction*100:.1f}%")
        logger.info(f"  - Bootstrap samples: {num_bootstrap_samples}")
        logger.info(f"  - Metrics to compare: {metrics}")

        if plots_dir is None:
            plots_dir = Path(getattr(self.config, "plots_dir", "./plots"))
        plots_dir = Path(plots_dir)
        robustness_dir = plots_dir / "supernode_robustness"
        robustness_dir.mkdir(parents=True, exist_ok=True)

        # Get model and device
        next(self.model.parameters()).device
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        # Find down_proj layers to analyze
        down_proj_layers = []
        for name, module in hf_model.named_modules():
            if "mlp.down_proj" in name and hasattr(module, "weight"):
                # Filter by target_layers if specified
                if target_layers:
                    matches = any(t in name for t in target_layers)
                    if not matches:
                        continue
                down_proj_layers.append((name, module))

        if not down_proj_layers:
            logger.warning("No down_proj layers found for robustness analysis")
            return {}

        logger.info(f"  - Analyzing {len(down_proj_layers)} layers")

        # Get calibration texts
        texts = []
        if hasattr(self, "dataset") and self.dataset is not None:
            if hasattr(self.dataset, "texts"):
                texts = self.dataset.texts[:max_samples]
            elif hasattr(self.dataset, "__getitem__"):
                texts = [self.dataset[i] for i in range(min(len(self.dataset), max_samples))]

        if not texts:
            logger.warning("No calibration texts available, using default")
            texts = ["The quick brown fox jumps over the lazy dog."] * max_samples

        results = {}
        viz = UnifiedVisualizer()

        def _lookup_importance_metric(layer_name: str, metric_name: str) -> Optional[torch.Tensor]:
            # importance_scores keys can vary (model.layers vs model.model.layers, etc.)
            candidates = [
                layer_name,
                layer_name.replace("model.layers.", "model.model.layers."),
                layer_name.replace("model.model.layers.", "model.layers."),
                layer_name.replace("model.", ""),
            ]
            seen = set()
            for key in candidates:
                if key in seen:
                    continue
                seen.add(key)
                layer_scores = self.importance_scores.get(key) or {}
                metric_scores = layer_scores.get(metric_name)
                if metric_scores is not None:
                    return metric_scores
            return None

        for layer_name, layer_module in down_proj_layers:
            logger.info(f"\n  Analyzing layer: {layer_name}")

            # Get layer dimensions
            weight = layer_module.weight.data
            hidden_dim, intermediate_dim = weight.shape
            num_supernodes = max(1, int(supernode_fraction * intermediate_dim))

            logger.info(f"    - Intermediate dim: {intermediate_dim}, Supernodes: {num_supernodes}")

            layer_results = {
                "num_neurons": intermediate_dim,
                "num_supernodes": num_supernodes,
                "metrics_analyzed": [],
                "metric_scores": {},
                "metric_supernode_indices": {},
            }

            # =========================================================
            # Part 1: Compute scores for each metric (full dataset)
            # =========================================================
            logger.info("    Computing metric scores on full dataset...")

            # Run SCAR metrics if available
            scar_scores = {}
            if getattr(self.config, "do_scar_metrics", True):
                try:
                    # Temporarily compute SCAR for this analysis
                    scar_results = self.compute_scar_supernode_metrics(
                        num_samples=min(max_samples, 128),
                        max_length=getattr(self.config, "max_length", 512),
                    )
                    if layer_name in scar_results:
                        scar_scores = scar_results[layer_name]
                except Exception as e:
                    logger.warning(f"    Failed to compute SCAR metrics: {e}")

            # Collect all metric scores for this layer
            metric_scores_layer = {}
            for metric_name in metrics:
                if metric_name.startswith("scar_"):
                    # SCAR metrics from compute_scar_supernode_metrics
                    if metric_name in scar_scores:
                        metric_scores_layer[metric_name] = scar_scores[metric_name].float().cpu()
                else:
                    precomputed_scores = _lookup_importance_metric(layer_name, metric_name)
                    if precomputed_scores is not None:
                        metric_scores_layer[metric_name] = precomputed_scores.float().cpu()
                        continue
                    # Try computing on the fly
                    try:
                        if metric_name == "activation_l2_norm":
                            # Compute activation magnitude
                            scores = self._compute_activation_magnitude(layer_name, texts[:max_samples], batch_size)
                            if scores is not None:
                                metric_scores_layer[metric_name] = scores
                        elif metric_name == "rayleigh_quotient":
                            scores = self._compute_rq_for_layer(layer_name, texts[:max_samples], batch_size)
                            if scores is not None:
                                metric_scores_layer[metric_name] = scores
                        elif metric_name == "gaussian_mi_analytic":
                            scores = self._compute_mi_for_layer(layer_name, texts[:max_samples], batch_size)
                            if scores is not None:
                                metric_scores_layer[metric_name] = scores
                    except Exception as e:
                        logger.warning(f"    Could not compute {metric_name}: {e}")

            if len(metric_scores_layer) < 1:
                logger.warning("    No metric scores available for layer; skipping")
                continue

            # Identify supernodes for each metric
            metric_supernode_indices = {}
            for metric_name, scores in metric_scores_layer.items():
                scores_flat = scores.flatten()
                if scores_flat.numel() != intermediate_dim:
                    logger.warning(f"    Score dim {scores_flat.numel()} != intermediate_dim {intermediate_dim} for {metric_name}")
                    continue
                sorted_vals, sorted_indices = torch.sort(scores_flat, descending=True)
                supernode_idx = sorted_indices[:num_supernodes].numpy()
                metric_supernode_indices[metric_name] = set(supernode_idx.tolist())
                layer_results["metrics_analyzed"].append(metric_name)

            layer_results["metric_scores"] = {k: v.numpy().tolist() for k, v in metric_scores_layer.items()}
            layer_results["metric_supernode_indices"] = {k: list(v) for k, v in metric_supernode_indices.items()}

            # =========================================================
            # Part 2: Compute Jaccard similarity between metrics
            # =========================================================
            analyzed_metrics = layer_results["metrics_analyzed"]
            n_metrics = len(analyzed_metrics)

            jaccard_matrix = np.zeros((n_metrics, n_metrics))
            for i, m1 in enumerate(analyzed_metrics):
                for j, m2 in enumerate(analyzed_metrics):
                    set1 = metric_supernode_indices[m1]
                    set2 = metric_supernode_indices[m2]
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard_matrix[i, j] = intersection / union if union > 0 else 0

            layer_results["jaccard_matrix"] = jaccard_matrix.tolist()

            # =========================================================
            # Part 3: Compute Spearman correlation between metrics
            # =========================================================
            spearman_matrix = np.zeros((n_metrics, n_metrics))
            for i, m1 in enumerate(analyzed_metrics):
                for j, m2 in enumerate(analyzed_metrics):
                    scores1 = np.array(layer_results["metric_scores"][m1])
                    scores2 = np.array(layer_results["metric_scores"][m2])
                    if len(scores1) == len(scores2):
                        corr, _ = stats.spearmanr(scores1, scores2)
                        spearman_matrix[i, j] = corr if not np.isnan(corr) else 0

            layer_results["spearman_matrix"] = spearman_matrix.tolist()

            # =========================================================
            # Part 4: Bootstrap stability analysis
            # =========================================================
            logger.info(f"    Running bootstrap stability analysis ({num_bootstrap_samples} samples)...")

            # Track how many times each neuron is identified as supernode
            supernode_counts = np.zeros(intermediate_dim)
            bootstrap_supernode_sets = []

            for b in range(num_bootstrap_samples):
                # Bootstrap sample from texts
                bootstrap_indices = np.random.choice(len(texts), size=min(batch_size * 4, len(texts)), replace=True)
                bootstrap_texts = [texts[i] for i in bootstrap_indices]

                # Compute activation magnitude for this bootstrap sample
                try:
                    bootstrap_scores = self._compute_activation_magnitude(layer_name, bootstrap_texts, batch_size)
                    if bootstrap_scores is not None:
                        scores_flat = bootstrap_scores.flatten()
                        if scores_flat.numel() == intermediate_dim:
                            sorted_vals, sorted_indices = torch.sort(scores_flat, descending=True)
                            bootstrap_supernode_idx = sorted_indices[:num_supernodes].numpy()
                            supernode_counts[bootstrap_supernode_idx] += 1
                            bootstrap_supernode_sets.append(set(bootstrap_supernode_idx.tolist()))
                except Exception as e:
                    logger.warning(f"    Bootstrap {b} failed: {e}")

            # Normalize to get stability scores (0 to 1)
            stability_scores = supernode_counts / num_bootstrap_samples
            layer_results["bootstrap_stability"] = stability_scores.tolist()

            # Identify highly stable supernodes (appear in >80% of bootstrap samples)
            highly_stable_mask = stability_scores > 0.8
            highly_stable_count = np.sum(highly_stable_mask)
            layer_results["highly_stable_supernodes"] = np.where(highly_stable_mask)[0].tolist()
            layer_results["num_highly_stable"] = int(highly_stable_count)

            logger.info(f"    - Highly stable supernodes (>80%): {highly_stable_count}")

            # =========================================================
            # Part 5: Cross-metric consistency
            # =========================================================
            # Find neurons that are supernodes in ALL metrics
            if len(metric_supernode_indices) >= 2:
                consistent_supernodes = set.intersection(*metric_supernode_indices.values())
                layer_results["consistent_across_all_metrics"] = list(consistent_supernodes)
                layer_results["num_consistent"] = len(consistent_supernodes)
                logger.info(f"    - Consistent across all metrics: {len(consistent_supernodes)}")

            # =========================================================
            # Part 6: Generate visualizations
            # =========================================================
            layer_suffix = layer_name.replace(".", "_")

            # Plot 1: Jaccard similarity heatmap
            try:
                fig = viz.plot_metric_similarity_heatmap(
                    similarity_matrix=jaccard_matrix,
                    metric_names=analyzed_metrics,
                    title=f"Supernode Overlap (Jaccard Similarity)\n{layer_name}",
                    save_path=robustness_dir / f"jaccard_heatmap_{layer_suffix}.png",
                )
                plt.close(fig)
            except Exception as e:
                logger.error(f"    Failed to plot Jaccard heatmap: {e}")

            # Plot 2: Spearman correlation heatmap
            try:
                fig = viz.plot_metric_similarity_heatmap(
                    similarity_matrix=spearman_matrix,
                    metric_names=analyzed_metrics,
                    title=f"Score Correlation (Spearman)\n{layer_name}",
                    save_path=robustness_dir / f"spearman_heatmap_{layer_suffix}.png",
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                )
                plt.close(fig)
            except Exception as e:
                logger.error(f"    Failed to plot Spearman heatmap: {e}")

            # Plot 3: Bootstrap stability distribution
            try:
                fig = viz.plot_supernode_stability_distribution(
                    stability_scores=stability_scores,
                    num_supernodes=num_supernodes,
                    layer_name=layer_name,
                    save_path=robustness_dir / f"bootstrap_stability_{layer_suffix}.png",
                )
                plt.close(fig)
            except Exception as e:
                logger.error(f"    Failed to plot stability distribution: {e}")

            # Plot 4: Supernode consistency across metrics (Venn-style bar chart)
            try:
                fig = viz.plot_supernode_consistency_bars(
                    metric_supernode_indices=metric_supernode_indices,
                    total_neurons=intermediate_dim,
                    layer_name=layer_name,
                    save_path=robustness_dir / f"consistency_bars_{layer_suffix}.png",
                )
                plt.close(fig)
            except Exception as e:
                logger.error(f"    Failed to plot consistency bars: {e}")

            # Plot 5: Metric score correlations scatter matrix
            try:
                if len(analyzed_metrics) >= 2:
                    fig = viz.plot_metric_score_scatter_matrix(
                        metric_scores={m: np.array(layer_results["metric_scores"][m]) for m in analyzed_metrics[:4]},
                        supernode_indices=metric_supernode_indices.get(analyzed_metrics[0], set()),
                        layer_name=layer_name,
                        save_path=robustness_dir / f"score_scatter_matrix_{layer_suffix}.png",
                    )
                    plt.close(fig)
            except Exception as e:
                logger.error(f"    Failed to plot scatter matrix: {e}")

            results[layer_name] = layer_results

        # =========================================================
        # Summary statistics across all layers
        # =========================================================
        if results:
            jaccard_means = []
            spearman_means = []
            stable_fracs = []
            for layer_result in results.values():
                n_metrics = len(layer_result.get("metrics_analyzed", []))
                if n_metrics >= 2 and "jaccard_matrix" in layer_result:
                    j_vals = np.array(layer_result["jaccard_matrix"])[np.triu_indices(n_metrics, k=1)]
                    if j_vals.size:
                        jaccard_means.append(float(np.mean(j_vals)))
                if n_metrics >= 2 and "spearman_matrix" in layer_result:
                    s_vals = np.array(layer_result["spearman_matrix"])[np.triu_indices(n_metrics, k=1)]
                    if s_vals.size:
                        spearman_means.append(float(np.mean(s_vals)))
                if "num_highly_stable" in layer_result and "num_supernodes" in layer_result and layer_result["num_supernodes"] > 0:
                    stable_fracs.append(float(layer_result["num_highly_stable"]) / float(layer_result["num_supernodes"]))

            summary = {
                "num_layers_analyzed": len(results),
                "avg_jaccard_across_metrics": float(np.mean(jaccard_means)) if jaccard_means else float("nan"),
                "avg_spearman_across_metrics": float(np.mean(spearman_means)) if spearman_means else float("nan"),
                "avg_highly_stable_fraction": float(np.mean(stable_fracs)) if stable_fracs else float("nan"),
            }
            results["summary"] = summary

            logger.info(f"\n  Summary across {len(results)-1} layers:")
            logger.info(f"    - Avg Jaccard similarity: {summary['avg_jaccard_across_metrics']:.3f}")
            logger.info(f"    - Avg Spearman correlation: {summary['avg_spearman_across_metrics']:.3f}")
            logger.info(f"    - Avg highly stable fraction: {summary['avg_highly_stable_fraction']:.1%}")

        return results

    def _compute_activation_magnitude(
        self,
        layer_name: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> Optional[torch.Tensor]:
        """Compute activation L2 norm for a specific layer."""
        device = next(self.model.parameters()).device

        # Register hook to capture activations
        activations = []
        hook_handle = None

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            # For gate_proj/up_proj output before down_proj
            activations.append(act.detach().float())

        # Find the gate_proj layer (input to the down_proj)
        gate_proj_name = layer_name.replace("down_proj", "gate_proj")
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        for name, module in hf_model.named_modules():
            if name == gate_proj_name or name.endswith(gate_proj_name):
                hook_handle = module.register_forward_hook(hook_fn)
                break

        if hook_handle is None:
            # Try up_proj instead
            up_proj_name = layer_name.replace("down_proj", "up_proj")
            for name, module in hf_model.named_modules():
                if name == up_proj_name or name.endswith(up_proj_name):
                    hook_handle = module.register_forward_hook(hook_fn)
                    break

        if hook_handle is None:
            return None

        try:
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(device)
                    self.model(**inputs)

            if activations:
                all_acts = torch.cat([a.view(-1, a.shape[-1]) for a in activations], dim=0)
                # L2 norm per neuron
                scores = torch.norm(all_acts, p=2, dim=0)
                return scores.cpu()
        finally:
            hook_handle.remove()

        return None

    def _compute_rq_for_layer(
        self,
        layer_name: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> Optional[torch.Tensor]:
        """Compute Rayleigh Quotient for a specific layer.

        Standard RQ formula: RQ(w) = (w^T Σ w) / (w^T w)
        where Σ is input covariance and w is a weight vector.

        For down_proj layers:
        - weight W: [hidden_dim, intermediate_dim]
        - input X: [batch, intermediate_dim]
        - Σ = Cov(X): [intermediate_dim, intermediate_dim]

        We compute RQ per ROW of W (each row w_i is [intermediate_dim]):
            RQ_i = (w_i @ Σ @ w_i^T) / (w_i @ w_i^T)

        Then aggregate to get per-intermediate-neuron scores by looking at
        how much each intermediate neuron j contributes across all output RQs.
        """
        device = next(self.model.parameters()).device

        # Get weight matrix
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        weight = None
        for name, module in hf_model.named_modules():
            if name == layer_name or name.endswith(layer_name):
                if hasattr(module, "weight"):
                    weight = module.weight.data.float()
                    break

        if weight is None:
            return None

        # Collect activations for covariance
        activations = []
        hook_handle = None

        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            activations.append(inp.detach().float())

        for name, module in hf_model.named_modules():
            if name == layer_name or name.endswith(layer_name):
                hook_handle = module.register_forward_hook(hook_fn)
                break

        if hook_handle is None:
            return None

        try:
            self.model.eval()
            with torch.no_grad():
                for i in range(0, min(len(texts), batch_size * 4), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                    ).to(device)
                    self.model(**inputs)

            if activations:
                all_acts = torch.cat([a.view(-1, a.shape[-1]) for a in activations], dim=0)
                all_acts = all_acts.to(device)
                input_dim = all_acts.shape[1]

                # Compute covariance of inputs: Σ = (X - μ)^T (X - μ) / (n-1)
                mean = all_acts.mean(dim=0, keepdim=True)
                centered = all_acts - mean
                cov = (centered.T @ centered) / (all_acts.shape[0] - 1)
                # cov shape: [input_dim, input_dim]

                weight = weight.to(device)
                out_dim, in_dim = weight.shape  # weight: [out_dim, in_dim]

                if "down_proj" in layer_name:
                    # weight W: [hidden_dim, intermediate_dim]
                    # cov Σ: [intermediate_dim, intermediate_dim]
                    #
                    # Standard RQ per OUTPUT neuron (row i of W):
                    #   w_i = W[i, :] has shape [intermediate_dim]
                    #   RQ_i = (w_i @ Σ @ w_i^T) / ||w_i||^2
                    #
                    # Vectorized: W @ Σ @ W^T gives [hidden_dim, hidden_dim]
                    # Diagonal gives per-output-neuron RQ

                    # Compute W @ Σ: [hidden_dim, intermediate_dim]
                    w_cov = weight @ cov  # [hidden_dim, intermediate_dim]

                    # Compute (W @ Σ) * W and sum over intermediate dim -> w^T Σ w per row
                    w_cov_w = (w_cov * weight).sum(dim=1)  # [hidden_dim]

                    # Compute ||w||^2 per row
                    w_norm_sq = (weight**2).sum(dim=1)  # [hidden_dim]

                    # RQ per output neuron
                    w_cov_w / (w_norm_sq + 1e-10)  # [hidden_dim]

                    # Now we need per-INTERMEDIATE-neuron scores for pruning.
                    # Contribution of intermediate neuron j to all output RQs:
                    # The term W[:, j] * Σ[j, :] @ W^T contributes to each output's RQ.
                    #
                    # Per-intermediate importance = how much does neuron j contribute to
                    # the total output variance? This is captured by:
                    #   Σ[j, j] * ||W[:, j]||^2  (diagonal contribution)
                    # Plus weighted covariance contribution from correlations.
                    #
                    # Alternatively, use activation variance weighted by weight magnitude:
                    # This captures supernodes (high variance + high weight = high impact)

                    # Diagonal of covariance = per-neuron variance
                    var_j = torch.diag(cov)  # [intermediate_dim]

                    # Column norms squared = weight contribution
                    col_norm_sq = (weight**2).sum(dim=0)  # [intermediate_dim]

                    # Per-intermediate RQ proxy: Var(j) * ||W[:, j]||^2
                    # This is the diagonal contribution to output variance from neuron j
                    rq_per_intermediate = var_j * col_norm_sq

                    return rq_per_intermediate.cpu()

                # For up_proj/gate_proj: weight [intermediate, hidden], input [hidden]
                # Check if weight columns align with input covariance
                elif in_dim == input_dim:
                    # Standard case: weight rows receive from input dimension
                    # RQ for each output neuron: w_i @ cov @ w_i.T / ||w_i||^2
                    w_cov = weight @ cov  # [out_dim, in_dim]
                    w_cov_w = torch.sum(w_cov * weight, dim=1)  # [out_dim]
                    w_w = torch.sum(weight**2, dim=1)  # [out_dim]
                    rq = w_cov_w / (w_w + 1e-10)  # [out_dim]
                    return rq.cpu()

                else:
                    # Dimension mismatch - use activation variance as proxy
                    logger.debug(f"RQ dimension mismatch for {layer_name}: weight {weight.shape}, cov {cov.shape}")
                    input_var = torch.var(all_acts, dim=0)  # [input_dim]
                    return input_var.cpu()

        finally:
            hook_handle.remove()

        return None

    def _compute_mi_for_layer(
        self,
        layer_name: str,
        texts: List[str],
        batch_size: int = 32,
    ) -> Optional[torch.Tensor]:
        """Compute Gaussian MI for a specific layer."""
        # Similar to RQ but with MI formula
        rq_scores = self._compute_rq_for_layer(layer_name, texts, batch_size)
        if rq_scores is not None:
            # MI = 0.5 * log(1 + SNR), where SNR ~ RQ / noise_var
            # Use a fixed noise variance estimate
            noise_var = 0.1
            snr = rq_scores / (noise_var + 1e-10)
            mi = 0.5 * torch.log1p(snr.clamp(min=0))
            return mi
        return None

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

        logger.info(f"  Computing metrics for {len(follower_indices)} high-connection positions " f"(inputs to layer {next_layer_idx})...")

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
        # Approximated using correlation: MI ~ -0.5 * log(1 - r^2)
        # =====================================================================
        mi_scores = torch.zeros(num_followers)

        # Compute variance of each follower
        torch.var(all_acts, dim=0)

        # For MI, we compute how much each neuron's variance is explained by others
        # Using the average squared correlation as a proxy
        for i in range(num_followers):
            # Get correlations of neuron i with all others
            corr_with_others = corr_matrix[i, :].clone()
            corr_with_others[i] = 0  # Exclude self

            # Average squared correlation (R^2)
            r_squared = (corr_with_others**2).mean()

            # MI approximation: higher R^2 means more information shared
            # MI = -0.5 * log(1 - R^2) for Gaussian
            mi_scores[i] = -0.5 * torch.log(1 - r_squared.clamp(max=0.999) + 1e-8)

        # =====================================================================
        # Plot results
        # =====================================================================

        # Use UnifiedVisualizer for all plots
        viz = UnifiedVisualizer()
        layer_suffix = current_layer_name.replace(".", "_")
        import matplotlib.pyplot as plt

        # Create layer_analysis directory
        layer_analysis_dir = plots_dir / "layer_analysis"
        layer_analysis_dir.mkdir(parents=True, exist_ok=True)

        # Create descriptive title prefix
        title_prefix = f"High-Connection Neurons (Layer {next_layer_idx} input)"

        # Plot correlation matrix
        try:
            fig = viz.plot_correlation_matrix(
                corr_matrix=corr_matrix,
                title=f"{title_prefix}\nPairwise Correlations (Mean |r|={mean_redundancy:.3f})",
                xlabel="Neuron Index (high supernode connection)",
                ylabel="Neuron Index (high supernode connection)",
                save_path=layer_analysis_dir / f"next_layer_correlation_{layer_suffix}.png",
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
                save_path=layer_analysis_dir / f"next_layer_redundancy_hist_{layer_suffix}.png",
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
                color="green",
                save_path=layer_analysis_dir / f"next_layer_rq_hist_{layer_suffix}.png",
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
                color="purple",
                save_path=layer_analysis_dir / f"next_layer_mi_hist_{layer_suffix}.png",
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
                save_path=layer_analysis_dir / f"next_layer_rq_vs_mi_{layer_suffix}.png",
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

        logger.info("    Metrics for high-connection neurons (next layer input):")
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
        logger.info("  Comparing redundancy: high vs low supernode-connected neurons...")

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

        logger.info(
            f"    High-connected group: {num_group} neurons, influence range [{high_influence_values[-1]:.4f}, {high_influence_values[0]:.4f}]"
        )
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
        low_acts = torch.cat(low_activations, dim=0)  # [total_tokens, num_group]

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

        # Create redundancy subfolder for organized plots
        redundancy_dir = plots_dir / "redundancy"
        redundancy_dir.mkdir(parents=True, exist_ok=True)

        # Create layer_analysis subfolder for scatter and other analysis plots
        layer_analysis_dir = plots_dir / "layer_analysis"
        layer_analysis_dir.mkdir(parents=True, exist_ok=True)

        # Plots 1-3: Redundancy comparison (side-by-side, overlay, boxplot)
        try:
            figs = viz.plot_redundancy_comparison(
                high_redundancy=high_pairwise,
                low_redundancy=low_pairwise,
                high_mean=high_mean_redundancy,
                low_mean=low_mean_redundancy,
                layer_name=layer_name,
                follower_fraction=follower_fraction,
                save_dir=redundancy_dir,
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
            all_labels = ["High"] * len(high_influence_values) + ["Low"] * len(low_influence_values)

            fig = viz.plot_metric_scatter_by_group(
                x_values=all_influence,
                y_values=all_redundancy,
                group_labels=all_labels,
                xlabel="Supernode Influence (Total Abs Weight)",
                ylabel="Mean Redundancy (Avg |Correlation| with Group)",
                title=f"Supernode Influence vs Redundancy per Neuron\n{layer_name}",
                save_path=layer_analysis_dir / f"redundancy_vs_influence_scatter_{layer_name.replace('.', '_')}.png",
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

    def compute_halo_redundancy_within_hidden_outputs(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        halo_fraction: float = 0.10,
        num_samples: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """
        (Legacy/diagnostic) Compute redundancy among *hidden-dimension* output neurons that are strongly
        influenced by supernodes.

        Note: This is NOT the SCAR definition of "directed redundancy" (which is defined on loss-relevant
        per-channel contribution signals). This helper is kept for exploratory plots and is not used
        for pruning decisions.

        It:
        1. Identifies intermediate-dim supernodes (top by loss proxy / activation power)
        2. Defines a "halo" in the *hidden* output space: hidden neurons receiving large total weight from supernodes
        3. Computes within-halo redundancy using activation correlation

        Args:
            scar_scores: SCAR metrics per layer from compute_scar_supernode_metrics
            supernode_fraction: Fraction to identify as supernodes (top by loss proxy)
            halo_fraction: Fraction of neurons to consider as "halo" around supernodes
            num_samples: Number of calibration samples to use

        Returns:
            Dict with per-layer results including:
            - halo_indices: Indices of neurons in the halo
            - halo_redundancy: Per-neuron redundancy score within the halo
            - protection_score: 1 - normalized_redundancy (high = protect, low = prune)
            - pruning_candidates: Indices of highly redundant halo neurons
        """
        logger.info("=" * 60)
        logger.info("Computing DIRECTED REDUNDANCY (supernode -> halo)")
        logger.info("=" * 60)
        logger.info(f"  Supernode fraction: {supernode_fraction*100:.1f}%")
        logger.info(f"  Halo fraction: {halo_fraction*100:.1f}%")

        results = {}

        # Get model
        hf_model = self.wrapped_model._model if hasattr(self.wrapped_model, "_model") else self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        # Get calibration texts
        calibration_texts = []
        if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
            calibration_texts = list(self.dataset.texts)[:num_samples]
        if not calibration_texts:
            logger.warning("No calibration texts available for directed redundancy")
            return {}

        # Process each layer with SCAR scores
        for layer_name, layer_scores in scar_scores.items():
            if "down_proj" not in layer_name:
                continue

            logger.info(f"\nProcessing {layer_name}...")

            # Get loss proxy scores to identify supernodes
            loss_proxy = layer_scores.get("scar_loss_proxy")
            if loss_proxy is None:
                continue
            loss_proxy = loss_proxy.float().cpu()

            # Get down_proj weights to find connections
            down_proj_weight = None
            for name, module in hf_model.named_modules():
                if name.endswith(layer_name.replace("model.model.", "model.")):
                    if hasattr(module, "weight"):
                        down_proj_weight = module.weight.data.float().cpu()
                        break

            if down_proj_weight is None:
                # Try alternative naming
                layer_pattern = layer_name.replace("model.model.", "")
                for name, module in hf_model.named_modules():
                    if layer_pattern in name or name.endswith(layer_pattern):
                        if hasattr(module, "weight"):
                            down_proj_weight = module.weight.data.float().cpu()
                            break

            if down_proj_weight is None:
                logger.warning(f"  Could not find weights for {layer_name}")
                continue

            intermediate_dim = loss_proxy.numel()  # e.g., 14336
            hidden_dim = down_proj_weight.shape[0]  # e.g., 4096

            # Step 1: Identify supernodes (top by loss proxy)
            num_supernodes = max(1, int(supernode_fraction * intermediate_dim))
            _, supernode_indices = torch.topk(loss_proxy, num_supernodes)
            supernode_indices = supernode_indices.numpy()

            logger.info(f"  Identified {num_supernodes} supernodes")

            # Step 2: Find "halo" - neurons with large weights TO supernodes
            # down_proj shape: [hidden_dim, intermediate_dim]
            # Each row i is the weights from all intermediate neurons to output i
            # Supernode columns are the weights FROM supernodes

            # Sum of absolute weights FROM supernodes for each output neuron
            supernode_weights = down_proj_weight[:, supernode_indices]  # [hidden_dim, num_supernodes]
            connection_strength = torch.abs(supernode_weights).sum(dim=1)  # [hidden_dim]

            # Top halo_fraction of neurons by connection to supernodes
            num_halo = max(1, int(halo_fraction * hidden_dim))
            _, halo_indices = torch.topk(connection_strength, num_halo)
            halo_indices = halo_indices.numpy()

            logger.info(f"  Identified {num_halo} halo neurons (connected to supernodes)")

            # Step 3: Capture activations for halo neurons
            halo_activations = []

            def capture_hook(module, inputs, outputs):
                if outputs is not None:
                    out = outputs.detach().float()
                    if out.ndim == 3:
                        out = out.reshape(-1, out.shape[-1])
                    halo_activations.append(out[:, halo_indices].cpu())

            # Find and hook down_proj
            hook_handle = None
            for name, module in hf_model.named_modules():
                if "mlp.down_proj" in name and any(p in layer_name for p in [name, name.split(".")[-3]]):
                    hook_handle = module.register_forward_hook(capture_hook)
                    break

            if hook_handle is None:
                # Try more flexible matching
                layer_idx = None
                if "layers." in layer_name:
                    try:
                        parts = layer_name.split("layers.")
                        if len(parts) > 1:
                            layer_idx = int(parts[1].split(".")[0])
                    except:
                        pass

                if layer_idx is not None:
                    for name, module in hf_model.named_modules():
                        if f"layers.{layer_idx}.mlp.down_proj" in name:
                            hook_handle = module.register_forward_hook(capture_hook)
                            break

            if hook_handle is None:
                logger.warning(f"  Could not hook module for {layer_name}")
                continue

            # Run forward passes
            self.wrapped_model._model.eval()
            with torch.no_grad():
                for text in calibration_texts:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=False,
                    )
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    try:
                        self.wrapped_model._model(**inputs)
                    except Exception:
                        pass

            hook_handle.remove()

            if not halo_activations:
                logger.warning(f"  No activations captured for {layer_name}")
                continue

            # Step 4: Compute pairwise redundancy WITHIN the halo
            all_acts = torch.cat(halo_activations, dim=0)  # [total_tokens, num_halo]

            # Correlation matrix
            acts_centered = all_acts - all_acts.mean(dim=0, keepdim=True)
            acts_std = acts_centered.std(dim=0, keepdim=True)
            acts_std = torch.where(acts_std > 1e-8, acts_std, torch.ones_like(acts_std))
            acts_norm = acts_centered / acts_std

            corr_matrix = (acts_norm.T @ acts_norm) / (all_acts.shape[0] - 1)
            corr_matrix = torch.clamp(corr_matrix, -1, 1)

            # Redundancy: average |correlation| with other halo neurons (excluding self)
            abs_corr = torch.abs(corr_matrix)
            abs_corr.fill_diagonal_(0)  # Exclude self
            halo_redundancy = abs_corr.mean(dim=1)  # [num_halo]

            # Gaussian MI approximation: MI = -0.5 * log(1 - rho^2)
            rho_sq = corr_matrix**2
            rho_sq = torch.clamp(rho_sq, 0, 0.9999)
            rho_sq.fill_diagonal_(0)
            mi_matrix = -0.5 * torch.log(1 - rho_sq)
            halo_redundancy_mi = mi_matrix.mean(dim=1)  # [num_halo]

            logger.info(f"  Halo redundancy: mean={halo_redundancy.mean():.4f}, std={halo_redundancy.std():.4f}")

            # Step 5: Compute protection score (low redundancy = unique = protect)
            # Normalize redundancy to [0, 1]
            red_min, red_max = halo_redundancy.min(), halo_redundancy.max()
            if red_max > red_min:
                normalized_redundancy = (halo_redundancy - red_min) / (red_max - red_min)
            else:
                normalized_redundancy = torch.zeros_like(halo_redundancy)

            protection_score = 1.0 - normalized_redundancy  # High = protect, low = prune

            # Step 6: Identify pruning candidates (high redundancy within halo)
            # Bottom 50% of protection score = top 50% redundant
            prune_threshold = protection_score.median()
            pruning_candidates = halo_indices[protection_score.numpy() < prune_threshold.item()]

            logger.info(f"  Pruning candidates (high redundancy): {len(pruning_candidates)} neurons")

            # Store results
            results[layer_name] = {
                "supernode_indices": supernode_indices.tolist(),
                "num_supernodes": num_supernodes,
                "halo_indices": halo_indices.tolist(),
                "num_halo": num_halo,
                "halo_redundancy": halo_redundancy.numpy().tolist(),
                "halo_redundancy_mi": halo_redundancy_mi.numpy().tolist(),
                "protection_score": protection_score.numpy().tolist(),
                "pruning_candidates": pruning_candidates.tolist(),
                "num_pruning_candidates": len(pruning_candidates),
                "stats": {
                    "redundancy_mean": float(halo_redundancy.mean()),
                    "redundancy_std": float(halo_redundancy.std()),
                    "protection_mean": float(protection_score.mean()),
                    "protection_std": float(protection_score.std()),
                },
            }

            # Store in importance_scores for use in pruning
            if layer_name in self.importance_scores:
                layer_scores = self.importance_scores[layer_name]

                # Create full-size tensors
                full_halo_mask = torch.zeros(hidden_dim, dtype=torch.bool)
                full_halo_mask[halo_indices] = True

                # Build protection score with base from L2 or RQ
                full_protection = torch.zeros(hidden_dim)
                if "activation_l2_norm" in layer_scores:
                    base = layer_scores["activation_l2_norm"].detach().clone()
                    if base.max() > base.min():
                        full_protection = (base - base.min()) / (base.max() - base.min())

                # Halo neurons get boosted protection based on uniqueness
                max_base = full_protection.max().item() if full_protection.max() > 0 else 1.0
                full_protection[halo_indices] = full_protection[halo_indices] + max_base * 5 + protection_score

                full_prune_candidates = torch.zeros(hidden_dim, dtype=torch.bool)
                full_prune_candidates[pruning_candidates] = True

                layer_scores["halo_mask"] = full_halo_mask
                layer_scores["supernode_protection_score"] = full_protection
                layer_scores["prune_candidate_mask"] = full_prune_candidates

        logger.info("\nDirected redundancy computation complete!")
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

    def compute_directed_redundancy(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        num_samples: int = 8,
        max_length: int = 256,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute directed redundancy from supernodes to downstream neurons.

        Directed redundancy measures how much each supernode's activation explains
        variance in downstream neurons. Unlike symmetric correlation, this captures
        the causal/directional flow of information through the network weights.

        For each supernode i and downstream neuron j:
            DirectedRedundancy(i->j) = |weight_ij| × R²(activation_i -> activation_j)

        Where R² is the coefficient of determination (variance explained).

        This is important for pruning because:
        1. High directed redundancy means the supernode strongly controls downstream neurons
        2. Pruning such supernodes would significantly disrupt downstream computation
        3. Low directed redundancy supernodes may be safer to prune

        Args:
            scar_scores: SCAR metrics per layer (from compute_scar_supernode_metrics)
            supernode_fraction: Fraction of neurons to consider as supernodes
            num_samples: Number of calibration samples for activation capture
            max_length: Maximum sequence length for calibration

        Returns:
            Dictionary with directed redundancy metrics per layer
        """
        logger.info("Computing directed redundancy from supernodes to downstream neurons...")
        logger.info(f"  - Supernode fraction: top {supernode_fraction*100:.1f}%")

        results: Dict[str, Dict[str, Any]] = {}

        # Get underlying HF model
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        # Get calibration texts
        calibration_texts: List[str] = []
        if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
            calibration_texts = list(self.dataset.texts)[:num_samples]

        if not calibration_texts:
            logger.warning("No calibration texts available for directed redundancy computation")
            return {}

        for layer_name, layer_metrics in scar_scores.items():
            if "mlp.down_proj" not in layer_name:
                continue

            # Get supernode scores (use activation power by default)
            supernode_scores = layer_metrics.get("scar_activation_power")
            if supernode_scores is None:
                continue

            supernode_scores = supernode_scores.float().cpu()
            num_neurons = supernode_scores.numel()  # intermediate_dim (e.g., 14336)

            # Identify supernodes (top neurons by activation power)
            num_supernodes = max(1, int(supernode_fraction * num_neurons))
            _, sorted_indices = torch.sort(supernode_scores, descending=True)
            supernode_indices = sorted_indices[:num_supernodes].numpy()

            logger.info(f"  {layer_name}: {num_supernodes} supernodes")

            # Get down_proj weights: [hidden_dim, intermediate_dim]
            down_proj_weight = None
            for name, module in hf_model.named_modules():
                if "mlp.down_proj" in name:
                    if name == layer_name or name.endswith(layer_name):
                        if hasattr(module, "weight"):
                            down_proj_weight = module.weight.detach().float().cpu()
                            break

            if down_proj_weight is None:
                logger.warning(f"  Could not find weights for {layer_name}")
                continue

            hidden_dim, intermediate_dim = down_proj_weight.shape

            # Capture intermediate activations (input to down_proj)
            intermediate_activations: List[torch.Tensor] = []
            output_activations: List[torch.Tensor] = []

            def capture_hook(module, inputs, outputs):
                if inputs and inputs[0] is not None:
                    inp = inputs[0].detach().float()
                    if inp.ndim == 3:
                        inp = inp.reshape(-1, inp.shape[-1])
                    intermediate_activations.append(inp.cpu())
                if outputs is not None:
                    out = outputs.detach().float()
                    if out.ndim == 3:
                        out = out.reshape(-1, out.shape[-1])
                    output_activations.append(out.cpu())

            # Find and hook the down_proj module
            hook_handle = None
            for name, module in hf_model.named_modules():
                if name == layer_name or (name.endswith(layer_name) and "mlp.down_proj" in name):
                    hook_handle = module.register_forward_hook(capture_hook)
                    break

            if hook_handle is None:
                continue

            # Run forward passes to capture activations
            self.model.eval()
            with torch.no_grad():
                for text in calibration_texts:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=False,
                    )
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    try:
                        self.model(**inputs)
                    except Exception:
                        pass

            hook_handle.remove()

            if not intermediate_activations or not output_activations:
                continue

            # Concatenate activations
            all_intermediate = torch.cat(intermediate_activations, dim=0)  # [N, intermediate_dim]
            all_output = torch.cat(output_activations, dim=0)  # [N, hidden_dim]

            N = all_intermediate.shape[0]

            # Extract supernode activations
            supernode_acts = all_intermediate[:, supernode_indices]  # [N, num_supernodes]

            # =====================================================================
            # Compute Directed Redundancy: R²(supernode_i -> output_j) × |weight_ij|
            # =====================================================================

            # For efficiency, compute correlations in batch
            # Center the activations
            supernode_centered = supernode_acts - supernode_acts.mean(dim=0, keepdim=True)
            output_centered = all_output - all_output.mean(dim=0, keepdim=True)

            # Compute variance of each supernode
            supernode_var = (supernode_centered**2).sum(dim=0) / (N - 1)  # [num_supernodes]
            output_var = (output_centered**2).sum(dim=0) / (N - 1)  # [hidden_dim]

            # Compute covariance between supernodes and outputs
            # cov_matrix[i, j] = cov(supernode_i, output_j)
            cov_matrix = (supernode_centered.T @ output_centered) / (N - 1)  # [num_supernodes, hidden_dim]

            # Compute R² (coefficient of determination)
            # R²(i->j) = cov(i,j)² / (var(i) × var(j))
            denom = supernode_var.unsqueeze(1) * output_var.unsqueeze(0) + 1e-8
            r_squared = (cov_matrix**2) / denom  # [num_supernodes, hidden_dim]

            # Get weight magnitudes from supernodes to all outputs
            weight_magnitudes = torch.abs(down_proj_weight[:, supernode_indices].T)  # [num_supernodes, hidden_dim]

            # Directed redundancy: R² × |weight|
            directed_redundancy = r_squared * weight_magnitudes  # [num_supernodes, hidden_dim]

            # Aggregate metrics
            # Total influence of each supernode (sum over all outputs)
            supernode_total_influence = directed_redundancy.sum(dim=1)  # [num_supernodes]

            # Mean directed redundancy per supernode
            supernode_mean_dr = directed_redundancy.mean(dim=1)  # [num_supernodes]

            # Max directed redundancy per supernode (strongest downstream connection)
            supernode_max_dr = directed_redundancy.max(dim=1).values  # [num_supernodes]

            # Store as importance scores for potential use in pruning
            # Lower directed redundancy = safer to prune
            layer_scores = self.importance_scores.get(layer_name, {})

            # Create full-size score tensor (zeros for non-supernodes)
            full_directed_redundancy = torch.zeros(intermediate_dim)
            full_directed_redundancy[supernode_indices] = supernode_total_influence
            layer_scores["directed_redundancy"] = full_directed_redundancy

            # Diagnostic score (NOT used as SCAR-Prot): downstream influence of supernodes.
            # This measures which *supernodes* strongly explain downstream hidden activations.
            # We keep it under a separate key to avoid clobbering the SCAR-Prot pruning score.
            base_protection = torch.zeros(intermediate_dim)

            # Get base importance from L2 norm, RQ, or scar_loss_proxy if available
            # First check self.importance_scores, then scar_scores
            base_metric = None
            for metric_name in ["activation_l2_norm", "rayleigh_quotient", "scar_loss_proxy", "scar_activation_power"]:
                if metric_name in layer_scores:
                    base_metric = layer_scores[metric_name]
                    break
                elif metric_name in layer_metrics:
                    base_metric = layer_metrics[metric_name]
                    break

            if base_metric is not None:
                base_protection = base_metric.float().cpu().detach().clone()
                # Normalize to [0, 1]
                if base_protection.max() > base_protection.min():
                    base_protection = (base_protection - base_protection.min()) / (base_protection.max() - base_protection.min())

            # Supernodes get very high protection (10x boost above max base)
            max_base = base_protection.max().item() if base_protection.max() > 0 else 1.0
            downstream_influence_scores = base_protection.clone()
            downstream_influence_scores[supernode_indices] = max_base * 10 + supernode_total_influence
            layer_scores["supernode_downstream_influence_score"] = downstream_influence_scores

            self.importance_scores[layer_name] = layer_scores

            results[layer_name] = {
                "num_supernodes": num_supernodes,
                "supernode_indices": supernode_indices.tolist(),
                "directed_redundancy": {
                    "total_per_supernode": supernode_total_influence.numpy().tolist(),
                    "mean_per_supernode": supernode_mean_dr.numpy().tolist(),
                    "max_per_supernode": supernode_max_dr.numpy().tolist(),
                    "overall_mean": float(supernode_mean_dr.mean().item()),
                    "overall_std": float(supernode_mean_dr.std().item()),
                },
                "statistics": {
                    "mean_r_squared": float(r_squared.mean().item()),
                    "mean_weight_magnitude": float(weight_magnitudes.mean().item()),
                    "mean_directed_redundancy": float(directed_redundancy.mean().item()),
                },
            }

            logger.info(f"    Mean directed redundancy: {directed_redundancy.mean().item():.6f}")
            logger.info(f"    Top supernode total influence: {supernode_total_influence.max().item():.4f}")

        logger.info(f"Computed directed redundancy for {len(results)} layers")
        return results

    def compute_supernode_connectivity_pruning_score(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        high_connectivity_fraction: float = 0.10,
        redundancy_weight: float = 0.5,
        num_samples: int = 8,
        plots_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute SCAR-style halo-aware pruning scores.

        This routine computes, per FFN channel i in each layer:
        - **Supernodes**: top `supernode_fraction` by `scar_loss_proxy`
        - **Connectivity** Conn_i: overlap of downstream write pattern |v_i| with the aggregated
          supernode write pattern a = Σ_{s in supernodes} |v_s|
        - **Halo**: top `high_connectivity_fraction` of non-supernodes by Conn_i
        - **Loss-relevant redundancy to core** (halo only): using the scalar contribution
          q_i = u_i * (v_i^T g_y), compute Gaussian MI to each supernode and aggregate
          via a Top-k mean (default k=5; reduces max-inflation / multiple-comparisons effects)
        - **Protection** Protect_i in [0, 1] (halo only): 1 - normalized(redundancy_to_core)

        It then produces two **importance scores** (high = keep; prune with mode="low"):
        - `supernode_protection_score` (SCAR-Prot): LP_i * Protect_i (non-halo Protect=1)
        - `supernode_connectivity_score` (SCAR-Conn): LP_i * ((1-Conn_i) + Conn_i * Protect_i)

        Notes:
        - `redundancy_weight` is retained for backward compatibility but not used in the
          default estimator (MI already yields a redundancy scale).

        Args:
            scar_scores: SCAR scores dictionary with supernode metrics
            supernode_fraction: Fraction of neurons considered supernodes
            high_connectivity_fraction: Halo fraction (fraction of non-supernodes placed in halo)
            redundancy_weight: (unused) kept for backward compatibility
            num_samples: Calibration samples for redundancy / protection computation
            plots_dir: Directory to save analysis plots

        Returns:
            Dictionary with pruning scores and analysis per layer
        """
        logger.info("Computing SCAR halo connectivity + protection pruning scores...")
        logger.info(f"  Supernode fraction (rho): {supernode_fraction*100:.1f}%")
        logger.info(f"  Halo fraction (eta): {high_connectivity_fraction*100:.1f}%")

        eps = 1e-8
        results: Dict[str, Dict[str, Any]] = {}
        supernode_cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
        # Default to positive-only redundancy (anti-correlation does NOT count as redundancy),
        # matching the default definition; can be disabled for sensitivity analyses.
        positive_redundancy = bool(supernode_cfg.get("positive_redundancy", True))
        if positive_redundancy:
            logger.info("  Redundancy: using positive-only correlation (anti-correlation does NOT count as redundancy)")

        # Optional: cross-layer read-halo pruning modifier (analysis/ablation; disabled by default).
        # This does not change SCAR unless explicitly enabled and selected as a pruning strategy.
        read_halo_prune_cfg = supernode_cfg.get("read_halo_pruning", {}) or supernode_cfg.get("read_halo_prune", {}) or {}
        read_halo_prune_enabled = bool(read_halo_prune_cfg.get("enabled", False)) if isinstance(read_halo_prune_cfg, dict) else False
        if read_halo_prune_enabled:
            try:
                _rh_frac = float(read_halo_prune_cfg.get("read_halo_fraction", read_halo_prune_cfg.get("fraction", 0.10)))
            except Exception:
                _rh_frac = 0.10
            _rh_frac = float(min(1.0, max(0.0, _rh_frac)))
            try:
                _rh_gamma = float(read_halo_prune_cfg.get("rank_power", read_halo_prune_cfg.get("protection_rank_power", 8.0)))
            except Exception:
                _rh_gamma = 8.0
            if not (_rh_gamma > 0):
                _rh_gamma = 8.0
            try:
                _rh_floor = float(read_halo_prune_cfg.get("protection_floor", 0.2))
            except Exception:
                _rh_floor = 0.2
            _rh_floor = float(min(1.0, max(0.0, _rh_floor)))
            logger.info(f"  Read-halo pruning: enabled (fraction={_rh_frac*100:.1f}%, rank_power={_rh_gamma:g}, floor={_rh_floor:g})")
        else:
            _rh_frac = 0.10
            _rh_gamma = 8.0
            _rh_floor = 0.2

        # Underlying HF model for module lookup / hook registration
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model

        module_dict = dict(hf_model.named_modules())

        # Calibration texts
        calibration_texts: List[str] = []
        if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
            calibration_texts = list(self.dataset.texts)[:num_samples]
        if not calibration_texts:
            logger.warning("No calibration texts available for SCAR protection/connectivity computation")
            return {}

        # Determine which layers to process (down_proj layers only)
        layer_names = [ln for ln in scar_scores.keys() if "mlp.down_proj" in ln]
        if not layer_names:
            logger.warning("No down_proj layers found in scar_scores; skipping SCAR connectivity/pruning score")
            return {}

        # ------------------------------------------------------------------
        # Phase 1: Per-layer supernodes + connectivity + halo indices (weights-only)
        # ------------------------------------------------------------------
        plan: Dict[str, Dict[str, Any]] = {}
        for layer_name in layer_names:
            layer_metrics = scar_scores.get(layer_name, {}) or {}
            lp = layer_metrics.get("scar_loss_proxy")
            if lp is None:
                # fallbacks for older runs
                lp = layer_metrics.get("scar_activation_power")
            if lp is None:
                continue

            lp_cpu = lp.detach().float().cpu()
            m = lp_cpu.numel()
            if m == 0:
                continue

            module = module_dict.get(layer_name)
            if module is None or not hasattr(module, "weight"):
                logger.warning(f"SCAR connectivity: could not resolve module/weight for {layer_name}")
                continue

            # Identify supernodes by LP
            num_supernodes = max(1, int(supernode_fraction * m))
            _, super_idx = torch.topk(lp_cpu, k=num_supernodes, largest=True)
            super_idx = super_idx.long()
            super_mask = torch.zeros(m, dtype=torch.bool)
            super_mask[super_idx] = True

            # Compute Conn_i from down_proj weights.
            #
            # IMPORTANT: the classic "probability overlap" Conn
            #   <|v_i|, a> / (||v_i||_1 ||a||_1)
            # tends to collapse to ~1/hidden_dim for dense matrices (~ 2.4e-4 for d=4096),
            # which makes SCAR-Conn numerically ineffective. Instead, we measure the fraction
            # of each channel's write mass that falls on the *core write support*:
            # the top-K hidden dimensions by aggregated supernode write mass a.
            #
            # Conn_i := sum_{h in TopK(a)} |v_i[h]| / ||v_i||_1  in [0, 1]
            W = module.weight.detach().float().cpu()  # [hidden_dim, m]
            abs_W = W.abs()
            a = abs_W[:, super_idx].sum(dim=1)  # [hidden_dim]
            v_norm = abs_W.sum(dim=0) + eps  # [m]

            hidden_dim = int(abs_W.shape[0])
            k = int(supernode_cfg.get("connectivity_topk", 256))
            mass_frac = supernode_cfg.get("connectivity_mass_fraction", None)
            a_sorted, a_order = torch.sort(a, descending=True)
            if mass_frac is not None:
                try:
                    mf = float(mass_frac)
                except Exception:
                    mf = None
                if mf is not None and 0.0 < mf < 1.0 and a_sorted.numel() > 0:
                    cdf = torch.cumsum(a_sorted, dim=0)
                    total = float(cdf[-1].item())
                    if total > 0:
                        target = mf * total
                        k = int(torch.searchsorted(cdf, torch.tensor(target)).item()) + 1
            k = max(1, min(int(k), hidden_dim))
            core_idx = a_order[:k]
            conn = abs_W.index_select(0, core_idx).sum(dim=0) / v_norm
            conn = conn.clamp(0.0, 1.0)

            # Optional post-processing to give Conn more dynamic range when needed.
            #
            # - rank-normalize Conn among non-supernodes (maps to [0,1] by empirical CDF)
            # - apply a power transform (power < 1 increases small Conn values; power > 1 shrinks them)
            if bool(supernode_cfg.get("connectivity_rank_normalize", False)):
                non_super_idx_for_rank = (~super_mask).nonzero(as_tuple=True)[0]
                if non_super_idx_for_rank.numel() > 1:
                    vals = conn[non_super_idx_for_rank]
                    _, order = torch.sort(vals, stable=True)  # ascending
                    ranks = torch.empty_like(order, dtype=torch.float32)
                    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
                    ranks = ranks / float(max(1, order.numel() - 1))
                    conn_rank = conn.clone()
                    conn_rank[non_super_idx_for_rank] = ranks
                    conn_rank[super_idx] = 1.0
                    conn = conn_rank

            conn_power = supernode_cfg.get("connectivity_power", 1.0)
            try:
                conn_power_f = float(conn_power)
            except Exception:
                conn_power_f = 1.0
            if conn_power_f != 1.0:
                conn = conn.clamp(0.0, 1.0).pow(conn_power_f).clamp(0.0, 1.0)

            # Halo: top eta among non-supernodes by Conn
            non_super_idx = (~super_mask).nonzero(as_tuple=True)[0]
            if non_super_idx.numel() == 0:
                continue
            num_halo = max(1, int(high_connectivity_fraction * non_super_idx.numel()))
            halo_scores = conn[non_super_idx]
            _, halo_rel = torch.topk(halo_scores, k=num_halo, largest=True)
            halo_idx = non_super_idx[halo_rel].long()

            # Extract layer index once (used for deterministic sampling seeds)
            try:
                layer_idx_int = int(layer_name.split("layers.")[-1].split(".")[0])
            except Exception:
                layer_idx_int = 0

            # Optional: sample a subset of *non-halo* channels for redundancy-to-core analysis.
            # This lets us explicitly compare halo-to-core redundancy vs non-halo-to-core redundancy
            # without the prohibitive cost of computing redundancy for *all* non-halo channels.
            non_halo_sample_size = int(supernode_cfg.get("non_halo_sample_size", 256) or 0)
            non_halo_idx = torch.empty((0,), dtype=torch.long)
            rand_core_idx = torch.empty((0,), dtype=torch.long)
            compute_random_core = bool(supernode_cfg.get("compute_random_core_baseline", True))
            if non_halo_sample_size > 0 or compute_random_core:
                halo_mask_tmp = torch.zeros(m, dtype=torch.bool)
                halo_mask_tmp[halo_idx] = True
                non_halo_all = (~super_mask & ~halo_mask_tmp).nonzero(as_tuple=True)[0]
                if non_halo_all.numel() > 0:
                    if non_halo_sample_size > 0:
                        sample_n = min(non_halo_sample_size, int(non_halo_all.numel()))
                        seed_base = int(supernode_cfg.get("non_halo_sample_seed", 0) or 0)
                        g = torch.Generator()
                        g.manual_seed(seed_base + layer_idx_int)
                        perm = torch.randperm(int(non_halo_all.numel()), generator=g)
                        non_halo_idx = non_halo_all[perm[:sample_n]].long()

                    # Random-core baseline (multiple-comparisons control): pick a random set
                    # of the same size as the supernode core from the non-halo pool.
                    if compute_random_core:
                        rand_n = min(int(num_supernodes), int(non_halo_all.numel()))
                        seed_base_rand = int(supernode_cfg.get("random_core_seed", 12345) or 0)
                        g2 = torch.Generator()
                        g2.manual_seed(seed_base_rand + layer_idx_int)
                        perm2 = torch.randperm(int(non_halo_all.numel()), generator=g2)
                        rand_core_idx = non_halo_all[perm2[:rand_n]].long()

            plan[layer_name] = {
                "lp_cpu": lp_cpu,
                "conn_cpu": conn,
                "super_idx_cpu": super_idx,
                "halo_idx_cpu": halo_idx,
                "non_halo_idx_cpu": non_halo_idx,
                "rand_core_idx_cpu": rand_core_idx,
                # Layer index + core hidden support (used by optional read-halo pruning diagnostics)
                "layer_idx_int": layer_idx_int,
                "core_hidden_idx_cpu": core_idx.long(),
                "m": m,
                # device-side indices + streaming sums (initialized lazily in hooks)
                "super_idx": None,
                "halo_idx": None,
                "non_halo_idx": None,
                "rand_core_idx": None,
                "sum_q_super": None,
                "sum_q2_super": None,
                "sum_q3_super": None,
                "sum_q4_super": None,
                "sum_q_halo": None,
                "sum_q2_halo": None,
                "sum_q3_halo": None,
                "sum_q4_halo": None,
                "sum_q_halo_super": None,
                "sum_q_non_halo": None,
                "sum_q2_non_halo": None,
                "sum_q3_non_halo": None,
                "sum_q4_non_halo": None,
                "sum_q_non_halo_super": None,
                "sum_q_rand": None,
                "sum_q2_rand": None,
                "sum_q_halo_rand": None,
                "sum_q_non_halo_rand": None,
                "count": 0,
            }

        if not plan:
            logger.warning("SCAR connectivity: no layers eligible after filtering; skipping")
            return {}

        # Map plans by transformer block index (for optional cross-layer read-halo modifier).
        plan_by_layer_idx: Dict[int, Dict[str, Any]] = {}
        for _ln, _st in plan.items():
            try:
                li = int(_st.get("layer_idx_int", 0) or 0)
            except Exception:
                li = 0
            plan_by_layer_idx[li] = _st

        # ------------------------------------------------------------------
        # Phase 2: Calibration passes to estimate redundancy-to-core via q=u*(v^T g_y)
        # ------------------------------------------------------------------
        hooks: List[Any] = []

        def make_hooks(name: str):
            def fwd_hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
                if not inputs or inputs[0] is None:
                    return
                mod._scar_conn_last_u = inputs[0].detach()

            def bwd_hook(mod: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
                st = plan.get(name)
                if st is None:
                    return
                if not grad_input or grad_input[0] is None:
                    return
                if not hasattr(mod, "_scar_conn_last_u"):
                    return

                u = mod._scar_conn_last_u
                delattr(mod, "_scar_conn_last_u")

                g_u = grad_input[0]

                # Flatten to [N_tokens, dim]
                if u.ndim > 2:
                    u_flat = u.reshape(-1, u.shape[-1])
                else:
                    u_flat = u.reshape(-1, u.shape[-1])
                if g_u.ndim > 2:
                    g_u_flat = g_u.reshape(-1, g_u.shape[-1])
                else:
                    g_u_flat = g_u.reshape(-1, g_u.shape[-1])

                if u_flat.numel() == 0:
                    return

                # Move indices to the correct device once
                if st["super_idx"] is None or st["super_idx"].device != u_flat.device:
                    st["super_idx"] = st["super_idx_cpu"].to(device=u_flat.device)
                if st["halo_idx"] is None or st["halo_idx"].device != u_flat.device:
                    st["halo_idx"] = st["halo_idx_cpu"].to(device=u_flat.device)
                if st.get("non_halo_idx") is None or (st.get("non_halo_idx") is not None and st["non_halo_idx"].device != u_flat.device):
                    st["non_halo_idx"] = st.get("non_halo_idx_cpu", torch.empty((0,), dtype=torch.long)).to(device=u_flat.device)
                if st.get("rand_core_idx") is None or (st.get("rand_core_idx") is not None and st["rand_core_idx"].device != u_flat.device):
                    st["rand_core_idx"] = st.get("rand_core_idx_cpu", torch.empty((0,), dtype=torch.long)).to(device=u_flat.device)

                super_idx_dev = st["super_idx"]
                halo_idx_dev = st["halo_idx"]
                non_halo_idx_dev = st.get("non_halo_idx")
                if non_halo_idx_dev is None:
                    non_halo_idx_dev = torch.empty((0,), device=u_flat.device, dtype=torch.long)
                rand_core_idx_dev = st.get("rand_core_idx")
                if rand_core_idx_dev is None:
                    rand_core_idx_dev = torch.empty((0,), device=u_flat.device, dtype=torch.long)

                # Compute q = u * s where s := dL/du is already computed by backprop.
                # We only materialize the supernode+halo indices.
                idx_union = torch.cat([super_idx_dev, halo_idx_dev, non_halo_idx_dev, rand_core_idx_dev], dim=0)  # [|M|+|H|+|N|+|R|]
                try:
                    u_sel = u_flat.index_select(1, idx_union).float()  # [N, |M|+|H|]
                    s_sel = g_u_flat.index_select(1, idx_union).float()  # [N, |M|+|H|]
                except Exception:
                    return

                q_sel = u_sel * s_sel  # [N, |M|+|H|+|N|+|R|]
                n_super = super_idx_dev.numel()
                n_halo = halo_idx_dev.numel()
                n_non_halo = non_halo_idx_dev.numel()
                rand_core_idx_dev.numel()
                q_super = q_sel[:, :n_super]  # [N, |M|]
                q_halo = q_sel[:, n_super : n_super + n_halo]  # [N, |H|]
                q_non_halo = q_sel[:, n_super + n_halo : n_super + n_halo + n_non_halo]  # [N, |N|]
                q_rand = q_sel[:, n_super + n_halo + n_non_halo :]  # [N, |R|]

                N = q_sel.shape[0]

                # Initialize streaming sums on first batch
                if st["sum_q_super"] is None:
                    st["sum_q_super"] = torch.zeros(q_super.shape[1], device=q_super.device, dtype=torch.float32)
                    st["sum_q2_super"] = torch.zeros_like(st["sum_q_super"])
                    st["sum_q3_super"] = torch.zeros_like(st["sum_q_super"])
                    st["sum_q4_super"] = torch.zeros_like(st["sum_q_super"])
                    st["sum_q_halo"] = torch.zeros(q_halo.shape[1], device=q_halo.device, dtype=torch.float32)
                    st["sum_q2_halo"] = torch.zeros_like(st["sum_q_halo"])
                    st["sum_q3_halo"] = torch.zeros_like(st["sum_q_halo"])
                    st["sum_q4_halo"] = torch.zeros_like(st["sum_q_halo"])
                    st["sum_q_halo_super"] = torch.zeros((q_halo.shape[1], q_super.shape[1]), device=q_halo.device, dtype=torch.float32)
                    st["sum_q_non_halo"] = torch.zeros(q_non_halo.shape[1], device=q_non_halo.device, dtype=torch.float32)
                    st["sum_q2_non_halo"] = torch.zeros_like(st["sum_q_non_halo"])
                    st["sum_q3_non_halo"] = torch.zeros_like(st["sum_q_non_halo"])
                    st["sum_q4_non_halo"] = torch.zeros_like(st["sum_q_non_halo"])
                    st["sum_q_non_halo_super"] = torch.zeros((q_non_halo.shape[1], q_super.shape[1]), device=q_non_halo.device, dtype=torch.float32)
                    st["sum_q_rand"] = torch.zeros(q_rand.shape[1], device=q_sel.device, dtype=torch.float32)
                    st["sum_q2_rand"] = torch.zeros_like(st["sum_q_rand"])
                    st["sum_q_halo_rand"] = torch.zeros((q_halo.shape[1], q_rand.shape[1]), device=q_sel.device, dtype=torch.float32)
                    st["sum_q_non_halo_rand"] = torch.zeros((q_non_halo.shape[1], q_rand.shape[1]), device=q_sel.device, dtype=torch.float32)

                st["sum_q_super"] += q_super.sum(dim=0)
                st["sum_q2_super"] += (q_super * q_super).sum(dim=0)
                st["sum_q3_super"] += (q_super * q_super * q_super).sum(dim=0)
                st["sum_q4_super"] += (q_super * q_super * q_super * q_super).sum(dim=0)
                st["sum_q_halo"] += q_halo.sum(dim=0)
                st["sum_q2_halo"] += (q_halo * q_halo).sum(dim=0)
                st["sum_q3_halo"] += (q_halo * q_halo * q_halo).sum(dim=0)
                st["sum_q4_halo"] += (q_halo * q_halo * q_halo * q_halo).sum(dim=0)
                st["sum_q_halo_super"] += q_halo.transpose(0, 1) @ q_super  # [|H|,|M|]
                if q_non_halo.numel() > 0:
                    st["sum_q_non_halo"] += q_non_halo.sum(dim=0)
                    st["sum_q2_non_halo"] += (q_non_halo * q_non_halo).sum(dim=0)
                    st["sum_q3_non_halo"] += (q_non_halo * q_non_halo * q_non_halo).sum(dim=0)
                    st["sum_q4_non_halo"] += (q_non_halo * q_non_halo * q_non_halo * q_non_halo).sum(dim=0)
                    st["sum_q_non_halo_super"] += q_non_halo.transpose(0, 1) @ q_super  # [|N|,|M|]
                if q_rand.numel() > 0:
                    st["sum_q_rand"] += q_rand.sum(dim=0)
                    st["sum_q2_rand"] += (q_rand * q_rand).sum(dim=0)
                    st["sum_q_halo_rand"] += q_halo.transpose(0, 1) @ q_rand  # [|H|,|R|]
                    if q_non_halo.numel() > 0:
                        st["sum_q_non_halo_rand"] += q_non_halo.transpose(0, 1) @ q_rand  # [|N|,|R|]
                st["count"] += N

            return fwd_hook, bwd_hook

        for layer_name, module in module_dict.items():
            if layer_name not in plan:
                continue
            fwd, bwd = make_hooks(layer_name)
            hooks.append(module.register_forward_hook(fwd))
            hooks.append(module.register_full_backward_hook(bwd))

        # Run calibration (full forward+backward) for q-statistics
        self.model.eval()
        device = torch.device(self.config.device)

        # Try to use halo_analysis.max_length if present
        halo_cfg = getattr(self.config, "halo_analysis", {}) or {}
        if hasattr(halo_cfg, "__dict__"):
            halo_cfg = vars(halo_cfg)
        max_length = int(halo_cfg.get("max_length", 256))

        try:
            for idx, text in enumerate(calibration_texts):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                labels = inputs["input_ids"].clone()
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
                labels[labels == pad_token_id] = -100
                inputs["labels"] = labels

                self.model.zero_grad(set_to_none=True)
                out = self.model(**inputs)
                loss = out.loss
                loss.backward()

                if (idx + 1) % 1 == 0:
                    logger.info(f"  SCAR q-stats: processed {idx+1}/{len(calibration_texts)} samples, loss={loss.item():.4f}")
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Phase 3: Compute Protect + final importance scores; store into importance_scores
        # ------------------------------------------------------------------
        agg_red_halo: List[float] = []
        agg_red_non_halo: List[float] = []
        agg_red_rand_halo: List[float] = []
        agg_red_rand_non_halo: List[float] = []
        for layer_name, st in plan.items():
            N = int(st.get("count", 0))
            if N <= 1 or st["sum_q_halo_super"] is None:
                logger.warning(f"SCAR connectivity: insufficient q-stats for {layer_name} (N={N}); skipping layer")
                continue

            sum_q_super = st["sum_q_super"].detach().cpu()
            sum_q2_super = st["sum_q2_super"].detach().cpu()
            sum_q3_super = st.get("sum_q3_super")
            sum_q4_super = st.get("sum_q4_super")
            sum_q_halo = st["sum_q_halo"].detach().cpu()
            sum_q2_halo = st["sum_q2_halo"].detach().cpu()
            sum_q3_halo = st.get("sum_q3_halo")
            sum_q4_halo = st.get("sum_q4_halo")
            sum_q_halo_super = st["sum_q_halo_super"].detach().cpu()

            # Optional Gaussianity diagnostics for the q-signal: skewness/kurtosis of q_i over tokens.
            # This is a light-weight check of the Gaussian MI approximation used for redundancy.
            def _q_gaussianity(sum1, sum2, sum3, sum4, N_tokens: int) -> Dict[str, Any]:
                if sum1 is None:
                    return {"n_channels": 0}
                if hasattr(sum1, "detach"):
                    sum1 = sum1.detach().cpu()
                if hasattr(sum2, "detach"):
                    sum2 = sum2.detach().cpu()
                if hasattr(sum3, "detach"):
                    sum3 = sum3.detach().cpu()
                if hasattr(sum4, "detach"):
                    sum4 = sum4.detach().cpu()
                if int(N_tokens) <= 1 or int(getattr(sum1, "numel", lambda: 0)()) <= 0:
                    return {"n_channels": int(getattr(sum1, "numel", lambda: 0)())}

                Nf = float(N_tokens)
                m1 = sum1.float() / Nf
                m2 = sum2.float() / Nf
                m3 = sum3.float() / Nf
                m4 = sum4.float() / Nf

                var = (m2 - m1 * m1).clamp_min(0.0)
                std = var.sqrt()

                mu3 = m3 - 3.0 * m1 * m2 + 2.0 * (m1 * m1 * m1)
                mu4 = m4 - 4.0 * m1 * m3 + 6.0 * (m1 * m1) * m2 - 3.0 * (m1 * m1 * m1 * m1)

                denom3 = (std * std * std).clamp_min(eps)
                denom4 = (var * var).clamp_min(eps)

                skew = torch.where(std > 0.0, mu3 / denom3, torch.zeros_like(mu3))
                kurt_excess = torch.where(var > 0.0, (mu4 / denom4) - 3.0, torch.zeros_like(mu4))

                abs_skew = skew.abs()
                return {
                    "n_channels": int(abs_skew.numel()),
                    "mean_abs_skew": float(abs_skew.mean().item()),
                    "median_abs_skew": float(abs_skew.median().item()),
                    "frac_abs_skew_lt_0_5": float((abs_skew < 0.5).float().mean().item()),
                    "mean_excess_kurtosis": float(kurt_excess.mean().item()),
                    "median_excess_kurtosis": float(kurt_excess.median().item()),
                }

            q_gauss_super = _q_gaussianity(sum_q_super, sum_q2_super, sum_q3_super, sum_q4_super, N)
            q_gauss_halo = _q_gaussianity(sum_q_halo, sum_q2_halo, sum_q3_halo, sum_q4_halo, N)
            # non-halo gaussianity is computed later if the non-halo sample exists
            q_gauss_non_halo = {"n_channels": 0}

            mean_super = sum_q_super / float(N)
            mean_halo = sum_q_halo / float(N)

            cov = (sum_q_halo_super / float(N)) - (mean_halo.unsqueeze(1) * mean_super.unsqueeze(0))
            var_halo = (sum_q2_halo / float(N)) - (mean_halo * mean_halo)
            var_super = (sum_q2_super / float(N)) - (mean_super * mean_super)

            denom = torch.sqrt(var_halo.clamp_min(0).unsqueeze(1) * var_super.clamp_min(0).unsqueeze(0) + eps)
            corr = torch.where(denom > 0, cov / denom, torch.zeros_like(cov))
            corr = corr.clamp(-0.9999, 0.9999)

            corr_eff = torch.clamp(corr, min=0.0) if positive_redundancy else corr
            rho_sq = (corr_eff * corr_eff).clamp(0.0, 0.9999)
            mi = -0.5 * torch.log(1 - rho_sq)

            # Aggregate redundancy-to-core with a Top-k mean (reduces max inflation / multiple-comparisons effects).
            red_reduce = str(supernode_cfg.get("redundancy_reduce", "topk_mean")).lower()
            try:
                red_topk = int(supernode_cfg.get("redundancy_topk", 5))
            except Exception:
                red_topk = 5
            red_topk = max(1, min(int(red_topk), int(mi.shape[1])))

            redundancy_to_core_max = mi.max(dim=1).values  # [|H|]
            redundancy_to_core_topk_mean = torch.topk(mi, k=red_topk, dim=1, largest=True).values.mean(dim=1)  # [|H|]
            redundancy_to_core = (
                redundancy_to_core_topk_mean
                if red_reduce in {"topk", "topk_mean", "mean_topk", "avg_topk", "average_topk"}
                else redundancy_to_core_max
            )

            # Optional: redundancy-to-core for a sampled set of non-halo channels (analysis only).
            redundancy_to_core_non_halo = None
            non_halo_idx_cpu = st.get("non_halo_idx_cpu", None)
            if (
                non_halo_idx_cpu is not None
                and hasattr(non_halo_idx_cpu, "numel")
                and int(non_halo_idx_cpu.numel()) > 0
                and st.get("sum_q_non_halo_super") is not None
            ):
                sum_q_non = st["sum_q_non_halo"].detach().cpu()
                sum_q2_non = st["sum_q2_non_halo"].detach().cpu()
                sum_q3_non = st.get("sum_q3_non_halo")
                sum_q4_non = st.get("sum_q4_non_halo")
                sum_q_non_super = st["sum_q_non_halo_super"].detach().cpu()

                q_gauss_non_halo = _q_gaussianity(sum_q_non, sum_q2_non, sum_q3_non, sum_q4_non, N)

                mean_non = sum_q_non / float(N)
                cov_non = (sum_q_non_super / float(N)) - (mean_non.unsqueeze(1) * mean_super.unsqueeze(0))
                var_non = (sum_q2_non / float(N)) - (mean_non * mean_non)
                denom_non = torch.sqrt(var_non.clamp_min(0).unsqueeze(1) * var_super.clamp_min(0).unsqueeze(0) + eps)
                corr_non = torch.where(denom_non > 0, cov_non / denom_non, torch.zeros_like(cov_non))
                corr_non = corr_non.clamp(-0.9999, 0.9999)

                corr_eff_non = torch.clamp(corr_non, min=0.0) if positive_redundancy else corr_non
                rho_sq_non = (corr_eff_non * corr_eff_non).clamp(0.0, 0.9999)
                mi_non = -0.5 * torch.log(1 - rho_sq_non)
                redundancy_to_core_non_halo_max = mi_non.max(dim=1).values  # [|N|]
                red_topk_non = max(1, min(int(red_topk), int(mi_non.shape[1])))
                redundancy_to_core_non_halo_topk_mean = torch.topk(mi_non, k=red_topk_non, dim=1, largest=True).values.mean(dim=1)  # [|N|]
                redundancy_to_core_non_halo = (
                    redundancy_to_core_non_halo_topk_mean
                    if red_reduce in {"topk", "topk_mean", "mean_topk", "avg_topk", "average_topk"}
                    else redundancy_to_core_non_halo_max
                )

            # Optional multiple-comparisons control: redundancy-to-core against a matched random core
            # (same size as the supernode core, sampled from the non-halo pool).
            redundancy_to_rand_core = None
            redundancy_to_rand_core_non_halo = None
            rand_core_idx_cpu = st.get("rand_core_idx_cpu", None)
            if (
                rand_core_idx_cpu is not None
                and hasattr(rand_core_idx_cpu, "numel")
                and int(rand_core_idx_cpu.numel()) > 0
                and st.get("sum_q_halo_rand") is not None
                and st.get("sum_q_rand") is not None
                and st.get("sum_q2_rand") is not None
            ):
                sum_q_rand = st["sum_q_rand"].detach().cpu()
                sum_q2_rand = st["sum_q2_rand"].detach().cpu()
                sum_q_halo_rand = st["sum_q_halo_rand"].detach().cpu()

                mean_rand = sum_q_rand / float(N)
                var_rand = (sum_q2_rand / float(N)) - (mean_rand * mean_rand)

                cov_hr = (sum_q_halo_rand / float(N)) - (mean_halo.unsqueeze(1) * mean_rand.unsqueeze(0))
                denom_hr = torch.sqrt(var_halo.clamp_min(0).unsqueeze(1) * var_rand.clamp_min(0).unsqueeze(0) + eps)
                corr_hr = torch.where(denom_hr > 0, cov_hr / denom_hr, torch.zeros_like(cov_hr))
                corr_hr = corr_hr.clamp(-0.9999, 0.9999)

                corr_eff_hr = torch.clamp(corr_hr, min=0.0) if positive_redundancy else corr_hr
                rho_sq_hr = (corr_eff_hr * corr_eff_hr).clamp(0.0, 0.9999)
                mi_rand = -0.5 * torch.log(1 - rho_sq_hr)

                rand_topk = max(1, min(int(red_topk), int(mi_rand.shape[1])))
                redundancy_to_rand_core_max = mi_rand.max(dim=1).values  # [|H|]
                redundancy_to_rand_core_topk_mean = torch.topk(mi_rand, k=rand_topk, dim=1, largest=True).values.mean(dim=1)  # [|H|]
                redundancy_to_rand_core = (
                    redundancy_to_rand_core_topk_mean
                    if red_reduce in {"topk", "topk_mean", "mean_topk", "avg_topk", "average_topk"}
                    else redundancy_to_rand_core_max
                )

                # Non-halo sampled channels vs random core (optional)
                if st.get("sum_q_non_halo_rand") is not None and st.get("sum_q_non_halo") is not None and st.get("sum_q2_non_halo") is not None:
                    sum_q_non = st["sum_q_non_halo"].detach().cpu()
                    sum_q2_non = st["sum_q2_non_halo"].detach().cpu()
                    sum_q_non_rand = st["sum_q_non_halo_rand"].detach().cpu()

                    mean_non = sum_q_non / float(N)
                    var_non = (sum_q2_non / float(N)) - (mean_non * mean_non)

                    cov_nr = (sum_q_non_rand / float(N)) - (mean_non.unsqueeze(1) * mean_rand.unsqueeze(0))
                    denom_nr = torch.sqrt(var_non.clamp_min(0).unsqueeze(1) * var_rand.clamp_min(0).unsqueeze(0) + eps)
                    corr_nr = torch.where(denom_nr > 0, cov_nr / denom_nr, torch.zeros_like(cov_nr))
                    corr_nr = corr_nr.clamp(-0.9999, 0.9999)

                    corr_eff_nr = torch.clamp(corr_nr, min=0.0) if positive_redundancy else corr_nr
                    rho_sq_nr = (corr_eff_nr * corr_eff_nr).clamp(0.0, 0.9999)
                    mi_non_rand = -0.5 * torch.log(1 - rho_sq_nr)

                    non_rand_topk = max(1, min(int(red_topk), int(mi_non_rand.shape[1])))
                    redundancy_to_rand_core_non_halo_max = mi_non_rand.max(dim=1).values
                    redundancy_to_rand_core_non_halo_topk_mean = torch.topk(mi_non_rand, k=non_rand_topk, dim=1, largest=True).values.mean(dim=1)
                    redundancy_to_rand_core_non_halo = (
                        redundancy_to_rand_core_non_halo_topk_mean
                        if red_reduce in {"topk", "topk_mean", "mean_topk", "avg_topk", "average_topk"}
                        else redundancy_to_rand_core_non_halo_max
                    )

            # Convert redundancy-to-core into a [0, 1] protection score.
            #
            # Empirically, redundancy magnitudes can be extremely small; min-max normalization
            # then collapses most halo channels near Protect~1. But a fully linear rank/CDF
            # can be too aggressive when redundancy estimates are noisy. We therefore default
            # to a *soft* rank-power mapping that mainly penalizes only the most redundant tail.
            norm_mode = str(supernode_cfg.get("protection_normalization", "rank_power")).lower()
            if norm_mode == "minmax":
                red_min = redundancy_to_core.min()
                red_max = redundancy_to_core.max()
                if red_max > red_min:
                    red_norm = (redundancy_to_core - red_min) / (red_max - red_min + eps)
                else:
                    red_norm = torch.zeros_like(redundancy_to_core)
                protect_halo = (1.0 - red_norm).clamp(0.0, 1.0)
            elif norm_mode in {"rank", "cdf"}:
                if redundancy_to_core.numel() <= 1:
                    protect_halo = torch.ones_like(redundancy_to_core)
                else:
                    # Ascending ranks: lowest redundancy -> highest protection.
                    _, order = torch.sort(redundancy_to_core, stable=True)
                    ranks = torch.empty_like(order, dtype=torch.float32)
                    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
                    red_rank = ranks / float(max(1, order.numel() - 1))
                    protect_halo = (1.0 - red_rank).clamp(0.0, 1.0)
            else:
                # rank_power (default): Protect = floor + (1-floor)*(1 - rank^gamma)
                if redundancy_to_core.numel() <= 1:
                    protect_halo = torch.ones_like(redundancy_to_core)
                else:
                    _, order = torch.sort(redundancy_to_core, stable=True)
                    ranks = torch.empty_like(order, dtype=torch.float32)
                    ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
                    red_rank = ranks / float(max(1, order.numel() - 1))
                    red_rank = red_rank.clamp(0.0, 1.0)

                    gamma = supernode_cfg.get("protection_rank_power", 8.0)
                    try:
                        gamma_f = float(gamma)
                    except Exception:
                        gamma_f = 8.0
                    if not (gamma_f > 0):
                        gamma_f = 8.0

                    floor = supernode_cfg.get("protection_floor", 0.2)
                    try:
                        floor_f = float(floor)
                    except Exception:
                        floor_f = 0.2
                    floor_f = float(min(1.0, max(0.0, floor_f)))

                    protect_halo = floor_f + (1.0 - floor_f) * (1.0 - red_rank.pow(gamma_f))
                    protect_halo = protect_halo.clamp(0.0, 1.0)

            m = st["m"]
            lp = st["lp_cpu"].float()
            conn = st["conn_cpu"].float()
            super_idx = st["super_idx_cpu"]
            halo_idx = st["halo_idx_cpu"]

            protect_full = torch.ones(m, dtype=torch.float32)
            protect_full[halo_idx] = protect_halo
            protect_full[super_idx] = 1.0

            # Store redundancy-to-core in full channel space (defined only for halo channels)
            redundancy_full = torch.full((m,), float("nan"), dtype=torch.float32)
            try:
                redundancy_full[halo_idx] = redundancy_to_core.float()
            except Exception:
                pass
            if redundancy_to_core_non_halo is not None and non_halo_idx_cpu is not None:
                try:
                    redundancy_full[non_halo_idx_cpu] = redundancy_to_core_non_halo.float()
                except Exception:
                    pass

            # SCAR-Prot and SCAR-Conn importance scores (high=keep)
            prot_score = (lp * protect_full).float()
            conn_score = (lp * ((1.0 - conn) + conn * protect_full)).float()

            # Explicitly protect supernodes (also enforced later by apply_pruning via supernode_mask)
            prot_boost = float(prot_score.max().item()) + 1.0
            conn_boost = float(conn_score.max().item()) + 1.0
            prot_score[super_idx] = prot_boost
            conn_score[super_idx] = conn_boost

            # Optional: cross-layer read-halo pruning score (weight-based; ablation).
            # This applies an extra protection multiplier to channels in layer ℓ based on how
            # strongly they READ from the previous layer's supernode-written hidden subspace.
            #
            # By default, this is disabled and does not affect SCAR.
            read_halo_score = prot_score  # legacy ablation: prune high-ReadConn "readers"
            read_halo_protect_score = prot_score  # ablation: protect high-ReadConn "readers"
            two_halo_score = prot_score  # write-halo (SCAR-Prot) × read-halo redundancy protection
            read_conn_full: Optional[torch.Tensor] = None
            read_protect_full: Optional[torch.Tensor] = None  # for `supernode_read_halo_score`
            read_protect_conn_full: Optional[torch.Tensor] = None  # for `supernode_read_halo_protect_score`
            read_redundancy_full: Optional[torch.Tensor] = None  # similarity-to-centroid (read-halo only)
            read_protect_redund_full: Optional[torch.Tensor] = None  # for `supernode_two_halo_score`
            read_halo_mask: Optional[torch.Tensor] = None
            read_halo_stats: Optional[Dict[str, Any]] = None
            if read_halo_prune_enabled:
                try:
                    li = int(st.get("layer_idx_int", 0) or 0)
                except Exception:
                    li = 0

                if li > 0 and (li - 1) in plan_by_layer_idx:
                    prev = plan_by_layer_idx.get(li - 1) or {}
                    prev_core = prev.get("core_hidden_idx_cpu", None)
                    if prev_core is not None and hasattr(prev_core, "numel") and int(prev_core.numel()) > 0:
                        # Resolve gate/up weights for the *current* layer.
                        gate_name = layer_name.replace("down_proj", "gate_proj")
                        up_name = layer_name.replace("down_proj", "up_proj")
                        gate_mod = module_dict.get(gate_name) or module_dict.get("model.model." + gate_name)
                        up_mod = module_dict.get(up_name) or module_dict.get("model.model." + up_name)
                        if gate_mod is None or up_mod is None:
                            # Suffix match fallback
                            for _k, _v in module_dict.items():
                                if gate_mod is None and _k.endswith(gate_name):
                                    gate_mod = _v
                                if up_mod is None and _k.endswith(up_name):
                                    up_mod = _v

                        if gate_mod is not None and up_mod is not None and hasattr(gate_mod, "weight") and hasattr(up_mod, "weight"):
                            Wg = gate_mod.weight.detach().float().cpu().abs()  # [m, hidden]
                            Wu = up_mod.weight.detach().float().cpu().abs()
                            if Wg.ndim == 2 and Wu.ndim == 2 and Wg.shape == Wu.shape and int(Wg.shape[0]) == int(m):
                                hidden_dim = int(Wg.shape[1])
                                S = prev_core.detach().long().cpu()
                                S = S[(S >= 0) & (S < hidden_dim)]
                                if int(S.numel()) > 0:
                                    num = Wg.index_select(1, S).sum(dim=1) + Wu.index_select(1, S).sum(dim=1)
                                    den = Wg.sum(dim=1) + Wu.sum(dim=1) + eps
                                    read_conn = (num / den).clamp(0.0, 1.0)  # [m]

                                    # Define read-halo among non-supernodes (top by ReadConn).
                                    non_super_idx = (~super_mask).nonzero(as_tuple=True)[0]
                                    if non_super_idx.numel() > 0:
                                        num_read_halo = max(1, int(_rh_frac * int(non_super_idx.numel())))
                                        vals = read_conn[non_super_idx]
                                        _, rel = torch.topk(vals, k=num_read_halo, largest=True)
                                        read_halo_idx = non_super_idx[rel].long()

                                        # (A) Legacy ablation: Convert ReadConn to a protection multiplier within the read-halo:
                                        # high ReadConn => lower protection => pruned more.
                                        read_vals = read_conn[read_halo_idx]
                                        _, order = torch.sort(read_vals, stable=True)  # ascending
                                        ranks = torch.empty_like(order, dtype=torch.float32)
                                        ranks[order] = torch.arange(order.numel(), dtype=torch.float32)
                                        rank = ranks / float(max(1, order.numel() - 1))
                                        protect_read = _rh_floor + (1.0 - _rh_floor) * (1.0 - rank.pow(float(_rh_gamma)))
                                        protect_read = protect_read.clamp(0.0, 1.0)

                                        read_protect = torch.ones(m, dtype=torch.float32)
                                        read_protect[read_halo_idx] = protect_read
                                        read_protect[super_idx] = 1.0

                                        read_halo_score = (prot_score * read_protect).float()

                                        # (B) Alternative ablation: protect high-ReadConn readers (opposite direction).
                                        protect_read_conn = _rh_floor + (1.0 - _rh_floor) * rank.pow(float(_rh_gamma))
                                        protect_read_conn = protect_read_conn.clamp(0.0, 1.0)
                                        read_protect_conn = torch.ones(m, dtype=torch.float32)
                                        read_protect_conn[read_halo_idx] = protect_read_conn
                                        read_protect_conn[super_idx] = 1.0
                                        read_halo_protect_score = (prot_score * read_protect_conn).float()

                                        # (C) Two-halo score: keep read-halo computation but only penalize *redundant* readers.
                                        #
                                        # We estimate within-read-halo redundancy using *weight signatures* restricted to
                                        # the previous layer's supernode-written hidden support S:
                                        #   sig_j = concat(|W_gate[j,S]|, |W_up[j,S]|).
                                        # Redundancy proxy = cosine similarity of sig_j to the read-halo centroid.
                                        two_halo_read_protect = torch.ones(m, dtype=torch.float32)
                                        sim_to_centroid = torch.full((m,), float("nan"), dtype=torch.float32)
                                        try:
                                            if read_halo_idx.numel() >= 2:
                                                sig = torch.cat(
                                                    [
                                                        Wg.index_select(0, read_halo_idx).index_select(1, S),
                                                        Wu.index_select(0, read_halo_idx).index_select(1, S),
                                                    ],
                                                    dim=1,
                                                ).float()  # [R, 2|S|]
                                                sig = sig / (sig.norm(dim=1, keepdim=True) + eps)
                                                centroid = sig.mean(dim=0, keepdim=True)
                                                centroid = centroid / (centroid.norm(dim=1, keepdim=True) + eps)
                                                sim = (sig @ centroid.T).squeeze(1).clamp(0.0, 1.0)  # [R]
                                                sim_to_centroid[read_halo_idx] = sim.cpu()

                                                # High similarity => more redundant => lower protection.
                                                _, order2 = torch.sort(sim, stable=True)  # ascending
                                                ranks2 = torch.empty_like(order2, dtype=torch.float32)
                                                ranks2[order2] = torch.arange(order2.numel(), dtype=torch.float32)
                                                rank2 = ranks2 / float(max(1, order2.numel() - 1))
                                                protect_redund = _rh_floor + (1.0 - _rh_floor) * (1.0 - rank2.pow(float(_rh_gamma)))
                                                protect_redund = protect_redund.clamp(0.0, 1.0)
                                                two_halo_read_protect[read_halo_idx] = protect_redund.cpu()
                                                two_halo_read_protect[super_idx] = 1.0

                                                # Random baseline (for reporting only): same-size random set from non-supernodes
                                                # using the same signature definition.
                                                g = torch.Generator()
                                                seed_base = (
                                                    int(read_halo_prune_cfg.get("random_seed", 0) or 0)
                                                    if isinstance(read_halo_prune_cfg, dict)
                                                    else 0
                                                )
                                                g.manual_seed(seed_base + int(li))
                                                perm = torch.randperm(int(non_super_idx.numel()), generator=g)
                                                rand_idx = non_super_idx[perm[: int(read_halo_idx.numel())]].long()
                                                sig_r = torch.cat(
                                                    [
                                                        Wg.index_select(0, rand_idx).index_select(1, S),
                                                        Wu.index_select(0, rand_idx).index_select(1, S),
                                                    ],
                                                    dim=1,
                                                ).float()
                                                sig_r = sig_r / (sig_r.norm(dim=1, keepdim=True) + eps)
                                                centroid_r = sig_r.mean(dim=0, keepdim=True)
                                                centroid_r = centroid_r / (centroid_r.norm(dim=1, keepdim=True) + eps)
                                                sim_r = (sig_r @ centroid_r.T).squeeze(1).clamp(0.0, 1.0)

                                                read_halo_stats = {
                                                    "prev_layer_idx": int(li - 1),
                                                    "support_size": int(S.numel()),
                                                    "read_halo_size": int(read_halo_idx.numel()),
                                                    "readconn": {
                                                        "mean": float(read_conn.mean().item()),
                                                        "std": float(read_conn.std().item()),
                                                        "threshold": float(read_vals.min().item()) if read_vals.numel() else None,
                                                    },
                                                    "weight_redundancy": {
                                                        "cosine_to_centroid_mean": float(sim.mean().item()),
                                                        "cosine_to_centroid_std": float(sim.std().item()),
                                                        "random_cosine_to_centroid_mean": float(sim_r.mean().item()),
                                                        "random_cosine_to_centroid_std": float(sim_r.std().item()),
                                                        "difference_mean": float((sim.mean() - sim_r.mean()).item()),
                                                    },
                                                }
                                        except Exception:
                                            pass

                                        two_halo_score = (prot_score * two_halo_read_protect).float()

                                        read_conn_full = read_conn.float()
                                        read_protect_full = read_protect.float()
                                        read_protect_conn_full = read_protect_conn.float()
                                        read_redundancy_full = sim_to_centroid.float()
                                        read_protect_redund_full = two_halo_read_protect.float()
                                        read_halo_mask = torch.zeros(m, dtype=torch.bool)
                                        read_halo_mask[read_halo_idx] = True
                                        read_halo_mask[super_idx] = False
                # else: no previous layer (layer 0) -> read_halo_score stays == prot_score

            halo_mask = torch.zeros(m, dtype=torch.bool)
            halo_mask[halo_idx] = True

            super_mask = torch.zeros(m, dtype=torch.bool)
            super_mask[super_idx] = True

            layer_scores = self.importance_scores.get(layer_name, {})
            layer_scores["supernode_protection_score"] = prot_score
            layer_scores["supernode_connectivity_score"] = conn_score
            layer_scores["connectivity_score"] = conn
            layer_scores["protection_score"] = protect_full
            layer_scores["redundancy_to_core"] = redundancy_full
            # Always store the read-halo score keys so config lists can include them safely.
            # If read-halo pruning is disabled, these default to SCAR-Prot behavior.
            layer_scores["supernode_read_halo_score"] = read_halo_score
            layer_scores["supernode_read_halo_protect_score"] = read_halo_protect_score
            layer_scores["supernode_two_halo_score"] = two_halo_score
            if read_conn_full is not None:
                layer_scores["read_halo_readconn"] = read_conn_full
            if read_protect_full is not None:
                layer_scores["read_halo_protection"] = read_protect_full
            if read_protect_conn_full is not None:
                layer_scores["read_halo_protection_readconn"] = read_protect_conn_full
            if read_redundancy_full is not None:
                layer_scores["read_halo_weight_cosine_to_centroid"] = read_redundancy_full
            if read_protect_redund_full is not None:
                layer_scores["read_halo_protection_redundancy"] = read_protect_redund_full
            if read_halo_mask is not None:
                layer_scores["read_halo_mask"] = read_halo_mask
            layer_scores["halo_mask"] = halo_mask
            layer_scores["supernode_mask"] = super_mask
            self.importance_scores[layer_name] = layer_scores

            # Propagate the read-halo pruning score to sibling MLP projections (gate/up) so that
            # channel masking is consistent when pruning code looks up scores on those modules.
            if isinstance(layer_name, str) and "down_proj" in layer_name:
                for sibling_proj in ("gate_proj", "up_proj"):
                    sibling_name = layer_name.replace("down_proj", sibling_proj)
                    sib = self.importance_scores.get(sibling_name, {}) or {}
                    sib["supernode_read_halo_score"] = read_halo_score
                    sib["supernode_read_halo_protect_score"] = read_halo_protect_score
                    sib["supernode_two_halo_score"] = two_halo_score
                    if read_conn_full is not None:
                        sib["read_halo_readconn"] = read_conn_full
                    if read_protect_full is not None:
                        sib["read_halo_protection"] = read_protect_full
                    if read_protect_conn_full is not None:
                        sib["read_halo_protection_readconn"] = read_protect_conn_full
                    if read_redundancy_full is not None:
                        sib["read_halo_weight_cosine_to_centroid"] = read_redundancy_full
                    if read_protect_redund_full is not None:
                        sib["read_halo_protection_redundancy"] = read_protect_redund_full
                    if read_halo_mask is not None:
                        sib["read_halo_mask"] = read_halo_mask
                    # Also ensure supernode_mask is available on siblings (safety)
                    if "supernode_mask" not in sib:
                        sib["supernode_mask"] = super_mask
                    self.importance_scores[sibling_name] = sib

            results[layer_name] = {
                "num_supernodes": int(super_idx.numel()),
                "num_halo": int(halo_idx.numel()),
                "num_non_halo_sample": int(non_halo_idx_cpu.numel()) if non_halo_idx_cpu is not None else 0,
                "rand_core_size": int(st.get("rand_core_idx_cpu").numel()) if st.get("rand_core_idx_cpu") is not None else 0,
                "q_samples": N,
                "conn_mean": float(conn.mean().item()),
                "redundancy_reduce": str(red_reduce),
                "redundancy_topk": int(red_topk),
                "protect_halo_mean": float(protect_halo.mean().item()) if protect_halo.numel() else 0.0,
                "redundancy_to_core_mean": float(redundancy_to_core.mean().item()) if redundancy_to_core.numel() else 0.0,
                "non_halo_redundancy_to_core_mean": (
                    float(redundancy_to_core_non_halo.mean().item())
                    if redundancy_to_core_non_halo is not None and redundancy_to_core_non_halo.numel()
                    else 0.0
                ),
                "redundancy_to_rand_core_mean": (
                    float(redundancy_to_rand_core.mean().item())
                    if redundancy_to_rand_core is not None and hasattr(redundancy_to_rand_core, "numel") and int(redundancy_to_rand_core.numel()) > 0
                    else None
                ),
                "non_halo_redundancy_to_rand_core_mean": (
                    float(redundancy_to_rand_core_non_halo.mean().item())
                    if redundancy_to_rand_core_non_halo is not None
                    and hasattr(redundancy_to_rand_core_non_halo, "numel")
                    and int(redundancy_to_rand_core_non_halo.numel()) > 0
                    else None
                ),
                "q_gaussianity": {
                    "supernodes": q_gauss_super,
                    "halo": q_gauss_halo,
                    "non_halo_sample": q_gauss_non_halo,
                },
            }
            if read_halo_prune_enabled and read_halo_stats is not None:
                results[layer_name]["read_halo"] = read_halo_stats

            # Aggregate distributions (for tables / sanity checks)
            try:
                halo_vals = redundancy_to_core.detach().float()
                halo_vals = halo_vals[torch.isfinite(halo_vals)]
                agg_red_halo.extend([float(x) for x in halo_vals.tolist() if x == x])
            except Exception:
                pass
            if redundancy_to_rand_core is not None:
                try:
                    rand_vals = redundancy_to_rand_core.detach().float()
                    rand_vals = rand_vals[torch.isfinite(rand_vals)]
                    agg_red_rand_halo.extend([float(x) for x in rand_vals.tolist() if x == x])
                except Exception:
                    pass
            if redundancy_to_core_non_halo is not None:
                try:
                    non_vals = redundancy_to_core_non_halo.detach().float()
                    non_vals = non_vals[torch.isfinite(non_vals)]
                    agg_red_non_halo.extend([float(x) for x in non_vals.tolist() if x == x])
                except Exception:
                    pass
            if redundancy_to_rand_core_non_halo is not None:
                try:
                    non_rand_vals = redundancy_to_rand_core_non_halo.detach().float()
                    non_rand_vals = non_rand_vals[torch.isfinite(non_rand_vals)]
                    agg_red_rand_non_halo.extend([float(x) for x in non_rand_vals.tolist() if x == x])
                except Exception:
                    pass

        # Add aggregate stats for summary tables (useful even when per-layer values are noisy).
        if agg_red_halo or agg_red_non_halo:

            def _stats(vals: List[float]) -> Dict[str, Any]:
                arr = np.asarray(vals, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return {"n": 0, "mean": None, "std": None, "median": None}
                return {
                    "n": int(arr.size),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                }

            halo_stats = _stats(agg_red_halo)
            non_stats = _stats(agg_red_non_halo)
            effect: Dict[str, Any] = {}
            if halo_stats.get("mean") is not None and non_stats.get("mean") is not None:
                try:
                    mean_h = float(halo_stats["mean"])
                    mean_n = float(non_stats["mean"])
                    effect["mean_diff"] = float(mean_h - mean_n)
                    effect["mean_ratio"] = float(mean_h / max(mean_n, 1e-12))
                except Exception:
                    pass

            # Bootstrap CIs over channels (quick diagnostic; does not re-bootstrap tokens).
            try:
                rng = np.random.default_rng(0)
                halo_arr = np.asarray(agg_red_halo, dtype=np.float64)
                halo_arr = halo_arr[np.isfinite(halo_arr)]
                non_arr = np.asarray(agg_red_non_halo, dtype=np.float64)
                non_arr = non_arr[np.isfinite(non_arr)]

                max_bs = int(supernode_cfg.get("redundancy_bootstrap_max", 5000) or 5000)
                n_boot = int(supernode_cfg.get("redundancy_bootstrap_samples", 200) or 200)
                max_bs = max(100, max_bs)
                n_boot = max(50, n_boot)

                if halo_arr.size > max_bs:
                    halo_arr = rng.choice(halo_arr, size=max_bs, replace=False)
                if non_arr.size > max_bs:
                    non_arr = rng.choice(non_arr, size=max_bs, replace=False)

                if halo_arr.size > 10 and non_arr.size > 10:
                    diffs = np.empty(n_boot, dtype=np.float64)
                    ratios = np.empty(n_boot, dtype=np.float64)
                    for b in range(n_boot):
                        mh = float(rng.choice(halo_arr, size=halo_arr.size, replace=True).mean())
                        mn = float(rng.choice(non_arr, size=non_arr.size, replace=True).mean())
                        diffs[b] = mh - mn
                        ratios[b] = mh / max(mn, 1e-12)
                    effect["bootstrap"] = {
                        "n_boot": int(n_boot),
                        "max_samples_per_group": int(max_bs),
                        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                        "ratio_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                    }
            except Exception:
                pass

            agg_out: Dict[str, Any] = {
                "redundancy_to_core": {
                    "halo": halo_stats,
                    "non_halo_sample": non_stats,
                    "effect": effect,
                }
            }

            # Matched random-core baseline (multiple-comparisons control), if available.
            if agg_red_rand_halo or agg_red_rand_non_halo:
                rand_halo_stats = _stats(agg_red_rand_halo)
                rand_non_stats = _stats(agg_red_rand_non_halo)
                rand_effect: Dict[str, Any] = {}
                if rand_halo_stats.get("mean") is not None and rand_non_stats.get("mean") is not None:
                    try:
                        mean_h = float(rand_halo_stats["mean"])
                        mean_n = float(rand_non_stats["mean"])
                        rand_effect["mean_diff"] = float(mean_h - mean_n)
                        rand_effect["mean_ratio"] = float(mean_h / max(mean_n, 1e-12))
                    except Exception:
                        pass

                agg_out["redundancy_to_random_core"] = {
                    "halo": rand_halo_stats,
                    "non_halo_sample": rand_non_stats,
                    "effect": rand_effect,
                }

            results["_aggregate"] = agg_out

        logger.info(f"Computed SCAR protection/connectivity scores for {len(results)} layers")
        return results

    def analyze_halo_vs_nonhalo_redundancy(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        halo_fraction: float = 0.10,
        num_samples: int = 8,
        max_length: int = 256,
        sample_pairs: int = 2000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Paper-aligned halo redundancy analysis using the loss-relevant contribution signal.

        We compare redundancy between three groups (per layer), then aggregate across layers:
          1) **Halo-Halo**: both channels in the halo (high Conn to supernode write pattern)
          2) **Non-halo**: both channels outside halo and outside supernodes
          3) **Cross**: one halo channel and one non-halo channel

        Signal:
          \(q_i = u_i s_i\) where \(u\) is the FFN post-gate activation (down_proj input) and
          \(s=\nabla_u \mathcal{L}\) (down_proj grad_input[0]).

        Redundancy proxy:
          - \(\rho_{ij}=\mathrm{corr}(q_i,q_j)\) over calibration tokens
          - Optional **positive-only** redundancy: \(\rho^+_{ij}=\max(0,\rho_{ij})\)
          - \(\mathrm{Red}(i,j) = -\tfrac12 \log(1-(\rho^+_{ij})^2)\)

        Notes:
        - Supernodes are identified by `scar_loss_proxy` when available (default definition).
        - Halo membership is identified by Conn overlap with the aggregated supernode write pattern
          (same as `compute_supernode_connectivity_pruning_score`).

        Returns:
          Dict with:
            - per_layer: per-layer group stats
            - aggregate: aggregated stats across layers
        """
        logger.info("=" * 60)
        logger.info("ANALYZING HALO vs NON-HALO REDUNDANCY (q-signal)")
        logger.info("=" * 60)

        eps = 1e-8
        halo_cfg = getattr(self.config, "halo_analysis", {}) or {}
        if hasattr(halo_cfg, "__dict__"):
            halo_cfg = vars(halo_cfg)

        # Use positive-only redundancy when configured (matches SCAR ablation)
        supernode_cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
        # Default to positive-only redundancy (anti-correlation does NOT count as redundancy),
        # matching the default definition; can be disabled for sensitivity analyses.
        positive_redundancy = bool(supernode_cfg.get("positive_redundancy", True))
        if positive_redundancy:
            logger.info("  Redundancy: using positive-only correlation (anti-correlation does NOT count as redundancy)")

        # Respect optional config bounds
        max_pairs_per_group = int(halo_cfg.get("max_pairs_per_group", sample_pairs))
        pairs_per_group = max(1, min(int(sample_pairs), max_pairs_per_group))
        max_group_channels = int(halo_cfg.get("max_group_channels", 512))

        # Prefer the same calibration texts used in SCAR / importance computation
        calibration_texts: List[str] = []
        if getattr(self.config, "importance_computation_texts", None):
            calibration_texts = list(self.config.importance_computation_texts)
        elif getattr(self, "dataset", None) is not None and hasattr(self.dataset, "texts"):
            calibration_texts = list(self.dataset.texts)
        if not calibration_texts:
            # Last-resort fallback (keeps the analysis runnable in isolation)
            calibration_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models require careful tuning.",
                "In the beginning, there was darkness, then light.",
                "The stock market experienced significant volatility.",
                "Scientists discovered a new species of deep-sea fish.",
                "The conference will be held in San Francisco next month.",
                "Programming languages continue to evolve.",
                "Climate change poses challenges for future generations.",
            ]

        if not calibration_texts:
            logger.warning("No calibration texts available for halo redundancy analysis")
            return {}

        num_samples = max(1, int(num_samples))
        calibration_texts = calibration_texts[: min(num_samples, len(calibration_texts))]

        # Underlying HF model for module lookup / hook registration
        hf_model: nn.Module = self.model
        if hasattr(hf_model, "model"):
            hf_model = getattr(hf_model, "model")
        module_dict = dict(hf_model.named_modules())

        # Only analyze FFN down_proj layers (intermediate channels)
        layer_names = [ln for ln in scar_scores.keys() if "mlp.down_proj" in ln]
        if not layer_names:
            logger.warning("No down_proj layers found in scar_scores for halo redundancy analysis")
            return {}

        # Helper: sample pair positions (indices into a group of size n)
        def sample_pairs_pos(n: int, p: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if n < 2 or p <= 0:
                return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
            i = torch.randint(low=0, high=n, size=(p,), dtype=torch.long)
            j = torch.randint(low=0, high=n, size=(p,), dtype=torch.long)
            # ensure i != j (resample j where equal)
            same = i == j
            tries = 0
            while same.any() and tries < 10:
                j[same] = torch.randint(low=0, high=n, size=(int(same.sum().item()),), dtype=torch.long)
                same = i == j
                tries += 1
            # if still equal, shift deterministically
            if same.any():
                j[same] = (j[same] + 1) % n
            return i, j

        # ------------------------------------------------------------------
        # Phase 1: Per-layer supernodes + connectivity halo (weights-only) and pair plans
        # ------------------------------------------------------------------
        plan: Dict[str, Dict[str, Any]] = {}
        for layer_name in layer_names:
            layer_metrics = scar_scores.get(layer_name, {}) or {}
            lp = layer_metrics.get("scar_loss_proxy")
            if lp is None:
                lp = layer_metrics.get("scar_activation_power")
            if lp is None:
                continue

            lp_cpu = lp.detach().float().cpu()
            m = int(lp_cpu.numel())
            if m <= 0:
                continue

            module = module_dict.get(layer_name)
            if module is None or not hasattr(module, "weight"):
                logger.warning(f"Halo redundancy: could not resolve module/weight for {layer_name}")
                continue

            # Identify supernodes by LP (default definition)
            num_supernodes = max(1, int(supernode_fraction * m))
            _, super_idx = torch.topk(lp_cpu, k=num_supernodes, largest=True)
            super_idx = super_idx.long()
            super_mask = torch.zeros(m, dtype=torch.bool)
            super_mask[super_idx] = True

            # Compute Conn_i from down_proj weights (same definition as SCAR-Conn):
            # Conn_i := sum_{h in TopK(a)} |v_i[h]| / ||v_i||_1  (fraction of write mass on core support)
            W = module.weight.detach().float().cpu()  # [hidden_dim, m]
            abs_W = W.abs()
            a = abs_W[:, super_idx].sum(dim=1)  # [hidden_dim]
            v_norm = abs_W.sum(dim=0) + eps  # [m]

            hidden_dim = int(abs_W.shape[0])
            k = int(supernode_cfg.get("connectivity_topk", 256))
            mass_frac = supernode_cfg.get("connectivity_mass_fraction", None)
            a_sorted, a_order = torch.sort(a, descending=True)
            if mass_frac is not None:
                try:
                    mf = float(mass_frac)
                except Exception:
                    mf = None
                if mf is not None and 0.0 < mf < 1.0 and a_sorted.numel() > 0:
                    cdf = torch.cumsum(a_sorted, dim=0)
                    total = float(cdf[-1].item())
                    if total > 0:
                        target = mf * total
                        k = int(torch.searchsorted(cdf, torch.tensor(target)).item()) + 1
            k = max(1, min(int(k), hidden_dim))
            core_idx = a_order[:k]
            conn = abs_W.index_select(0, core_idx).sum(dim=0) / v_norm
            conn = conn.clamp(0.0, 1.0)

            non_super_idx = (~super_mask).nonzero(as_tuple=True)[0]
            if non_super_idx.numel() < 2:
                continue
            num_halo = max(1, int(halo_fraction * non_super_idx.numel()))
            _, halo_rel = torch.topk(conn[non_super_idx], k=num_halo, largest=True)
            halo_idx = non_super_idx[halo_rel].long()

            halo_mask = torch.zeros(m, dtype=torch.bool)
            halo_mask[halo_idx] = True
            non_halo_idx = ((~super_mask) & (~halo_mask)).nonzero(as_tuple=True)[0].long()
            if halo_idx.numel() < 2 or non_halo_idx.numel() < 2:
                continue

            # Subsample channels to keep the analysis lightweight and comparable across layers.
            halo_sel = halo_idx
            if halo_sel.numel() > max_group_channels:
                perm = torch.randperm(halo_sel.numel())
                halo_sel = halo_sel[perm[:max_group_channels]]

            non_halo_target = min(int(halo_sel.numel()), int(non_halo_idx.numel()), max_group_channels)
            if non_halo_target < 2:
                continue
            perm = torch.randperm(non_halo_idx.numel())
            non_halo_sel = non_halo_idx[perm[:non_halo_target]]

            # If the halo selection was larger, trim to match (keeps pair sampling symmetric).
            if halo_sel.numel() > non_halo_sel.numel():
                halo_sel = halo_sel[: non_halo_sel.numel()]

            H = int(halo_sel.numel())
            NH = int(non_halo_sel.numel())
            if H < 2 or NH < 2:
                continue

            P = int(min(pairs_per_group, H * (H - 1) // 2, NH * (NH - 1) // 2))
            if P <= 0:
                continue

            hh_i_cpu, hh_j_cpu = sample_pairs_pos(H, P)
            nn_i_cpu, nn_j_cpu = sample_pairs_pos(NH, P)
            cross_h_cpu = torch.randint(low=0, high=H, size=(P,), dtype=torch.long)
            cross_n_cpu = torch.randint(low=0, high=NH, size=(P,), dtype=torch.long)

            plan[layer_name] = {
                "num_supernodes": int(num_supernodes),
                "m": int(m),
                "halo_idx_cpu": halo_sel,
                "nonhalo_idx_cpu": non_halo_sel,
                "hh_i_cpu": hh_i_cpu,
                "hh_j_cpu": hh_j_cpu,
                "nn_i_cpu": nn_i_cpu,
                "nn_j_cpu": nn_j_cpu,
                "cross_h_cpu": cross_h_cpu,
                "cross_n_cpu": cross_n_cpu,
                # device-side cached tensors
                "halo_idx": None,
                "nonhalo_idx": None,
                "hh_i": None,
                "hh_j": None,
                "nn_i": None,
                "nn_j": None,
                "cross_h": None,
                "cross_n": None,
                # streaming sums
                "sum_q_halo": None,
                "sum_q2_halo": None,
                "sum_q_nonhalo": None,
                "sum_q2_nonhalo": None,
                "sum_qij_hh": None,
                "sum_qij_nn": None,
                "sum_qij_cross": None,
                "count": 0,
            }

        if not plan:
            logger.warning("Halo redundancy: no eligible layers after filtering; skipping")
            return {}

        # ------------------------------------------------------------------
        # Phase 2: Calibration passes (forward+backward) to accumulate q correlations
        # ------------------------------------------------------------------
        hooks: List[Any] = []

        def make_hooks(name: str):
            def fwd_hook(mod: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
                if not inputs or inputs[0] is None:
                    return
                mod._halo_last_u = inputs[0].detach()

            def bwd_hook(mod: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
                st = plan.get(name)
                if st is None:
                    return
                if not grad_input or grad_input[0] is None:
                    return
                if not hasattr(mod, "_halo_last_u"):
                    return

                u = mod._halo_last_u
                delattr(mod, "_halo_last_u")
                g_u = grad_input[0]

                # Flatten to [N_tokens, dim]
                u_flat = u.reshape(-1, u.shape[-1]) if u.ndim > 2 else u.reshape(-1, u.shape[-1])
                g_u_flat = g_u.reshape(-1, g_u.shape[-1]) if g_u.ndim > 2 else g_u.reshape(-1, g_u.shape[-1])
                if u_flat.shape != g_u_flat.shape or u_flat.numel() == 0:
                    return

                # Move cached indices/pairs to the correct device once
                dev = u_flat.device
                if st["halo_idx"] is None or st["halo_idx"].device != dev:
                    st["halo_idx"] = st["halo_idx_cpu"].to(device=dev)
                if st["nonhalo_idx"] is None or st["nonhalo_idx"].device != dev:
                    st["nonhalo_idx"] = st["nonhalo_idx_cpu"].to(device=dev)
                if st["hh_i"] is None or st["hh_i"].device != dev:
                    st["hh_i"] = st["hh_i_cpu"].to(device=dev)
                    st["hh_j"] = st["hh_j_cpu"].to(device=dev)
                    st["nn_i"] = st["nn_i_cpu"].to(device=dev)
                    st["nn_j"] = st["nn_j_cpu"].to(device=dev)
                    st["cross_h"] = st["cross_h_cpu"].to(device=dev)
                    st["cross_n"] = st["cross_n_cpu"].to(device=dev)

                halo_idx = st["halo_idx"]
                nonhalo_idx = st["nonhalo_idx"]
                idx_union = torch.cat([halo_idx, nonhalo_idx], dim=0)  # [H + NH]

                try:
                    u_sel = u_flat.index_select(1, idx_union).float()
                    s_sel = g_u_flat.index_select(1, idx_union).float()
                except Exception:
                    return

                q_sel = u_sel * s_sel
                H = int(halo_idx.numel())
                q_h = q_sel[:, :H]
                q_n = q_sel[:, H:]
                N = int(q_sel.shape[0])
                if N <= 0:
                    return

                # Initialize sums on first batch
                if st["sum_q_halo"] is None:
                    st["sum_q_halo"] = torch.zeros(H, device=dev, dtype=torch.float32)
                    st["sum_q2_halo"] = torch.zeros_like(st["sum_q_halo"])
                    st["sum_q_nonhalo"] = torch.zeros(q_n.shape[1], device=dev, dtype=torch.float32)
                    st["sum_q2_nonhalo"] = torch.zeros_like(st["sum_q_nonhalo"])
                    P = int(st["hh_i"].numel())
                    st["sum_qij_hh"] = torch.zeros(P, device=dev, dtype=torch.float32)
                    st["sum_qij_nn"] = torch.zeros(P, device=dev, dtype=torch.float32)
                    st["sum_qij_cross"] = torch.zeros(P, device=dev, dtype=torch.float32)

                st["sum_q_halo"] += q_h.sum(dim=0)
                st["sum_q2_halo"] += (q_h * q_h).sum(dim=0)
                st["sum_q_nonhalo"] += q_n.sum(dim=0)
                st["sum_q2_nonhalo"] += (q_n * q_n).sum(dim=0)

                # Pair cross-products (vectorized)
                hh_i = st["hh_i"]
                hh_j = st["hh_j"]
                nn_i = st["nn_i"]
                nn_j = st["nn_j"]
                ch = st["cross_h"]
                cn = st["cross_n"]

                if hh_i.numel() > 0:
                    qi = q_h.index_select(1, hh_i)
                    qj = q_h.index_select(1, hh_j)
                    st["sum_qij_hh"] += (qi * qj).sum(dim=0)

                    qi = q_n.index_select(1, nn_i)
                    qj = q_n.index_select(1, nn_j)
                    st["sum_qij_nn"] += (qi * qj).sum(dim=0)

                    qi = q_h.index_select(1, ch)
                    qj = q_n.index_select(1, cn)
                    st["sum_qij_cross"] += (qi * qj).sum(dim=0)

                st["count"] += N

            return fwd_hook, bwd_hook

        for layer_name, module in module_dict.items():
            if layer_name not in plan:
                continue
            fwd, bwd = make_hooks(layer_name)
            hooks.append(module.register_forward_hook(fwd))
            hooks.append(module.register_full_backward_hook(bwd))

        self.model.eval()
        device = torch.device(self.config.device)

        try:
            for idx, text in enumerate(calibration_texts):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(max_length),
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                labels = inputs["input_ids"].clone()
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
                labels[labels == pad_token_id] = -100
                inputs["labels"] = labels

                self.model.zero_grad(set_to_none=True)
                out = self.model(**inputs)
                loss = out.loss
                loss.backward()

                if (idx + 1) % 1 == 0:
                    logger.info(f"  Halo q-stats: processed {idx+1}/{len(calibration_texts)} samples, loss={loss.item():.4f}")
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Phase 3: Compute redundancy distributions and aggregate across layers
        # ------------------------------------------------------------------
        per_layer: Dict[str, Dict[str, Any]] = {}
        agg_vals: Dict[str, List[float]] = {"halo_halo": [], "non_halo": [], "cross": []}

        def corr_to_red(corr: torch.Tensor) -> torch.Tensor:
            corr = corr.clamp(-0.9999, 0.9999)
            if positive_redundancy:
                corr = torch.clamp(corr, min=0.0)
            rho_sq = (corr * corr).clamp(0.0, 0.9999)
            return (-0.5 * torch.log(1.0 - rho_sq + eps)).float()

        for layer_name, st in plan.items():
            N = int(st.get("count", 0))
            if N <= 1 or st["sum_qij_hh"] is None:
                continue

            sum_q_h = st["sum_q_halo"].detach().cpu()
            sum_q2_h = st["sum_q2_halo"].detach().cpu()
            sum_q_n = st["sum_q_nonhalo"].detach().cpu()
            sum_q2_n = st["sum_q2_nonhalo"].detach().cpu()

            mean_h = sum_q_h / float(N)
            mean_n = sum_q_n / float(N)
            var_h = (sum_q2_h / float(N)) - (mean_h * mean_h)
            var_n = (sum_q2_n / float(N)) - (mean_n * mean_n)
            std_h = torch.sqrt(torch.clamp(var_h, min=eps))
            std_n = torch.sqrt(torch.clamp(var_n, min=eps))

            hh_i = st["hh_i_cpu"]
            hh_j = st["hh_j_cpu"]
            nn_i = st["nn_i_cpu"]
            nn_j = st["nn_j_cpu"]
            ch = st["cross_h_cpu"]
            cn = st["cross_n_cpu"]

            # E[q_i q_j]
            e_hh = st["sum_qij_hh"].detach().cpu() / float(N)
            e_nn = st["sum_qij_nn"].detach().cpu() / float(N)
            e_cn = st["sum_qij_cross"].detach().cpu() / float(N)

            # corr (halo-halo)
            cov = e_hh - (mean_h[hh_i] * mean_h[hh_j])
            corr_hh = cov / (std_h[hh_i] * std_h[hh_j] + eps)
            red_hh = corr_to_red(corr_hh)

            # corr (non-halo, non-halo)
            cov = e_nn - (mean_n[nn_i] * mean_n[nn_j])
            corr_nn = cov / (std_n[nn_i] * std_n[nn_j] + eps)
            red_nn = corr_to_red(corr_nn)

            # corr (cross)
            cov = e_cn - (mean_h[ch] * mean_n[cn])
            corr_cn = cov / (std_h[ch] * std_n[cn] + eps)
            red_cn = corr_to_red(corr_cn)

            def stats(x: torch.Tensor) -> Dict[str, float]:
                if x.numel() == 0:
                    return {"mean": 0.0, "std": 0.0, "median": 0.0, "count": 0}
                return {
                    "mean": float(x.mean().item()),
                    "std": float(x.std(unbiased=False).item()),
                    "median": float(x.median().item()),
                    "count": int(x.numel()),
                }

            per_layer[layer_name] = {
                "num_supernodes": int(st.get("num_supernodes", 0)),
                "num_halo": int(st["halo_idx_cpu"].numel()),
                "num_non_halo": int(st["nonhalo_idx_cpu"].numel()),
                "halo_halo": stats(red_hh),
                "non_halo": stats(red_nn),
                "cross": stats(red_cn),
            }

            agg_vals["halo_halo"].extend(red_hh.tolist())
            agg_vals["non_halo"].extend(red_nn.tolist())
            agg_vals["cross"].extend(red_cn.tolist())

        aggregate_stats: Dict[str, Dict[str, Any]] = {}
        for group, vals in agg_vals.items():
            if not vals:
                aggregate_stats[group] = {"mean": 0.0, "std": 0.0, "median": 0.0, "count": 0}
                continue
            arr = np.asarray(vals, dtype=np.float64)
            aggregate_stats[group] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "count": int(arr.size),
            }

        logger.info("\nHALO vs NON-HALO REDUNDANCY SUMMARY (q-signal):")
        logger.info(f"  Halo-Halo:     mean={aggregate_stats['halo_halo']['mean']:.4f}")
        logger.info(f"  Non-halo:      mean={aggregate_stats['non_halo']['mean']:.4f}")
        logger.info(f"  Cross-group:   mean={aggregate_stats['cross']['mean']:.4f}")

        return {
            "signal": "q",
            "positive_redundancy": positive_redundancy,
            "pairs_per_group": pairs_per_group,
            "max_group_channels": max_group_channels,
            "per_layer": per_layer,
            "aggregate": aggregate_stats,
        }

    def visualize_halo_nonhalo_metrics_by_layer(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        halo_fraction: float = 0.10,
        plots_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Visualize mean metrics (redundancy, MI, RQ, activation) for supernode, halo,
        and non-halo groups vs layer number.

        Creates summary plots showing how each group's metrics vary across layers.

        Args:
            scar_scores: SCAR metrics from compute_scar_metrics
            supernode_fraction: Fraction of neurons to consider as supernodes
            halo_fraction: Fraction of non-supernodes to consider as halo
            plots_dir: Directory to save plots

        Returns:
            Dictionary with per-layer group statistics
        """
        logger.info("=" * 60)
        logger.info("VISUALIZING HALO vs NON-HALO METRICS BY LAYER")
        logger.info("=" * 60)

        if plots_dir is None:
            plots_dir = Path(getattr(self.config, "plots_dir", "./plots"))
        plots_dir = Path(plots_dir)
        summary_dir = plots_dir / "supernode_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Get HF model
        hf_model = self.wrapped_model._model if hasattr(self.wrapped_model, "_model") else self.model

        # Collect metrics for each group across layers
        layer_indices = []
        supernode_stats = {"activation": [], "rq": [], "mi": [], "redundancy": [], "loss_proxy": []}
        halo_stats = {"activation": [], "rq": [], "mi": [], "redundancy": [], "loss_proxy": []}
        nonhalo_stats = {"activation": [], "rq": [], "mi": [], "redundancy": [], "loss_proxy": []}

        # Sort layers by index
        sorted_layers = sorted(
            [layer_name for layer_name in scar_scores.keys() if "down_proj" in layer_name],
            key=lambda x: int(x.split("layers.")[-1].split(".")[0]) if "layers." in x else 0,
        )

        for layer_name in sorted_layers:
            layer_metrics = scar_scores[layer_name]

            if "scar_activation_power" not in layer_metrics:
                continue

            # Extract layer index
            try:
                layer_idx = int(layer_name.split("layers.")[-1].split(".")[0])
            except (ValueError, IndexError):
                layer_idx = len(layer_indices)
            layer_indices.append(layer_idx)

            # Get down_proj weights for connectivity
            down_proj_weight = None
            for name, module in hf_model.named_modules():
                if name == layer_name and hasattr(module, "weight"):
                    down_proj_weight = module.weight.data.float().cpu()
                    break

            if down_proj_weight is None:
                # Skip if can't get weights
                for key in supernode_stats.keys():
                    supernode_stats[key].append(float("nan"))
                    halo_stats[key].append(float("nan"))
                    nonhalo_stats[key].append(float("nan"))
                continue

            hidden_dim, intermediate_dim = down_proj_weight.shape

            # Get metric tensors from SCAR scores
            activation_power = layer_metrics.get("scar_activation_power", torch.zeros(intermediate_dim)).float().cpu()
            loss_proxy = layer_metrics.get("scar_loss_proxy", torch.zeros(intermediate_dim)).float().cpu()
            curvature = layer_metrics.get("scar_curvature", torch.zeros(intermediate_dim)).float().cpu()
            taylor = layer_metrics.get("scar_taylor", torch.zeros(intermediate_dim)).float().cpu()

            # Get RQ, MI, redundancy - try multiple sources
            # First try importance_scores, then scar_scores, then fallback
            def get_metric(metric_name, fallback_size):
                """Helper to get metric from various sources."""
                # Try importance_scores first
                val = self.importance_scores.get(layer_name, {}).get(metric_name)
                if val is None:
                    # Try scar_scores
                    val = layer_metrics.get(metric_name)
                if val is None:
                    return torch.zeros(fallback_size)
                if torch.is_tensor(val):
                    return val.float().cpu()
                return torch.zeros(fallback_size)

            rq = get_metric("rayleigh_quotient", intermediate_dim)
            mi = get_metric("gaussian_mi_analytic", intermediate_dim)
            redundancy = get_metric("average_redundancy", intermediate_dim)

            # If RQ/MI/redundancy are empty, use SCAR metrics as proxies
            if rq.sum() == 0:
                rq = curvature  # Curvature is related to RQ
            if mi.sum() == 0:
                mi = taylor  # Taylor score relates to information content

            # Identify supernodes (default: top by loss proxy when available)
            supernode_metric = loss_proxy if loss_proxy is not None and loss_proxy.numel() == intermediate_dim else activation_power
            num_supernodes = max(1, int(supernode_fraction * intermediate_dim))
            _, supernode_indices = torch.topk(supernode_metric, num_supernodes)
            supernode_mask = torch.zeros(intermediate_dim, dtype=torch.bool)
            supernode_mask[supernode_indices] = True

            # Identify halo (high connectivity to supernodes among non-supernodes)
            non_supernode_mask = ~supernode_mask
            non_supernode_indices = non_supernode_mask.nonzero(as_tuple=True)[0]
            # Conn using overlap with aggregated supernode write pattern
            abs_W = down_proj_weight.abs()
            a = abs_W[:, supernode_indices].sum(dim=1)
            a_norm = a.sum() + 1e-8
            v_norm = abs_W.sum(dim=0) + 1e-8
            conn_num = (abs_W * a.unsqueeze(1)).sum(dim=0)
            conn = (conn_num / (v_norm * a_norm + 1e-8)).clamp(0.0, 1.0)
            non_supernode_connection = conn[non_supernode_indices]

            num_halo = max(1, int(halo_fraction * len(non_supernode_indices)))
            _, halo_relative_indices = torch.topk(non_supernode_connection, num_halo)
            halo_indices = non_supernode_indices[halo_relative_indices]

            halo_mask = torch.zeros(intermediate_dim, dtype=torch.bool)
            halo_mask[halo_indices] = True

            # Non-halo = not supernode and not halo
            non_halo_mask = non_supernode_mask & ~halo_mask

            # Compute mean metrics for each group
            def safe_mean(tensor, mask):
                if mask.sum() == 0:
                    return float("nan")
                return float(tensor[mask].mean().item())

            supernode_stats["activation"].append(safe_mean(activation_power, supernode_mask))
            supernode_stats["rq"].append(safe_mean(rq, supernode_mask))
            supernode_stats["mi"].append(safe_mean(mi, supernode_mask))
            supernode_stats["redundancy"].append(safe_mean(redundancy, supernode_mask))
            supernode_stats["loss_proxy"].append(safe_mean(loss_proxy, supernode_mask))

            halo_stats["activation"].append(safe_mean(activation_power, halo_mask))
            halo_stats["rq"].append(safe_mean(rq, halo_mask))
            halo_stats["mi"].append(safe_mean(mi, halo_mask))
            halo_stats["redundancy"].append(safe_mean(redundancy, halo_mask))
            halo_stats["loss_proxy"].append(safe_mean(loss_proxy, halo_mask))

            nonhalo_stats["activation"].append(safe_mean(activation_power, non_halo_mask))
            nonhalo_stats["rq"].append(safe_mean(rq, non_halo_mask))
            nonhalo_stats["mi"].append(safe_mean(mi, non_halo_mask))
            nonhalo_stats["redundancy"].append(safe_mean(redundancy, non_halo_mask))
            nonhalo_stats["loss_proxy"].append(safe_mean(loss_proxy, non_halo_mask))

        if not layer_indices:
            logger.warning("No layers found for halo/non-halo analysis")
            return {}

        # Log data summary for debugging
        logger.info(f"  Collected data for {len(layer_indices)} layers")
        for stat_name, stats in [("supernode", supernode_stats), ("halo", halo_stats), ("nonhalo", nonhalo_stats)]:
            valid_count = sum(1 for v in stats["activation"] if not np.isnan(v) if v != 0)
            logger.info(f"    {stat_name}: {valid_count}/{len(stats['activation'])} valid activation values")

        # Create visualization
        import matplotlib.pyplot as plt

        metrics_to_plot = [
            ("activation", "Activation Power", True),
            ("loss_proxy", "Loss Proxy", True),
            ("rq", "Curvature/RQ", False),
            ("mi", "Taylor/MI", False),
            ("redundancy", "Redundancy", False),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric_key, metric_name, use_log) in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
            ax = axes[idx]

            supernode_vals = np.array(supernode_stats[metric_key])
            halo_vals = np.array(halo_stats[metric_key])
            nonhalo_vals = np.array(nonhalo_stats[metric_key])

            # Filter out nan values
            valid_mask = ~(np.isnan(supernode_vals) | np.isnan(halo_vals) | np.isnan(nonhalo_vals))
            valid_layers = np.array(layer_indices)[valid_mask]

            if len(valid_layers) == 0:
                ax.set_title(f"{metric_name}\n(No data)")
                continue

            ax.plot(valid_layers, supernode_vals[valid_mask], "o-", color="coral", linewidth=2, markersize=6, label="Supernodes")
            ax.plot(valid_layers, halo_vals[valid_mask], "s-", color="steelblue", linewidth=2, markersize=5, label="Halo")
            ax.plot(valid_layers, nonhalo_vals[valid_mask], "^-", color="forestgreen", linewidth=2, markersize=5, label="Non-halo")

            ax.set_xlabel("Layer Index")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} by Group")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if use_log and np.nanmin(supernode_vals) > 0:
                ax.set_yscale("log")

        # Use the last subplot for a ratio plot
        ax = axes[-1]
        supernode_act = np.array(supernode_stats["activation"])
        nonhalo_act = np.array(nonhalo_stats["activation"])
        valid_mask = ~(np.isnan(supernode_act) | np.isnan(nonhalo_act) | (nonhalo_act == 0))
        valid_layers = np.array(layer_indices)[valid_mask]

        if len(valid_layers) > 0:
            ratio = supernode_act[valid_mask] / nonhalo_act[valid_mask]
            ax.bar(valid_layers, ratio, color="purple", alpha=0.7)
            ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Equal")
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Supernode / Non-halo Ratio")
            ax.set_title("Activation Power Ratio")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Supernode vs Halo vs Non-Halo Metrics Across Layers", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = summary_dir / "halo_nonhalo_metrics_by_layer.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved halo/non-halo metrics plot to {save_path}")

        return {
            "layer_indices": layer_indices,
            "supernode_stats": supernode_stats,
            "halo_stats": halo_stats,
            "nonhalo_stats": nonhalo_stats,
        }

    def compute_supernode_outlier_scores(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        plots_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Compute z-scores showing how much of an outlier supernodes are vs layer number.

        For each layer, computes:
        - Mean activation of supernodes
        - Mean and std of all neurons
        - Z-score = (supernode_mean - population_mean) / population_std

        Args:
            scar_scores: SCAR metrics from compute_scar_metrics
            supernode_fraction: Fraction of neurons to consider as supernodes
            plots_dir: Directory to save plots

        Returns:
            Dictionary with z-scores and statistics per layer
        """
        logger.info("=" * 60)
        logger.info("COMPUTING SUPERNODE OUTLIER Z-SCORES BY LAYER")
        logger.info("=" * 60)

        if plots_dir is None:
            plots_dir = Path(getattr(self.config, "plots_dir", "./plots"))
        plots_dir = Path(plots_dir)
        summary_dir = plots_dir / "supernode_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Collect z-scores across layers
        layer_indices = []
        z_scores_activation = []
        z_scores_loss_proxy = []
        z_scores_max = []  # Max activation z-score (single neuron)
        outlier_ratios = []  # Ratio of supernode mean to population mean

        # Sort layers by index
        sorted_layers = sorted(
            [layer_name for layer_name in scar_scores.keys() if "down_proj" in layer_name],
            key=lambda x: int(x.split("layers.")[-1].split(".")[0]) if "layers." in x else 0,
        )

        for layer_name in sorted_layers:
            layer_metrics = scar_scores[layer_name]

            if "scar_activation_power" not in layer_metrics:
                continue

            # Extract layer index
            try:
                layer_idx = int(layer_name.split("layers.")[-1].split(".")[0])
            except (ValueError, IndexError):
                layer_idx = len(layer_indices)
            layer_indices.append(layer_idx)

            # Get metric tensors
            activation_power = layer_metrics.get("scar_activation_power", torch.zeros(1)).float().cpu()
            loss_proxy = layer_metrics.get("scar_loss_proxy", torch.zeros(1)).float().cpu()

            intermediate_dim = activation_power.numel()

            # Identify supernodes
            num_supernodes = max(1, int(supernode_fraction * intermediate_dim))
            _, supernode_indices = torch.topk(activation_power, num_supernodes)

            # Compute z-scores for activation
            pop_mean = activation_power.mean().item()
            pop_std = activation_power.std().item()
            supernode_mean = activation_power[supernode_indices].mean().item()
            max_activation = activation_power.max().item()

            if pop_std > 1e-8:
                z_act = (supernode_mean - pop_mean) / pop_std
                z_max = (max_activation - pop_mean) / pop_std
            else:
                z_act = 0.0
                z_max = 0.0

            z_scores_activation.append(z_act)
            z_scores_max.append(z_max)

            # Ratio (how many times larger)
            if pop_mean > 1e-8:
                outlier_ratios.append(supernode_mean / pop_mean)
            else:
                outlier_ratios.append(1.0)

            # Z-score for loss proxy
            pop_mean_lp = loss_proxy.mean().item()
            pop_std_lp = loss_proxy.std().item()
            supernode_mean_lp = loss_proxy[supernode_indices].mean().item()

            if pop_std_lp > 1e-8:
                z_lp = (supernode_mean_lp - pop_mean_lp) / pop_std_lp
            else:
                z_lp = 0.0

            z_scores_loss_proxy.append(z_lp)

        if not layer_indices:
            logger.warning("No layers found for outlier score analysis")
            return {}

        # Create visualization
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Z-scores for activation power
        ax = axes[0, 0]
        ax.bar(layer_indices, z_scores_activation, color="coral", alpha=0.7, label="Supernode Mean")
        ax.plot(layer_indices, z_scores_max, "k^-", markersize=8, label="Max Neuron")
        ax.axhline(y=2.0, color="orange", linestyle="--", linewidth=2, label="z=2 (95%)")
        ax.axhline(y=3.0, color="red", linestyle="--", linewidth=2, label="z=3 (99.7%)")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Z-Score")
        ax.set_title("Supernode Activation Z-Score by Layer")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Z-scores for loss proxy
        ax = axes[0, 1]
        ax.bar(layer_indices, z_scores_loss_proxy, color="steelblue", alpha=0.7)
        ax.axhline(y=2.0, color="orange", linestyle="--", linewidth=2, label="z=2")
        ax.axhline(y=3.0, color="red", linestyle="--", linewidth=2, label="z=3")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Z-Score")
        ax.set_title("Supernode Loss Proxy Z-Score by Layer")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Outlier ratio
        ax = axes[1, 0]
        ax.bar(layer_indices, outlier_ratios, color="purple", alpha=0.7)
        ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1)
        ax.axhline(y=10.0, color="orange", linestyle="--", linewidth=2, label="10x larger")
        ax.axhline(y=100.0, color="red", linestyle="--", linewidth=2, label="100x larger")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Ratio (Supernode Mean / Population Mean)")
        ax.set_title("Supernode Activation Magnitude Ratio")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 4: Summary heatmap-style
        ax = axes[1, 1]
        combined = np.array([z_scores_activation, z_scores_loss_proxy])
        im = ax.imshow(combined, aspect="auto", cmap="YlOrRd")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Activation Z", "Loss Proxy Z"])
        ax.set_xlabel("Layer Index")
        ax.set_xticks(range(0, len(layer_indices), max(1, len(layer_indices) // 10)))
        ax.set_xticklabels([layer_indices[i] for i in range(0, len(layer_indices), max(1, len(layer_indices) // 10))])
        ax.set_title("Outlier Strength Heatmap")
        plt.colorbar(im, ax=ax, label="Z-Score")

        plt.suptitle("Supernode Outlier Analysis Across Layers", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = summary_dir / "supernode_outlier_zscores.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved supernode outlier z-scores plot to {save_path}")

        # Log summary
        logger.info("\nSUPERNODE OUTLIER SUMMARY:")
        logger.info(f"  Mean Z-score (activation): {np.mean(z_scores_activation):.2f}")
        logger.info(f"  Max Z-score (activation): {np.max(z_scores_activation):.2f} at layer {layer_indices[np.argmax(z_scores_activation)]}")
        logger.info(f"  Mean outlier ratio: {np.mean(outlier_ratios):.1f}x")
        logger.info(f"  Max outlier ratio: {np.max(outlier_ratios):.1f}x at layer {layer_indices[np.argmax(outlier_ratios)]}")

        return {
            "layer_indices": layer_indices,
            "z_scores_activation": z_scores_activation,
            "z_scores_loss_proxy": z_scores_loss_proxy,
            "z_scores_max": z_scores_max,
            "outlier_ratios": outlier_ratios,
        }

    def compute_generalized_importance(
        self,
        num_samples: int = 8,
        max_length: int = 256,
        neighborhood_fraction: float = 0.10,
        propagation_weight: float = 0.3,
        redundancy_penalty: float = 0.5,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute generalized importance scores that don't rely on outlier supernodes.

        This method works for models with continuous activation distributions by:
        1. Computing per-neuron activation magnitude (base importance)
        2. Computing local neighborhood redundancy for EACH neuron
        3. Propagating importance from downstream layers (backward pass)
        4. Combining: importance = base_importance * downstream_influence * (1 - redundancy)

        This generalizes the supernode approach:
        - Instead of binary supernode/halo, uses continuous importance weights
        - Neighborhood defined per-neuron by weight connectivity (not to outliers)
        - Works even without clear outlier structure

        Args:
            num_samples: Number of calibration samples
            max_length: Max sequence length
            neighborhood_fraction: Fraction of neurons to consider as each neuron's "neighborhood"
            propagation_weight: How much downstream importance influences current layer
            redundancy_penalty: How much to penalize redundant neurons (0-1)

        Returns:
            Dictionary of importance scores per layer
        """
        logger.info("=" * 60)
        logger.info("COMPUTING GENERALIZED IMPORTANCE (no outlier assumption)")
        logger.info("=" * 60)

        # Get HF model
        hf_model = self.wrapped_model._model if hasattr(self.wrapped_model, "_model") else self.model

        # Calibration texts
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require careful tuning.",
            "In the beginning, there was darkness, then light.",
            "The stock market experienced significant volatility.",
            "Scientists discovered a new species of deep-sea fish.",
            "The conference will be held in San Francisco next month.",
            "Programming languages continue to evolve with new features.",
            "Climate change poses challenges for future generations.",
        ][:num_samples]

        # Step 1: Collect all MLP layers and their weights
        layer_info = {}
        for name, module in hf_model.named_modules():
            if "mlp.down_proj" in name and hasattr(module, "weight"):
                # Extract layer index
                import re

                match = re.search(r"layers\.(\d+)", name)
                if match:
                    layer_idx = int(match.group(1))
                    layer_info[layer_idx] = {
                        "name": name,
                        "module": module,
                        "weight": module.weight.data.float().cpu(),  # [hidden, intermediate]
                    }

        num_layers = len(layer_info)
        logger.info(f"Found {num_layers} MLP layers")

        if num_layers == 0:
            return {}

        # Step 2: Capture activations for all layers
        layer_activations = {idx: [] for idx in layer_info.keys()}

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                if inputs and inputs[0] is not None:
                    inp = inputs[0].detach().float()
                    if inp.ndim == 3:
                        inp = inp.reshape(-1, inp.shape[-1])
                    layer_activations[layer_idx].append(inp.cpu())

            return hook

        hooks = []
        for layer_idx, info in layer_info.items():
            h = info["module"].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        hf_model.eval()
        with torch.no_grad():
            for text in calibration_texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                try:
                    hf_model(**inputs)
                except Exception:
                    pass

        for h in hooks:
            h.remove()

        # Step 3: Compute base importance and neighborhood redundancy for each layer
        layer_scores = {}

        for layer_idx in sorted(layer_info.keys()):
            info = layer_info[layer_idx]
            down_proj = info["weight"]  # [hidden_dim, intermediate_dim]
            hidden_dim, intermediate_dim = down_proj.shape

            logger.info(f"  Layer {layer_idx}: {intermediate_dim} neurons")

            # Concatenate activations
            if not layer_activations[layer_idx]:
                logger.warning(f"    No activations for layer {layer_idx}")
                continue

            all_acts = torch.cat(layer_activations[layer_idx], dim=0)  # [N, intermediate_dim]

            # 3a. Base importance: activation L2 norm (continuous, no outlier threshold)
            base_importance = (all_acts**2).mean(dim=0)  # [intermediate_dim]

            # 3b. Compute per-neuron neighborhood redundancy
            # For each neuron i, its neighborhood = neurons with highest |weight| correlation
            # Weight correlation: how similar are their output patterns (columns of down_proj)

            # Normalize weight columns
            weight_norms = down_proj.norm(dim=0, keepdim=True)
            weight_norms = torch.where(weight_norms > 1e-8, weight_norms, torch.ones_like(weight_norms))
            weight_normalized = down_proj / weight_norms  # [hidden, intermediate]

            # Weight similarity matrix (cosine similarity of output patterns)
            weight_sim = weight_normalized.T @ weight_normalized  # [intermediate, intermediate]
            weight_sim.fill_diagonal_(0)  # Exclude self

            # For each neuron, find top neighborhood_fraction most similar neurons
            num_neighbors = max(1, int(neighborhood_fraction * intermediate_dim))

            # Activation correlation matrix
            act_centered = all_acts - all_acts.mean(dim=0, keepdim=True)
            act_std = act_centered.std(dim=0, keepdim=True)
            act_std = torch.where(act_std > 1e-8, act_std, torch.ones_like(act_std))
            act_normalized = act_centered / act_std
            act_corr = (act_normalized.T @ act_normalized) / (all_acts.shape[0] - 1)
            act_corr = torch.clamp(act_corr, -1, 1)
            act_corr.fill_diagonal_(0)

            # Per-neuron neighborhood redundancy
            # Neighborhood = neurons with highest weight similarity
            # Redundancy = mean |activation correlation| with neighbors
            neighborhood_redundancy = torch.zeros(intermediate_dim)

            for i in range(intermediate_dim):
                # Get top neighbors by weight similarity
                _, neighbor_indices = torch.topk(weight_sim[i].abs(), num_neighbors)
                # Compute mean |correlation| with neighbors
                neighbor_corr = act_corr[i, neighbor_indices].abs()
                neighborhood_redundancy[i] = neighbor_corr.mean()

            # Normalize redundancy to [0, 1]
            if neighborhood_redundancy.max() > neighborhood_redundancy.min():
                norm_redundancy = (neighborhood_redundancy - neighborhood_redundancy.min()) / (
                    neighborhood_redundancy.max() - neighborhood_redundancy.min()
                )
            else:
                norm_redundancy = torch.zeros_like(neighborhood_redundancy)

            layer_scores[layer_idx] = {
                "base_importance": base_importance,
                "neighborhood_redundancy": neighborhood_redundancy,
                "norm_redundancy": norm_redundancy,
                "weight": down_proj,
            }

            logger.info(f"    Base importance: mean={base_importance.mean():.4f}, max={base_importance.max():.4f}")
            logger.info(f"    Neighborhood redundancy: mean={neighborhood_redundancy.mean():.4f}")

        # Step 4: Backward propagation of importance
        # Each layer's importance is influenced by how important its outputs are to the next layer
        logger.info("\n  Propagating importance backward...")

        sorted_layers = sorted(layer_scores.keys(), reverse=True)  # Start from last layer

        downstream_influence = {}
        for i, layer_idx in enumerate(sorted_layers):
            if i == 0:
                # Last layer: no downstream, influence = 1
                downstream_influence[layer_idx] = torch.ones(layer_scores[layer_idx]["base_importance"].shape)
            else:
                # Use weight connections to next layer to propagate importance
                prev_layer_idx = sorted_layers[i - 1]
                if prev_layer_idx in layer_scores:
                    # Simplified: downstream influence ∝ sum of |weights| to important neurons in next layer
                    layer_scores[prev_layer_idx]["base_importance"]
                    layer_scores[prev_layer_idx]["weight"]  # [hidden, intermediate]

                    # Current layer's outputs go to next layer's inputs (hidden dim)
                    # This is approximate - in reality there's attention and residual in between
                    # Use weight magnitude as proxy for connection strength
                    curr_weight = layer_scores[layer_idx]["weight"]  # [hidden, intermediate]

                    # Influence = how much this neuron's output affects important neurons downstream
                    # Proxy: sum of output weights (row of down_proj) weighted by downstream importance
                    influence = curr_weight.abs().sum(dim=0)  # [intermediate]

                    # Normalize
                    if influence.max() > influence.min():
                        influence = (influence - influence.min()) / (influence.max() - influence.min())

                    downstream_influence[layer_idx] = influence
                else:
                    downstream_influence[layer_idx] = torch.ones(layer_scores[layer_idx]["base_importance"].shape)

        # Step 5: Compute final generalized importance score
        results = {}

        for layer_idx, scores in layer_scores.items():
            info = layer_info[layer_idx]
            layer_name = info["name"]

            base = scores["base_importance"]
            redundancy = scores["norm_redundancy"]
            downstream = downstream_influence.get(layer_idx, torch.ones_like(base))

            # Normalize base importance
            if base.max() > base.min():
                base_norm = (base - base.min()) / (base.max() - base.min())
            else:
                base_norm = torch.zeros_like(base)

            # Final score: base * downstream * (1 - redundancy_penalty * redundancy)
            # High base importance + high downstream influence + low redundancy = high score (protect)
            generalized_importance = base_norm * (propagation_weight + (1 - propagation_weight) * downstream) * (1 - redundancy_penalty * redundancy)

            # Store in importance_scores
            layer_scores_dict = self.importance_scores.get(layer_name, {})
            layer_scores_dict["generalized_importance"] = generalized_importance
            layer_scores_dict["neighborhood_redundancy"] = scores["neighborhood_redundancy"]
            layer_scores_dict["downstream_influence"] = downstream
            self.importance_scores[layer_name] = layer_scores_dict

            results[layer_name] = {
                "generalized_importance": generalized_importance,
                "base_importance": base,
                "neighborhood_redundancy": scores["neighborhood_redundancy"],
                "downstream_influence": downstream,
            }

            logger.info(f"  {layer_name}: importance range [{generalized_importance.min():.4f}, {generalized_importance.max():.4f}]")

        logger.info("\nGeneralized importance computation complete!")
        logger.info("New metric 'generalized_importance' available for pruning")

        return results

    def apply_unstructured_baseline_pruning(
        self,
        *,
        sparsity: float,
        metric: str,
        mode: str = "low",
    ) -> Dict[str, torch.Tensor]:
        """
        Apply *unstructured* baseline pruning for faithful reproductions of common baselines.

        Supported metrics:
        - 'wanda_unstructured': Wanda score-based unstructured pruning.
        - 'sparsegpt_unstructured': SparseGPT-style unstructured pruning with reconstruction.

        By default this prunes FFN/MLP Linear projections (gate/up/down) only, since this
        routine focuses on FFN pruning. (We can generalize scope later if needed.)
        """
        if metric not in {"wanda_unstructured", "sparsegpt_unstructured"}:
            raise ValueError(f"Unknown unstructured baseline metric: {metric}")

        # Ensure baseline calibrations exist. Some runs disable SCAR metrics entirely,
        # which can leave scar_num_samples at 0 even though a general calibration
        # budget is configured elsewhere.
        num_calib = (
            getattr(self.config, "scar_num_samples", None)
            or getattr(self.config, "alignment_data_num_samples", None)
            or getattr(self.config, "n_calibration", None)
            or 128
        )
        if metric == "wanda_unstructured":
            wanda = getattr(self, "_wanda_baseline", None)
            if wanda is None:
                self.compute_baseline_pruning_scores(strategies=["wanda"], num_calibration_samples=num_calib)
                wanda = getattr(self, "_wanda_baseline", None)
            if wanda is None:
                raise RuntimeError("wanda_unstructured requested but Wanda baseline is not calibrated")
        else:
            sparsegpt = getattr(self, "_sparsegpt_baseline", None)
            if sparsegpt is None:
                self.compute_baseline_pruning_scores(strategies=["sparsegpt"], num_calibration_samples=num_calib)
                sparsegpt = getattr(self, "_sparsegpt_baseline", None)
            if sparsegpt is None:
                raise RuntimeError("sparsegpt_unstructured requested but SparseGPT baseline is not calibrated")

        import re

        layer_indices = set()
        for k in self.importance_scores.keys():
            m = re.search(r"layers\.(\d+)\.mlp", k)
            if m:
                layer_indices.add(int(m.group(1)))
        if not layer_indices:
            logger.warning("Unstructured baseline pruning: no MLP layers found in importance_scores; skipping")
            return {}

        underlying_model = self._get_underlying_model()
        module_dict = dict(underlying_model.named_modules())

        def _resolve_mlp_path(layer_idx: int) -> Optional[str]:
            candidates = [
                f"model.model.layers.{layer_idx}.mlp",
                f"model.layers.{layer_idx}.mlp",
                f"layers.{layer_idx}.mlp",
            ]
            for p in candidates:
                if p in module_dict:
                    return p
            return None

        masks: Dict[str, torch.Tensor] = {}

        for layer_idx in sorted(layer_indices):
            mlp_path = _resolve_mlp_path(layer_idx)
            if mlp_path is None:
                logger.warning(f"Unstructured baseline pruning: could not resolve MLP path for layer {layer_idx}")
                continue

            gate_name = f"{mlp_path}.gate_proj"
            up_name = f"{mlp_path}.up_proj"
            down_name = f"{mlp_path}.down_proj"

            if gate_name not in module_dict or up_name not in module_dict or down_name not in module_dict:
                logger.warning(f"Unstructured baseline pruning: missing projections for {mlp_path}")
                continue

            gate = module_dict[gate_name]
            up = module_dict[up_name]
            down = module_dict[down_name]

            if metric == "wanda_unstructured":
                for proj_name, proj in ((gate_name, gate), (up_name, up), (down_name, down)):
                    try:
                        mask = wanda.prune_unstructured_inplace(
                            proj,
                            sparsity,
                            layer_name=proj_name,
                            mode=mode,
                            per_row=True,
                        )
                        masks[proj_name] = mask
                    except Exception as e:
                        logger.warning(f"Wanda unstructured pruning failed for {proj_name}: {e}")
            else:
                for proj_name, proj in ((gate_name, gate), (up_name, up), (down_name, down)):
                    try:
                        keep_mask, W_new = sparsegpt.prune_and_reconstruct(
                            proj,
                            sparsity,
                            layer_name=proj_name,
                        )
                        with torch.no_grad():
                            proj.weight.data.copy_(W_new)
                        masks[proj_name] = keep_mask
                    except Exception as e:
                        logger.warning(f"SparseGPT unstructured pruning failed for {proj_name}: {e}")

        return masks

    def apply_pruning(self, sparsity: float = 0.2, metric: str = "activation_l2_norm", mode: str = "low") -> Dict[str, torch.Tensor]:
        """
        Apply pruning to MLP layers.
        - For WANDA and SparseGPT: applies unstructured (weight-level) pruning to match canonical baseline behavior
        - For other metrics: applies structured (channel-level) pruning

        Args:
            sparsity: Fraction of neurons/weights to prune
            metric: Which importance metric to use ('wanda', 'sparsegpt', 'activation_l2_norm', etc.)
            mode: 'low' to prune low-importance, 'high' for high-importance

        Returns:
            Dictionary of pruning masks
        """
        logger.info(f"Applying pruning: sparsity={sparsity}, metric={metric}, mode={mode}")

        if not self.importance_scores:
            raise ValueError("Must compute importance scores before pruning")

        # Per-call diagnostics that downstream artifact collection can use to explain
        # catastrophic baseline failures (e.g., pruning supernodes).
        #
        # Stored as a side effect to avoid changing the public return type.
        self._last_pruning_diagnostics = {}

        # Paper-faithful *unstructured* reproductions for Wanda/SparseGPT are kept separate
        # from the channel-adapted structured baselines (metric names: "wanda", "sparsegpt").
        if metric in {"wanda_unstructured", "sparsegpt_unstructured"}:
            return self.apply_unstructured_baseline_pruning(sparsity=sparsity, metric=metric, mode=mode)

        # Get pruning pipeline options from config
        pruning_distribution = getattr(self.config, "pruning_distribution", "uniform")
        pruning_min = getattr(self.config, "pruning_min_per_layer", 0.0)
        pruning_max = getattr(self.config, "pruning_max_per_layer", 0.95)

        # Store options for reference (can be used by downstream methods)
        self._pruning_options = PruningPipelineOptions(
            distribution=pruning_distribution,
            dependency_aware=getattr(self.config, "dependency_aware_pruning", False),
            min_amount=pruning_min,
            max_amount=pruning_max,
        )

        # Log pruning configuration
        if pruning_distribution != "uniform":
            logger.info(f"Using {pruning_distribution} distribution (min={pruning_min}, max={pruning_max})")

        config = PruningConfig(amount=sparsity, structured=True, pruning_mode=mode)

        # For SCAR metrics, baseline methods (wanda, sparsegpt), and other pre-computed scores,
        # use PrecomputedScorePruning since they're not in the metric registry
        precomputed_metrics = [
            # SCAR metrics (computed by SCAR analysis)
            "scar_loss_proxy",
            "scar_activation_power",
            "scar_taylor",
            "scar_curvature",
            # Learned combination (computed by compute_scar_optimal)
            "scar_optimal",
            # Supernode/connectivity metrics
            "directed_redundancy",
            "supernode_protection_score",
            "supernode_connectivity_score",
            # Random baseline (scores are generated and stored in importance_scores)
            "random",
            # Weight-only structured baseline (channel-group weight magnitude)
            "weight_magnitude",
            # Generalized importance (no outlier assumption)
            "generalized_importance",
            "neighborhood_redundancy",
            # LLM baseline methods (computed by compute_baseline_pruning_scores)
            "wanda",
            "sparsegpt",
            "owl",
            "llm_pruner",
            "flap",
            "ria",
            "slimllm",
        ]
        from alignment.pruning.base import PrecomputedScorePruning

        if metric in precomputed_metrics:
            pruner = PrecomputedScorePruning(config=config)
        else:
            # If a metric is not registered but scores are present in `importance_scores`,
            # fall back to the precomputed-score pruner. This makes it safe to add new
            # baseline strategies/ablations without touching a hard-coded allowlist.
            try:
                pruner = AlignmentPruning(metric=metric, config=config)
            except KeyError:
                logger.info(f"Metric '{metric}' not found in registry; using PrecomputedScorePruning")
                pruner = PrecomputedScorePruning(config=config)

        masks = {}
        processed_mlps = set()  # Track which MLPs we've already processed

        # Supernode "hit-rate" diagnostic: fraction of supernodes pruned by this method.
        super_total = 0
        super_pruned = 0
        layers_with_super = 0
        layers_with_super_pruned = 0
        nodes_total = 0
        nodes_pruned = 0
        both_total = 0
        both_pruned = 0

        supernode_cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}

        def _make_supernode_mask(metric_scores: torch.Tensor) -> torch.Tensor:
            num_neurons = int(metric_scores.numel())
            if num_neurons <= 0:
                return torch.zeros_like(metric_scores, dtype=torch.bool)

            top_k_cfg = supernode_cfg.get("top_k")
            core_fraction = float(supernode_cfg.get("core_fraction", 0.1))
            min_core = max(1, int(supernode_cfg.get("min_core_neurons", 1)))

            if top_k_cfg is not None:
                num_core = min(num_neurons, int(top_k_cfg))
            else:
                num_core = max(1, int(round(core_fraction * num_neurons)))

            num_core = max(num_core, min_core)
            num_core = min(num_core, num_neurons)

            _, top_indices = torch.topk(metric_scores, k=num_core, largest=True)
            out = torch.zeros_like(metric_scores, dtype=torch.bool)
            out[top_indices] = True
            return out

        def _get_layer_metric_scores(base_layer_name: str, layer_idx: str, metric_name: str) -> Optional[torch.Tensor]:
            key_candidates = [
                base_layer_name,
                base_layer_name.replace("model.model.", "model."),
                base_layer_name.replace("model.", "model.model.", 1),
                f"model.layers.{layer_idx}.mlp.down_proj",
                f"model.model.layers.{layer_idx}.mlp.down_proj",
                f"model.layers.{layer_idx}.mlp.gate_proj",
                f"model.model.layers.{layer_idx}.mlp.gate_proj",
                f"model.layers.{layer_idx}.mlp.up_proj",
                f"model.model.layers.{layer_idx}.mlp.up_proj",
            ]
            for cand in key_candidates:
                metric_vals = (self.importance_scores.get(cand) or {}).get(metric_name)
                if torch.is_tensor(metric_vals):
                    return metric_vals
            return None

        for layer_name in self.importance_scores.keys():
            if metric not in self.importance_scores[layer_name]:
                continue

            # Extract layer index (e.g., "model.layers.0.mlp.gate_proj" -> 0)
            import re

            match = re.search(r"layers\.(\d+)\.mlp", layer_name)
            if not match:
                continue
            layer_idx = match.group(1)

            # Skip if we already processed this MLP
            if layer_idx in processed_mlps:
                continue
            processed_mlps.add(layer_idx)

            # Get importance scores
            scores = self.importance_scores[layer_name][metric].clone()

            # Supernode masks may be stored under different module-name prefixes
            # (e.g., `model.layers.*` vs `model.model.layers.*`) depending on whether they were
            # produced via SCAR hooks (HF model) or via tracked-layer activation capture (wrapper).
            #
            # Try to get the mask from the same layer as the scores (matching input/output activations)
            core_mask = None
            try:
                key_candidates = [
                    layer_name,
                    layer_name.replace("model.model.", "model."),
                    layer_name.replace("model.", "model.model.", 1),
                    f"model.layers.{layer_idx}.mlp.down_proj",
                    f"model.model.layers.{layer_idx}.mlp.down_proj",
                ]
                for kcand in key_candidates:
                    core_mask = (self.importance_scores.get(kcand) or {}).get("supernode_mask")
                    if core_mask is not None:
                        break
            except Exception:
                core_mask = (self.importance_scores.get(layer_name) or {}).get("supernode_mask")

            if core_mask is not None and self._should_protect_supernodes_for_metric(metric):
                # Ensure shapes are compatible before applying mask
                if not torch.is_tensor(core_mask):
                    core_mask = torch.as_tensor(core_mask)

                # Only apply protection if mask shape matches scores shape
                # if core_mask.numel() == scores.numel():
                core_mask = core_mask.to(device=scores.device, dtype=torch.bool)
                if core_mask.shape != scores.shape:
                    core_mask = core_mask.reshape(scores.shape)

                margin = torch.abs(scores).max().detach().item() + 1.0
                if mode == "low":
                    scores[core_mask] = scores.max() + margin
                elif mode == "high":
                    scores[core_mask] = scores.min() - margin
                # else:
                #     core_mask = None

            # Create mask based on importance scores
            mask = pruner.create_pruning_mask(scores)

            try:
                nodes_total += int(mask.numel())
                nodes_pruned += int((mask == 0).sum().item())
            except Exception:
                pass

            # Diagnostic: how many supernodes did we prune in this layer?
            if core_mask is not None:
                try:
                    cm = core_mask
                    if not torch.is_tensor(cm):
                        cm = torch.as_tensor(cm)
                    cm = cm.to(device=mask.device, dtype=torch.bool)

                    if cm.numel() == mask.numel():
                        layers_with_super += 1
                        super_total += int(cm.sum().item())
                        pruned = mask == 0
                        pruned_super = int((pruned & cm).sum().item())
                        super_pruned += pruned_super
                        if pruned_super > 0:
                            layers_with_super_pruned += 1
                except Exception:
                    # Never fail pruning due to diagnostics.
                    pass

            try:
                scar_lp_scores = _get_layer_metric_scores(layer_name, layer_idx, "scar_loss_proxy")
                act_l2_scores = _get_layer_metric_scores(layer_name, layer_idx, "activation_l2_norm")
                if torch.is_tensor(scar_lp_scores) and torch.is_tensor(act_l2_scores):
                    if scar_lp_scores.numel() == act_l2_scores.numel() == mask.numel():
                        scar_mask = _make_supernode_mask(scar_lp_scores.to(device=mask.device, dtype=torch.float32))
                        act_mask = _make_supernode_mask(act_l2_scores.to(device=mask.device, dtype=torch.float32))
                        both_mask = scar_mask & act_mask
                        pruned = mask == 0

                        both_total += int(both_mask.sum().item())
                        both_pruned += int((pruned & both_mask).sum().item())
            except Exception:
                pass

            # Get the MLP module - use underlying model to handle HFCausalLM wrapper
            underlying_model = self._get_underlying_model()
            module_dict = dict(underlying_model.named_modules())

            # Try different module path patterns for compatibility
            # Order matters: try most specific first
            possible_paths = [
                f"model.model.layers.{layer_idx}.mlp",  # HFCausalLM wrapper with nested model
                f"model.layers.{layer_idx}.mlp",  # Direct HF model (LlamaForCausalLM)
                f"layers.{layer_idx}.mlp",  # Inner LlamaModel
            ]

            mlp_path = None
            for path in possible_paths:
                if path in module_dict:
                    mlp_path = path
                    break

            if mlp_path is None:
                logger.warning(f"Could not find MLP module for layer {layer_idx} (tried {possible_paths})")
                continue

            mlp_module = module_dict[mlp_path]

            try:
                # Verify we have the right modules
                if not all(hasattr(mlp_module, attr) for attr in ["gate_proj", "up_proj", "down_proj"]):
                    logger.warning(f"Layer {layer_idx} MLP missing expected projections")
                    continue

                # Verify mask shape matches intermediate dimension
                expected_dim = mlp_module.gate_proj.out_features

                if len(mask) != expected_dim:
                    logger.error(f"Mask size {len(mask)} doesn't match intermediate dim {expected_dim}")
                    continue

                # For LLMs, use make_permanent=True to avoid OOM from storing _original_weight buffers
                # We restore weights from CPU before each pruning iteration anyway

                # Prune gate_proj output dimension (rows of weight matrix)
                pruner.apply_pruning(mlp_module.gate_proj, mask, dim="output", make_permanent=True)
                masks[f"model.layers.{layer_idx}.mlp.gate_proj"] = mask

                # Prune up_proj output dimension (rows of weight matrix) - same mask
                pruner.apply_pruning(mlp_module.up_proj, mask, dim="output", make_permanent=True)
                masks[f"model.layers.{layer_idx}.mlp.up_proj"] = mask

                # Prune down_proj input dimension (columns of weight matrix)
                pruner.apply_pruning(mlp_module.down_proj, mask, dim="input", make_permanent=True)
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
        # Store diagnostics for the caller (run()) to attach into results JSON.
        self._last_pruning_diagnostics = {
            "supernode_pruning": {
                "supernodes_total": int(super_total),
                "supernodes_pruned": int(super_pruned),
                "supernodes_pruned_frac": (float(super_pruned) / float(super_total)) if super_total > 0 else None,
                "nodes_total": int(nodes_total),
                "nodes_pruned": int(nodes_pruned),
                "nodes_pruned_frac": (float(nodes_pruned) / float(nodes_total)) if nodes_total > 0 else None,
                "supernodes_both_scar_lp_activation_l2_total": int(both_total),
                "supernodes_both_scar_lp_activation_l2_pruned": int(both_pruned),
                "supernodes_both_scar_lp_activation_l2_pruned_frac": (float(both_pruned) / float(both_total)) if both_total > 0 else None,
                "layers_with_supernodes": int(layers_with_super),
                "layers_with_supernodes_pruned": int(layers_with_super_pruned),
            }
        }
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
                metric=metric,
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
                    logger.info(f"  Layer {layer_idx} attention: pruned {pruned_fraction:.2%} of Q/K/V outputs")

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
        metric: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[int]]:
        """
        Convert per-neuron attention scores into a shared mask aligned with heads.
        Returns (mask, heads_kept, total_heads).
        """
        scores = scores.flatten()
        device = scores.device

        core_mask = self.importance_scores.get(layer_key, {}).get("supernode_mask")
        do_protect = core_mask is not None and self._should_protect_supernodes_for_metric(metric)
        if do_protect:
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
                f"Attention score length {scores.numel()} is incompatible with num_heads={num_heads}; " f"falling back to per-neuron mask."
            )
            raw_mask = MaskOperations.create_structured_mask(scores, amount=sparsity, mode=mode)
            return raw_mask.float().to(device), None, None

        head_scores = scores.view(num_heads, head_dim).mean(dim=1)
        head_keep = MaskOperations.create_structured_mask(head_scores, amount=sparsity, mode=mode)

        # Ensure that any head containing a protected core neuron is always kept.
        if do_protect and core_mask is not None and core_mask.numel() == scores.numel():
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
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            logger.error("PEFT library not installed. Cannot run minimal repair.")
            return

        logger.info(f"Applying Minimal Repair (LoRA) for {epochs} epochs...")

        # Configure LoRA
        # We target the projection layers in MLPs.
        target_modules = ["gate_proj", "up_proj", "down_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules
        )

        # Wrap model
        # Note: We are wrapping the HUGGINGFACE model, not our wrapper
        # Our wrapper wrapper_model.model or similar needs to be accessed
        hf_model = self.model  # This is the AutoModelForCausalLM

        # Enable gradients for LoRA
        hf_model.enable_input_require_grads()

        model = get_peft_model(hf_model, peft_config)
        model.print_trainable_parameters()

        # Create trainer
        # Need a dataset loader
        from torch.utils.data import DataLoader

        from alignment.dataops.datasets.text_datasets import load_text_dataset

        # Minimal dataset for repair (calibration set)
        dataset = load_text_dataset(dataset_name, self.config.model_config.get("model_id"), split="train", max_samples=1000)

        # Create a simple collator if needed, or use default
        def collate_fn(batch):
            input_ids = [b["input_ids"] for b in batch]
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
        self.model = model  # Update self.model to the repaired one

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

        class _SkipScarVisualizations(Exception):
            """Internal sentinel to skip SCAR plotting when generate_plots=False."""

        results: Dict[str, Any] = {"config": self.config.to_dict(), "importance_scores": {}, "pruning_results": {}, "evaluation": {}}

        scores = self.compute_importance_scores(num_samples=self.config.alignment_data_num_samples)

        # Optional: SCAR-style supernode metrics (T_i, R_i, loss proxy) for FFN layers
        scar_scores: Dict[str, Any] = {}
        if getattr(self.config, "do_scar_metrics", False):
            try:
                scar_scores = self.compute_scar_supernode_metrics()
            except Exception as e:
                logger.error(f"Error while computing SCAR supernode metrics: {e}")
            else:
                if scar_scores:
                    # Many downstream SCAR analyses (robustness, connectivity, etc.) use `plots_dir`
                    # even when `generate_plots=False`. Define it unconditionally here.
                    plots_dir = Path(getattr(self.config, "plots_dir", Path(self.config.log_dir) / "plots"))
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        if not getattr(self.config, "generate_plots", True):
                            raise _SkipScarVisualizations()

                        import matplotlib.pyplot as plt

                        # Create organized subfolders
                        scar_plots_dir = plots_dir / "scar"
                        scar_plots_dir.mkdir(parents=True, exist_ok=True)

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
                            save_path=scar_plots_dir / "scar_loss_proxy_layers.png",
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
                            save_path=scar_plots_dir / "scar_metrics_heatmap.png",
                        )
                        plt.close(fig)

                        # Generate importance score histograms for each metric
                        logger.info("Generating importance score histograms...")
                        histogram_dir = plots_dir / "histograms"
                        histogram_dir.mkdir(parents=True, exist_ok=True)

                        for metric_name in scar_metric_list + [
                            "activation_l2_norm",
                            "rayleigh_quotient",
                            "gaussian_mi_analytic",
                            "average_redundancy",
                        ]:
                            try:
                                # Collect all scores for this metric across layers
                                all_scores = []
                                for layer_name, layer_metrics in self.importance_scores.items():
                                    if metric_name in layer_metrics:
                                        scores_tensor = layer_metrics[metric_name]
                                        if torch.is_tensor(scores_tensor):
                                            all_scores.append(scores_tensor.detach().float().cpu().numpy().flatten())

                                if all_scores:
                                    combined_scores = np.concatenate(all_scores)
                                    fig = viz.plot_1d_histogram(
                                        values=combined_scores,
                                        xlabel=metric_name.replace("_", " ").title(),
                                        ylabel="Count",
                                        title=f"{metric_name.replace('_', ' ').title()} Distribution (All Layers)",
                                        save_path=histogram_dir / f"histogram_{metric_name}.png",
                                    )
                                    plt.close(fig)
                            except Exception as hist_err:
                                logger.warning(f"Failed to generate histogram for {metric_name}: {hist_err}")

                        # Generate supernode comparison plots (supernode vs non-supernode metrics)
                        logger.info("Generating supernode comparison plots...")
                        supernode_dir = plots_dir / "supernode" / "comparison"
                        supernode_dir.mkdir(parents=True, exist_ok=True)

                        # Get supernode config
                        supernode_cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
                        supernode_fraction = supernode_cfg.get("core_fraction", 0.01)

                        if supernode_cfg.get("enabled", False) or scar_scores:
                            comparison_metrics = supernode_cfg.get(
                                "compute_metrics", ["activation_l2_norm", "rayleigh_quotient", "scar_activation_power", "scar_loss_proxy"]
                            )

                            # Aggregate supernodes vs non-supernodes across all layers
                            supernode_vals = {m: [] for m in comparison_metrics}
                            non_supernode_vals = {m: [] for m in comparison_metrics}

                            for layer_name, layer_scores in self.importance_scores.items():
                                # Try to get supernode_mask, or compute it from scar_scores
                                mask = layer_scores.get("supernode_mask")
                                if mask is None and scar_scores and layer_name in scar_scores:
                                    # Compute mask on-the-fly from scar_activation_power
                                    scar_layer = scar_scores[layer_name]
                                    act_power = scar_layer.get("scar_activation_power") or scar_layer.get("scar_loss_proxy")
                                    if act_power is not None:
                                        act_power = act_power.float().cpu()
                                        n = act_power.numel()
                                        num_supernodes = max(1, int(supernode_fraction * n))
                                        _, top_idx = torch.topk(act_power, num_supernodes)
                                        mask = torch.zeros(n, dtype=torch.bool)
                                        mask[top_idx] = True
                                        logger.info(f"  Created supernode mask for {layer_name}: {num_supernodes} supernodes")

                                if mask is None:
                                    continue
                                mask = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)

                                for metric_name in comparison_metrics:
                                    # Check both importance_scores and scar_scores for the metric
                                    metric_vals = None
                                    if metric_name in layer_scores:
                                        metric_vals = layer_scores[metric_name]
                                    elif scar_scores and layer_name in scar_scores and metric_name in scar_scores[layer_name]:
                                        metric_vals = scar_scores[layer_name][metric_name]

                                    if metric_vals is not None:
                                        metric_vals = metric_vals.float().cpu().numpy() if torch.is_tensor(metric_vals) else np.asarray(metric_vals)
                                        if len(metric_vals) == len(mask):
                                            supernode_vals[metric_name].extend(metric_vals[mask])
                                            non_supernode_vals[metric_name].extend(metric_vals[~mask])

                            # Plot comparison for each metric
                            for metric_name in comparison_metrics:
                                if supernode_vals[metric_name] and non_supernode_vals[metric_name]:
                                    try:
                                        sn_arr = np.array(supernode_vals[metric_name])
                                        non_sn_arr = np.array(non_supernode_vals[metric_name])

                                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                                        # Histogram comparison
                                        axes[0].hist(non_sn_arr, bins=50, alpha=0.7, label=f"Non-Supernode (n={len(non_sn_arr)})", color="#5dade2")
                                        axes[0].hist(sn_arr, bins=50, alpha=0.7, label=f"Supernode (n={len(sn_arr)})", color="#c0392b")
                                        axes[0].set_xlabel(metric_name.replace("_", " ").title())
                                        axes[0].set_ylabel("Count")
                                        axes[0].set_title(f"{metric_name.replace('_', ' ').title()}: Supernode vs Non-Supernode")
                                        axes[0].legend()
                                        axes[0].grid(True, alpha=0.3)

                                        # Box plot comparison
                                        box_data = [non_sn_arr, sn_arr]
                                        bp = axes[1].boxplot(box_data, labels=["Non-Supernode", "Supernode"], patch_artist=True)
                                        bp["boxes"][0].set_facecolor("#5dade2")
                                        bp["boxes"][1].set_facecolor("#c0392b")
                                        axes[1].set_ylabel(metric_name.replace("_", " ").title())
                                        axes[1].set_title("Distribution Comparison")
                                        axes[1].grid(True, alpha=0.3)

                                        # Add statistics annotation
                                        stats_text = f"Supernode: μ={sn_arr.mean():.4f}, σ={sn_arr.std():.4f}\nNon-SN: μ={non_sn_arr.mean():.4f}, σ={non_sn_arr.std():.4f}"
                                        axes[1].text(
                                            0.02,
                                            0.98,
                                            stats_text,
                                            transform=axes[1].transAxes,
                                            verticalalignment="top",
                                            fontsize=9,
                                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                                        )

                                        plt.tight_layout()
                                        fig.savefig(supernode_dir / f"supernode_comparison_{metric_name}.png", dpi=300, bbox_inches="tight")
                                        plt.close(fig)
                                        logger.info(f"Generated supernode comparison plot for {metric_name}")
                                    except Exception as cmp_err:
                                        logger.warning(f"Failed to generate supernode comparison for {metric_name}: {cmp_err}")

                    except _SkipScarVisualizations:
                        # Skip plot generation but keep running downstream SCAR analyses.
                        pass
                    except Exception as viz_err:
                        logger.error(f"Failed to generate SCAR visualizations: {viz_err}")

                    # Run supernode connection analysis
                    try:
                        supernode_config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
                        supernode_fraction = supernode_config.get("core_fraction", 0.01)
                        follower_fraction = supernode_config.get("follower_fraction", 0.10)
                        supernode_metric = supernode_config.get("score_metric", "scar_activation_power")
                        cross_layer_analysis = supernode_config.get("cross_layer_analysis", True)
                        compute_metrics = supernode_config.get(
                            "compute_metrics", ["activation", "rayleigh_quotient", "mutual_information", "redundancy"]
                        )
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

                    # Compute directed redundancy from supernodes to downstream neurons
                    if getattr(self.config, "do_directed_redundancy", True):
                        try:
                            supernode_config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
                            directed_redundancy_results = self.compute_directed_redundancy(
                                scar_scores=scar_scores,
                                supernode_fraction=supernode_config.get("core_fraction", 0.01),
                            )
                            results["directed_redundancy"] = directed_redundancy_results
                            logger.info("Directed redundancy computation complete")
                        except Exception as dr_err:
                            logger.error(f"Failed directed redundancy computation: {dr_err}")
                            import traceback

                            logger.error(traceback.format_exc())

                    # Optional: validate LP against true Δloss via single-channel ablations (expensive).
                    # Enable via `supernode.lp_ablation_validation.enabled=true`.
                    # NOTE: This probe depends only on `scar_scores` and does NOT require connectivity pruning.
                    try:
                        supernode_config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
                        v_cfg = supernode_config.get("lp_ablation_validation", {}) or {}
                        if isinstance(v_cfg, dict) and bool(v_cfg.get("enabled", False)):
                            v_res = self.compute_lp_ablation_validation(
                                scar_scores=scar_scores,
                                layer_stride=int(v_cfg.get("layer_stride", 8)),
                                layer_indices=v_cfg.get("layer_indices", None),
                                num_texts=int(v_cfg.get("num_texts", 8)),
                                max_length=int(v_cfg.get("max_length", 256)),
                                num_channels=int(v_cfg.get("num_channels", 128)),
                                quantile_bins=int(v_cfg.get("quantile_bins", 8)),
                                seed=int(v_cfg.get("seed", getattr(self.config, "seed", 0) or 0)),
                            )
                            results["lp_ablation_validation"] = v_res
                    except Exception as _val_err:
                        logger.error(f"Failed LP ablation validation: {_val_err}")

                    # Compute supernode-connectivity based pruning score
                    if getattr(self.config, "do_connectivity_pruning", True):
                        try:
                            supernode_config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
                            connectivity_results = self.compute_supernode_connectivity_pruning_score(
                                scar_scores=scar_scores,
                                supernode_fraction=supernode_config.get("core_fraction", 0.01),
                                high_connectivity_fraction=supernode_config.get("follower_fraction", 0.10),
                                redundancy_weight=supernode_config.get("redundancy_weight", 0.5),
                                plots_dir=plots_dir,
                            )
                            results["supernode_connectivity"] = connectivity_results
                            logger.info("Supernode-connectivity pruning score computation complete")

                            # Optional: conditional halo ablation (causal redundancy probe).
                            # Disabled by default; enable via `supernode.conditional_halo_ablation.enabled=true`.
                            try:
                                ca_cfg = (
                                    supernode_config.get("conditional_halo_ablation", {}) or supernode_config.get("conditional_ablation", {}) or {}
                                )
                                if isinstance(ca_cfg, dict) and bool(ca_cfg.get("enabled", False)):
                                    ca_res = self.compute_conditional_halo_ablation(
                                        scar_scores=scar_scores,
                                        supernode_fraction=float(supernode_config.get("core_fraction", 0.01)),
                                        halo_fraction=float(supernode_config.get("follower_fraction", 0.10)),
                                        layer_stride=int(ca_cfg.get("layer_stride", 4)),
                                        layer_indices=ca_cfg.get("layer_indices", None),
                                        num_texts=int(ca_cfg.get("num_texts", 16)),
                                        max_length=int(ca_cfg.get("max_length", 256)),
                                        match_bins=int(ca_cfg.get("match_bins", 10)),
                                        seed=int(ca_cfg.get("seed", getattr(self.config, "seed", 0) or 0)),
                                    )
                                    results["conditional_halo_ablation"] = ca_res
                            except Exception as _ca_err:
                                logger.error(f"Failed conditional halo ablation analysis: {_ca_err}")

                        except Exception as conn_err:
                            logger.error(f"Failed supernode-connectivity computation: {conn_err}")
                            import traceback

                            logger.error(traceback.format_exc())

                    # Halo vs Non-halo Redundancy Analysis
                    halo_config = getattr(self.config, "halo_analysis", {}) or {}
                    if getattr(self.config, "do_halo_analysis", False) or halo_config.get("enabled", False):
                        try:
                            logger.info("Running halo vs non-halo redundancy analysis...")
                            halo_results = self.analyze_halo_vs_nonhalo_redundancy(
                                scar_scores=scar_scores,
                                supernode_fraction=halo_config.get("supernode_fraction", 0.01),
                                halo_fraction=halo_config.get("halo_fraction", 0.10),
                                num_samples=halo_config.get("num_samples", 8),
                                max_length=halo_config.get("max_length", 256),
                                sample_pairs=halo_config.get("sample_pairs", 2000),
                            )
                            results["halo_analysis"] = halo_results
                            logger.info("Halo analysis complete")
                        except Exception as halo_err:
                            logger.error(f"Failed halo analysis: {halo_err}")
                            import traceback

                            logger.error(traceback.format_exc())

                    # Generalized Importance (no outlier assumption)
                    gen_config = getattr(self.config, "generalized_importance", {}) or {}
                    if getattr(self.config, "do_generalized_importance", False) or gen_config.get("enabled", False):
                        try:
                            logger.info("Computing generalized importance (no outlier assumption)...")
                            gen_results = self.compute_generalized_importance(
                                num_samples=gen_config.get("num_samples", 8),
                                max_length=gen_config.get("max_length", 256),
                                neighborhood_fraction=gen_config.get("neighborhood_fraction", 0.10),
                                propagation_weight=gen_config.get("propagation_weight", 0.3),
                                redundancy_penalty=gen_config.get("redundancy_penalty", 0.5),
                            )
                            results["generalized_importance"] = gen_results
                            logger.info("Generalized importance computation complete")
                        except Exception as gen_err:
                            logger.error(f"Failed generalized importance computation: {gen_err}")
                            import traceback

                            logger.error(traceback.format_exc())

                    # Supernode Robustness Analysis
                    # Analyzes consistency of supernode identification across metrics and batches
                    robustness_config = getattr(self.config, "supernode_robustness", None)
                    if robustness_config is None:
                        robustness_config = {}
                    elif hasattr(robustness_config, "__dict__"):
                        robustness_config = vars(robustness_config)
                    # Enable by default for LLM experiment runs (can be disabled via config).
                    logger.info(f"Supernode robustness config: enabled={robustness_config.get('enabled', True)}")
                    if robustness_config.get("enabled", True):
                        try:
                            logger.info("Running supernode robustness analysis...")
                            robustness_results = self.analyze_supernode_robustness(
                                supernode_fraction=robustness_config.get("supernode_fraction", 0.01),
                                num_bootstrap_samples=robustness_config.get("num_bootstrap_samples", 10),
                                batch_size=robustness_config.get("batch_size", 32),
                                max_samples=robustness_config.get("max_samples", 256),
                                metrics=robustness_config.get("metrics", None),
                                target_layers=robustness_config.get("target_layers", None),
                                plots_dir=plots_dir,
                            )
                            results["supernode_robustness"] = robustness_results
                            logger.info("Supernode robustness analysis complete")
                        except Exception as rob_err:
                            logger.error(f"Failed supernode robustness analysis: {rob_err}")
                            import traceback

                            logger.error(traceback.format_exc())

                    # Halo vs Non-Halo Metrics Visualization by Layer
                    # Visualize mean metrics for supernode/halo/non-halo groups across layers
                    summary_config = getattr(self.config, "supernode_summary", None)
                    if summary_config is None:
                        summary_config = {}
                    elif hasattr(summary_config, "__dict__"):
                        summary_config = vars(summary_config)
                    if summary_config.get("enabled", True):  # Enabled by default
                        try:
                            logger.info("Visualizing halo vs non-halo metrics by layer...")
                            supernode_config = getattr(self.config, "supernode", {}) or {}
                            halo_nonhalo_results = self.visualize_halo_nonhalo_metrics_by_layer(
                                scar_scores=scar_scores,
                                supernode_fraction=supernode_config.get("core_fraction", 0.01),
                                halo_fraction=supernode_config.get("follower_fraction", 0.10),
                                plots_dir=plots_dir,
                            )
                            results["halo_nonhalo_by_layer"] = halo_nonhalo_results
                            logger.info("Halo vs non-halo metrics visualization complete")
                        except Exception as hnh_err:
                            logger.error(f"Failed halo vs non-halo metrics visualization: {hnh_err}")
                            import traceback

                            logger.error(traceback.format_exc())

                    # Supernode Outlier Z-Score Analysis
                    # Compute z-scores showing how much of an outlier supernodes are per layer
                    if summary_config.get("outlier_analysis", True):  # Enabled by default
                        try:
                            logger.info("Computing supernode outlier z-scores by layer...")
                            supernode_config = getattr(self.config, "supernode", {}) or {}
                            outlier_results = self.compute_supernode_outlier_scores(
                                scar_scores=scar_scores,
                                supernode_fraction=supernode_config.get("core_fraction", 0.01),
                                plots_dir=plots_dir,
                            )
                            results["supernode_outlier_scores"] = outlier_results
                            logger.info("Supernode outlier z-score analysis complete")
                        except Exception as outlier_err:
                            logger.error(f"Failed supernode outlier z-score analysis: {outlier_err}")
                            import traceback

                            logger.error(traceback.format_exc())

        # Optional: Attention SCAR metrics (per-head loss proxy analysis)
        attn_scar_scores: Dict[str, Any] = {}
        if getattr(self.config, "do_attention_scar_metrics", False):
            try:
                attn_scar_scores = self.compute_attention_scar_metrics()
                results["attention_scar_scores"] = attn_scar_scores
            except Exception as attn_err:
                logger.error(f"Error while computing attention SCAR metrics: {attn_err}")
                import traceback

                logger.error(traceback.format_exc())
            else:
                if attn_scar_scores and getattr(self.config, "generate_plots", True):
                    try:
                        import matplotlib.pyplot as plt

                        plots_dir = Path(getattr(self.config, "plots_dir", Path(self.config.log_dir) / "plots"))
                        attn_plots_dir = plots_dir / "attention_scar"
                        attn_plots_dir.mkdir(parents=True, exist_ok=True)

                        # Convert to float32 for matplotlib
                        attn_scores_f32 = {}
                        for layer_name, layer_metrics in attn_scar_scores.items():
                            attn_scores_f32[layer_name] = {}
                            for metric_name, values in layer_metrics.items():
                                if torch.is_tensor(values):
                                    attn_scores_f32[layer_name][metric_name] = values.float().cpu()
                                else:
                                    attn_scores_f32[layer_name][metric_name] = values

                        # Plot 1: Attention loss proxy distribution across layers
                        fig, ax = plt.subplots(figsize=(14, 6))
                        layer_names = []
                        all_lp_per_layer = []
                        for ln in sorted(attn_scores_f32.keys(), key=lambda x: int(attn_scores_f32[x].get("layer_idx", "0"))):
                            if "attn_loss_proxy" in attn_scores_f32[ln]:
                                layer_names.append(attn_scores_f32[ln].get("layer_idx", ln))
                                all_lp_per_layer.append(attn_scores_f32[ln]["attn_loss_proxy"].numpy())
                        if all_lp_per_layer:
                            bp = ax.boxplot(all_lp_per_layer, labels=layer_names, patch_artist=True)
                            for box in bp["boxes"]:
                                box.set_facecolor("#85C1E9")
                            ax.set_xlabel("Layer Index")
                            ax.set_ylabel("Attention Loss Proxy")
                            ax.set_title("Per-Head Attention Loss Proxy Distribution Across Layers")
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            fig.savefig(attn_plots_dir / "attn_loss_proxy_by_layer.png", dpi=150)
                        plt.close(fig)

                        # Plot 2: Attention vs FFN concentration comparison
                        if scar_scores:
                            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                            # FFN concentration: top-1% captures X% of loss proxy mass
                            all_ffn_lp = []
                            for ln, lm in scar_scores.items():
                                if "scar_loss_proxy" in lm:
                                    lp = lm["scar_loss_proxy"]
                                    if torch.is_tensor(lp):
                                        all_ffn_lp.append(lp.float().cpu())
                            if all_ffn_lp:
                                ffn_lp_cat = torch.cat(all_ffn_lp)
                                ffn_sorted, _ = torch.sort(ffn_lp_cat, descending=True)
                                ffn_cumsum = torch.cumsum(ffn_sorted, dim=0) / (ffn_sorted.sum() + 1e-8)
                                axes[0].plot(
                                    torch.linspace(0, 100, len(ffn_cumsum)).numpy(),
                                    ffn_cumsum.numpy() * 100,
                                    color="#E74C3C",
                                    linewidth=2,
                                    label="FFN Channels",
                                )
                                axes[0].axvline(x=1.0, color="gray", linestyle="--", alpha=0.7)
                                axes[0].axhline(y=50, color="gray", linestyle=":", alpha=0.7)
                                axes[0].set_xlabel("Percentile of Channels")
                                axes[0].set_ylabel("Cumulative % of Loss Proxy Mass")
                                axes[0].set_title("FFN: Loss Proxy Concentration")
                                axes[0].legend()
                                axes[0].grid(True, alpha=0.3)

                            # Attention concentration: top-10% captures X% of loss proxy mass
                            all_attn_lp = []
                            for ln, lm in attn_scores_f32.items():
                                if "attn_loss_proxy" in lm:
                                    lp = lm["attn_loss_proxy"]
                                    if torch.is_tensor(lp):
                                        all_attn_lp.append(lp)
                                    else:
                                        all_attn_lp.append(torch.tensor(lp))
                            if all_attn_lp:
                                attn_lp_cat = torch.cat(all_attn_lp)
                                attn_sorted, _ = torch.sort(attn_lp_cat, descending=True)
                                attn_cumsum = torch.cumsum(attn_sorted, dim=0) / (attn_sorted.sum() + 1e-8)
                                axes[1].plot(
                                    torch.linspace(0, 100, len(attn_cumsum)).numpy(),
                                    attn_cumsum.numpy() * 100,
                                    color="#3498DB",
                                    linewidth=2,
                                    label="Attention Heads",
                                )
                                axes[1].axvline(x=10.0, color="gray", linestyle="--", alpha=0.7)
                                axes[1].axhline(y=50, color="gray", linestyle=":", alpha=0.7)
                                axes[1].set_xlabel("Percentile of Heads")
                                axes[1].set_ylabel("Cumulative % of Loss Proxy Mass")
                                axes[1].set_title("Attention: Loss Proxy Concentration")
                                axes[1].legend()
                                axes[1].grid(True, alpha=0.3)

                            plt.tight_layout()
                            fig.savefig(attn_plots_dir / "ffn_vs_attn_concentration.png", dpi=150)
                            plt.close(fig)

                        # Plot 3: Heatmap of attention metrics across layers
                        attn_metric_names = ["attn_activation_power", "attn_gradient_power", "attn_taylor", "attn_loss_proxy"]
                        num_layers = len(attn_scores_f32)
                        if num_layers > 0:
                            list(attn_scores_f32.values())[0].get("num_heads", 32)

                            for metric_name in attn_metric_names:
                                metric_data = []
                                layer_labels = []
                                for ln in sorted(attn_scores_f32.keys(), key=lambda x: int(attn_scores_f32[x].get("layer_idx", "0"))):
                                    if metric_name in attn_scores_f32[ln]:
                                        vals = attn_scores_f32[ln][metric_name]
                                        if torch.is_tensor(vals):
                                            vals = vals.numpy()
                                        metric_data.append(vals)
                                        layer_labels.append(f"L{attn_scores_f32[ln].get('layer_idx', ln)}")

                                if metric_data:
                                    metric_arr = np.array(metric_data)  # [num_layers, num_heads]
                                    fig, ax = plt.subplots(figsize=(16, 8))
                                    im = ax.imshow(metric_arr, aspect="auto", cmap="viridis")
                                    ax.set_xlabel("Head Index")
                                    ax.set_ylabel("Layer")
                                    ax.set_yticks(range(len(layer_labels)))
                                    ax.set_yticklabels(layer_labels)
                                    ax.set_title(f"{metric_name.replace('_', ' ').title()} per Attention Head")
                                    cbar = plt.colorbar(im, ax=ax)
                                    cbar.set_label(metric_name)
                                    plt.tight_layout()
                                    fig.savefig(attn_plots_dir / f"{metric_name}_heatmap.png", dpi=150)
                                    plt.close(fig)

                        logger.info(f"Attention SCAR plots saved to {attn_plots_dir}")
                    except Exception as attn_plot_err:
                        logger.error(f"Failed to generate attention SCAR plots: {attn_plot_err}")
                        import traceback

                        logger.error(traceback.format_exc())

        # Compute baseline pruning scores (Wanda, SparseGPT, OWL, LLM-Pruner, FLAP, RIA, SlimLLM) if configured
        # This runs OUTSIDE the SCAR metrics block so it can work independently
        baseline_scores: Dict[str, Any] = {}
        pruning_strategies = getattr(self.config, "pruning_strategies", None) or []
        # Baseline calibration is needed for all calibration-based methods
        ALL_CALIBRATION_BASELINES = ["wanda", "sparsegpt", "owl", "llm_pruner", "flap", "ria", "slimllm"]
        baseline_strategies = []
        for baseline in ALL_CALIBRATION_BASELINES:
            # Also check unstructured variants for wanda/sparsegpt
            variants = [baseline, f"{baseline}_unstructured"] if baseline in ["wanda", "sparsegpt"] else [baseline]
            if any(v in pruning_strategies for v in variants):
                baseline_strategies.append(baseline)
        logger.info(f"Checking baseline strategies: pruning_strategies={pruning_strategies}, baseline_strategies={baseline_strategies}")
        if baseline_strategies:
            try:
                baseline_num_calib = (
                    getattr(self.config, "scar_num_samples", None)
                    or getattr(self.config, "alignment_data_num_samples", None)
                    or getattr(self.config, "n_calibration", None)
                    or 128
                )
                baseline_scores = self.compute_baseline_pruning_scores(
                    strategies=baseline_strategies,
                    num_calibration_samples=baseline_num_calib,
                )
                logger.info(f"Computed baseline pruning scores for {len(baseline_scores)} layers")
            except Exception as base_err:
                logger.error(f"Failed baseline pruning score computation: {base_err}")
                import traceback

                logger.error(traceback.format_exc())

        # Fast, calibration-free channel magnitude baseline ("Magnitude (channel)")
        if "weight_magnitude" in pruning_strategies:
            try:
                self.compute_weight_magnitude_channel_scores()
            except Exception as mag_err:
                logger.error(f"Failed weight_magnitude score computation: {mag_err}")
                import traceback

                logger.error(traceback.format_exc())

        # Structured random baseline ("Random (channel)")
        if "random" in pruning_strategies:
            try:
                # Deterministic by default (seeded by config.seed).
                self.compute_random_channel_scores()
            except Exception as rand_err:
                logger.error(f"Failed random baseline score computation: {rand_err}")
                import traceback

                logger.error(traceback.format_exc())

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

        # Add Attention SCAR metrics summaries (if any)
        if attn_scar_scores:
            results["attention_scar_scores"] = {}
            for layer_name, attn_layer_scores in attn_scar_scores.items():
                results["attention_scar_scores"][layer_name] = {}
                for metric_name, vals in attn_layer_scores.items():
                    if torch.is_tensor(vals):
                        try:
                            results["attention_scar_scores"][layer_name][metric_name] = {
                                "mean": float(vals.mean().item()),
                                "std": float(vals.std().item()),
                                "min": float(vals.min().item()),
                                "max": float(vals.max().item()),
                            }
                        except Exception:
                            results["attention_scar_scores"][layer_name][metric_name] = {"summary": "unavailable"}
                    else:
                        results["attention_scar_scores"][layer_name][metric_name] = vals

            # Compute concentration metrics for attention heads
            all_attn_lp = []
            for ln, lm in attn_scar_scores.items():
                if "attn_loss_proxy" in lm:
                    lp = lm["attn_loss_proxy"]
                    if torch.is_tensor(lp):
                        all_attn_lp.append(lp.float().cpu())
            if all_attn_lp:
                attn_lp_cat = torch.cat(all_attn_lp)
                total_heads = len(attn_lp_cat)
                top_10pct = max(1, int(0.1 * total_heads))
                sorted_lp, _ = torch.sort(attn_lp_cat, descending=True)
                top_10_mass = sorted_lp[:top_10pct].sum() / (attn_lp_cat.sum() + 1e-8)
                results["attention_scar_summary"] = {
                    "total_heads": total_heads,
                    "top_10pct_heads": top_10pct,
                    "top_10pct_mass_fraction": float(top_10_mass.item()),
                    "coefficient_of_variation": float((attn_lp_cat.std() / (attn_lp_cat.mean() + 1e-8)).item()),
                }

        if self.config.do_perplexity_computation:
            baseline_ppl = self.evaluate_perplexity(dataset=self.config.evaluation_dataset, num_samples=self.config.evaluation_num_samples)
            results["evaluation"]["baseline_perplexity"] = baseline_ppl

        # For summary tables/plots: evaluate the unpruned model once on the full configured benchmark suite.
        # (This avoids hard-coding "Unpruned" numbers in the manuscript.)
        try:
            llm_cfg = getattr(self.config, "llm", {}) or {}
            eval_metrics = llm_cfg.get("evaluation_metrics") or getattr(self.config, "evaluation_metrics", ["perplexity"])
            if isinstance(eval_metrics, str):
                eval_metrics = [eval_metrics]
            if eval_metrics:
                baseline_eval = self.evaluate_multiple_metrics(
                    metrics=eval_metrics,
                    num_samples=self.config.evaluation_num_samples,
                )
                results["evaluation"]["baseline_metrics"] = baseline_eval
                # Keep baseline_perplexity in sync if evaluate_multiple_metrics produced it
                if results["evaluation"].get("baseline_perplexity") is None and baseline_eval.get("perplexity") is not None:
                    results["evaluation"]["baseline_perplexity"] = baseline_eval.get("perplexity")
        except Exception as e:
            logger.warning(f"Failed baseline full-metric evaluation: {e}")

        # Some SCAR pruning scores (e.g., `supernode_connectivity_score`) were historically computed
        # inside the `generate_plots` block. For fast sweeps we often run with
        # `generate_plots=false`, but we still need these scores for pruning to run.
        if scar_scores and not getattr(self.config, "generate_plots", True):
            supernode_config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}

            if getattr(self.config, "do_directed_redundancy", True):
                try:
                    directed_redundancy_results = self.compute_directed_redundancy(
                        scar_scores=scar_scores,
                        supernode_fraction=supernode_config.get("core_fraction", 0.01),
                    )
                    results["directed_redundancy"] = directed_redundancy_results
                    logger.info("Directed redundancy computation complete")
                except Exception as dr_err:
                    logger.error(f"Failed directed redundancy computation: {dr_err}")
                    import traceback

                    logger.error(traceback.format_exc())

            if getattr(self.config, "do_connectivity_pruning", True):
                try:
                    connectivity_results = self.compute_supernode_connectivity_pruning_score(
                        scar_scores=scar_scores,
                        supernode_fraction=supernode_config.get("core_fraction", 0.01),
                        high_connectivity_fraction=supernode_config.get("follower_fraction", 0.10),
                        redundancy_weight=supernode_config.get("redundancy_weight", 0.5),
                        plots_dir=None,
                    )
                    results["supernode_connectivity"] = connectivity_results
                    logger.info("Supernode-connectivity pruning score computation complete")
                except Exception as conn_err:
                    logger.error(f"Failed supernode-connectivity computation: {conn_err}")
                    import traceback

                    logger.error(traceback.format_exc())

        # Optional ablations that should run regardless of `generate_plots`.
        if scar_scores:
            supernode_config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}

            # SCAR Optimal: learned combination of SCAR components
            if getattr(self.config, "do_scar_optimal", False):
                try:
                    logger.info("Computing SCAR-optimal (learned component weights)...")
                    scar_optimal_results = self.compute_scar_optimal(
                        scar_scores=scar_scores,
                        num_validation_samples=32,
                        sparsity=0.3,
                        search_granularity=5,
                        plots_dir=None,
                    )
                    results["scar_optimal"] = scar_optimal_results
                    logger.info(f"SCAR-optimal complete: best_weights={scar_optimal_results.get('optimal_weights', {})}")
                except Exception as opt_err:
                    logger.error(f"Failed SCAR-optimal computation: {opt_err}")
                    import traceback

                    logger.error(traceback.format_exc())

            # Random supernode ablation: test importance of LP-based supernode identification
            if getattr(self.config, "do_random_supernode_ablation", False):
                try:
                    logger.info("Running random supernode ablation...")
                    random_ablation_results = self.compute_random_supernode_ablation(
                        scar_scores=scar_scores,
                        supernode_fraction=supernode_config.get("core_fraction", 0.01),
                        num_trials=5,
                        sparsity=0.5,
                    )
                    results["random_supernode_ablation"] = random_ablation_results
                    logger.info("Random supernode ablation complete")
                except Exception as abl_err:
                    logger.error(f"Failed random supernode ablation: {abl_err}")
                    import traceback

                    logger.error(traceback.format_exc())

            # Supernode hit-rate sweep: random masks conditioned on pruning a target fraction of supernodes.
            if getattr(self.config, "do_supernode_hit_rate_sweep", False):
                try:
                    cfg = getattr(self.config, "supernode_hit_rate_sweep", {}) or {}
                    hit_rates = cfg.get("hit_rates", None)
                    num_trials = int(cfg.get("num_trials", 3))
                    sweep_sparsity = float(cfg.get("sparsity", 0.5))
                    sweep_seed = cfg.get("seed", getattr(self.config, "seed", 0))
                    logger.info("Running supernode hit-rate sweep...")
                    sweep_results = self.compute_supernode_hit_rate_sweep(
                        scar_scores=scar_scores,
                        supernode_fraction=supernode_config.get("core_fraction", 0.01),
                        sparsity=sweep_sparsity,
                        hit_rates=hit_rates,
                        num_trials=num_trials,
                        seed=int(sweep_seed) if sweep_seed is not None else None,
                    )
                    results["supernode_hit_rate_sweep"] = sweep_results
                    logger.info("Supernode hit-rate sweep complete")
                except Exception as sweep_err:
                    logger.error(f"Failed supernode hit-rate sweep: {sweep_err}")
                    import traceback

                    logger.error(traceback.format_exc())

        if self.config.do_pruning_experiments:
            sparsity_levels = self.config.pruning_amounts

            # Get pruning strategies from config - these are now the metrics from metrics.enabled
            pruning_strategies = getattr(self.config, "pruning_strategies", None)
            if not pruning_strategies:
                # Fallback to single metric for backward compatibility
                pruning_strategies = [self.config.pruning_alignment_metric]

            # Augment pruning strategies with optional experiment-derived metrics.
            # - SCAR-optimal produces a precomputed per-channel score stored under metric="scar_optimal".
            # - Random-supernode ablation can expose additional precomputed metrics to evaluate.
            pruning_strategies = list(pruning_strategies)
            if getattr(self.config, "do_scar_optimal", False) and "scar_optimal" not in pruning_strategies:
                pruning_strategies.append("scar_optimal")

            random_ablation_cfg = results.get("random_supernode_ablation") if isinstance(results, dict) else None
            extra_ablation_metrics = (random_ablation_cfg or {}).get("pruning_metrics") if isinstance(random_ablation_cfg, dict) else None
            if isinstance(extra_ablation_metrics, list):
                for m in extra_ablation_metrics:
                    if isinstance(m, str) and m not in pruning_strategies:
                        pruning_strategies.append(m)

            hit_rate_cfg = results.get("supernode_hit_rate_sweep") if isinstance(results, dict) else None
            extra_hit_rate_metrics = (hit_rate_cfg or {}).get("pruning_metrics") if isinstance(hit_rate_cfg, dict) else None
            if isinstance(extra_hit_rate_metrics, list):
                for m in extra_hit_rate_metrics:
                    if isinstance(m, str) and m not in pruning_strategies:
                        pruning_strategies.append(m)

            # Check for single_strategy option (useful for memory-constrained LLM experiments)
            single_strategy = getattr(self.config, "single_strategy", None)
            if single_strategy:
                if single_strategy in pruning_strategies:
                    logger.info(f"Using single pruning strategy: {single_strategy}")
                    pruning_strategies = [single_strategy]
                else:
                    logger.warning(f"Requested single_strategy '{single_strategy}' not in pruning_strategies, using all")

            # Get selection modes
            selection_modes = self.config.pruning_selection_mode
            if isinstance(selection_modes, str):
                selection_modes = [selection_modes]

            baseline_ppl = results.get("evaluation", {}).get("baseline_perplexity", None)

            # For LLMs, save original state to CPU to avoid OOM
            # This is done once before all pruning experiments
            # First clear GPU cache to free any temporary allocations
            torch.cuda.empty_cache()
            logger.info("Saving original model state to CPU for pruning experiments...")
            original_state = {}
            for name, param in self.wrapped_model._model.named_parameters():
                # Use detach and clone to CPU directly, avoiding GPU allocation
                original_state[name] = param.data.detach().cpu().clone()
                # Periodically clear cache during large model copying
                if len(original_state) % 100 == 0:
                    torch.cuda.empty_cache()
            torch.cuda.empty_cache()  # Final cleanup
            logger.info(f"Saved {len(original_state)} parameter tensors to CPU")

            # Helper function to clear pruning state (hooks, buffers) from all modules
            def clear_all_pruning_state():
                for module in self.wrapped_model._model.modules():
                    if hasattr(module, "weight"):
                        # Remove pruning hooks
                        if hasattr(module, "_pruning_hook"):
                            module._pruning_hook.remove()
                            delattr(module, "_pruning_hook")
                        if hasattr(module, "_gradient_hook_handle"):
                            module._gradient_hook_handle.remove()
                            delattr(module, "_gradient_hook_handle")
                        # Remove pruning buffers
                        if hasattr(module, "_original_weight"):
                            delattr(module, "_original_weight")
                        if hasattr(module, "weight_mask"):
                            delattr(module, "weight_mask")

            # Helper function to restore weights from CPU state
            def restore_weights():
                # First clear all pruning state (hooks and buffers)
                clear_all_pruning_state()
                # Then restore the original weights
                for name, param in self.wrapped_model._model.named_parameters():
                    if name in original_state:
                        param.data.copy_(original_state[name].to(param.device))

            # Collect all pruning data for unified visualization
            all_pruning_data: Dict[str, Dict[str, List]] = {}

            # Iterate over all strategy/mode combinations
            for metric in pruning_strategies:
                # Check if we have importance scores for this metric
                # Some strategies (unstructured baseline reproductions) do not rely on
                # precomputed per-channel importance tensors in self.importance_scores.
                unstructured_baselines = {"wanda_unstructured", "sparsegpt_unstructured"}
                has_metric_scores = metric in unstructured_baselines or any(
                    metric in layer_scores for layer_scores in self.importance_scores.values()
                )
                if not has_metric_scores:
                    logger.warning(f"No importance scores computed for metric '{metric}', skipping pruning")
                    continue

                for mode in selection_modes:
                    logger.info(f"Pruning with strategy: {metric}, selection mode: {mode}")

                    # Strategy key for unified results structure (e.g., "rayleigh_quotient_low")
                    strategy_key = f"{metric}_{mode}"

                    # Collect pruning results for this strategy/mode combination
                    pruning_data = {
                        "sparsities": [],
                        "perplexities": [],
                    }

                    for sparsity in sparsity_levels:
                        # Restore original weights before applying new pruning level
                        restore_weights()

                        logger.info(f"  Applying pruning: sparsity={sparsity}, metric={metric}, mode={mode}")
                        masks = self.apply_pruning(sparsity=sparsity, mode=mode, metric=metric)

                        pruning_data["sparsities"].append(sparsity)

                        # Evaluate pruned model with configured metrics
                        llm_cfg = getattr(self.config, "llm", {}) or {}
                        eval_metrics = llm_cfg.get("evaluation_metrics") or getattr(self.config, "evaluation_metrics", ["perplexity"])
                        if isinstance(eval_metrics, str):
                            eval_metrics = [eval_metrics]

                        if self.config.do_perplexity_computation or "perplexity" in eval_metrics:
                            # Log which metrics are being evaluated
                            logger.info(f"Evaluating metrics: {eval_metrics}")

                            # Evaluate all requested metrics
                            eval_results = self.evaluate_multiple_metrics(metrics=eval_metrics, num_samples=self.config.evaluation_num_samples)

                            # Add loss (log of perplexity) if perplexity is available
                            if "perplexity" in eval_results and eval_results["perplexity"] is not None:
                                eval_results["loss"] = np.log(eval_results["perplexity"])

                            # Store perplexity for backward compatibility
                            pruned_ppl = eval_results.get("perplexity")
                            pruning_data["perplexities"].append(pruned_ppl)

                            # Store all metrics
                            for eval_metric, value in eval_results.items():
                                if eval_metric not in pruning_data:
                                    pruning_data[eval_metric] = []
                                pruning_data[eval_metric].append(value)

                            results["pruning_results"][f"{metric}_{mode}_sparsity_{sparsity}"] = {
                                **eval_results,  # Include all metrics
                                "sparsity": sparsity,
                                "num_pruned_layers": len(masks),
                                "metric": metric,
                                "mode": mode,
                                # Extra diagnostics for analysis (e.g., explain why some baselines collapse)
                                **(getattr(self, "_last_pruning_diagnostics", {}) or {}),
                            }
                        else:
                            pruning_data["perplexities"].append(None)

                    # Restore original weights after all sparsity levels for this strategy/mode
                    restore_weights()

                    # Store in unified format for visualization
                    # Filter out None perplexities
                    valid_data = [(s, p) for s, p in zip(pruning_data["sparsities"], pruning_data["perplexities"]) if p is not None]
                    if valid_data:
                        valid_sparsities, valid_perplexities = zip(*valid_data)
                        all_pruning_data[strategy_key] = {
                            "sparsities": list(valid_sparsities),
                            "perplexities": list(valid_perplexities),
                        }

                        # Also store other metrics if available
                        for eval_metric in pruning_data:
                            if eval_metric not in ["sparsities", "perplexities"] and pruning_data[eval_metric]:
                                # Filter to match valid sparsities
                                valid_metric_data = [
                                    v
                                    for s, v in zip(pruning_data["sparsities"], pruning_data[eval_metric])
                                    if s in valid_sparsities and v is not None
                                ]
                                if valid_metric_data:
                                    all_pruning_data[strategy_key][eval_metric] = valid_metric_data

            # Clean up original state from CPU memory
            del original_state
            torch.cuda.empty_cache()

            # Generate unified pruning visualizations using the centralized visualizer
            if getattr(self.config, "generate_plots", True) and all_pruning_data:
                try:
                    import matplotlib.pyplot as plt

                    plots_dir = Path(getattr(self.config, "plots_dir", Path(self.config.log_dir) / "plots"))
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    # Create pruning subfolder
                    pruning_plots_dir = plots_dir / "pruning"
                    pruning_plots_dir.mkdir(parents=True, exist_ok=True)

                    viz = UnifiedVisualizer()

                    # Calculate PRUNABLE parameters (only MLP layers being pruned, not total model params)
                    # This is important for accurate "remaining parameters" display
                    underlying_model = self._get_underlying_model()
                    prunable_params = 0
                    total_params = sum(p.numel() for p in self.wrapped_model._model.parameters())

                    # Count parameters in prunable MLP layers
                    for name, module in underlying_model.named_modules():
                        if any(pattern.replace("*", "") in name for pattern in ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]):
                            if hasattr(module, "weight"):
                                prunable_params += module.weight.numel()

                    logger.info(f"Total model parameters: {total_params:,}")
                    logger.info(f"Prunable MLP parameters: {prunable_params:,} ({100*prunable_params/total_params:.1f}% of total)")

                    # Use prunable params for the x-axis (more accurate representation)
                    display_params = prunable_params if prunable_params > 0 else total_params

                    # Get list of evaluation metrics to plot
                    llm_cfg = getattr(self.config, "llm", {}) or {}
                    eval_metrics = llm_cfg.get("evaluation_metrics") or getattr(self.config, "evaluation_metrics", ["perplexity"])
                    if isinstance(eval_metrics, str):
                        eval_metrics = [eval_metrics]

                    # Get baseline values for all metrics
                    baseline_values = results.get("evaluation", {})

                    # Generate plots for each evaluation metric
                    for eval_metric in eval_metrics:
                        metric_suffix = f"_{eval_metric}" if eval_metric != "perplexity" else ""

                        # Generate combined comparison plot with all strategies
                        comparison_path = pruning_plots_dir / f"pruning_comparison{metric_suffix}.png"
                        fig = viz.plot_llm_pruning_comparison(
                            results=all_pruning_data,
                            baseline_ppl=baseline_ppl if eval_metric == "perplexity" else None,
                            baseline_values=baseline_values,
                            metric=eval_metric,
                            title=f"LLM Pruning: Strategy Comparison ({eval_metric.replace('_', ' ').title()})",
                            save_path=comparison_path,
                            total_params=display_params,  # Use prunable params, not total
                        )
                        plt.close(fig)

                        # Also generate per-algorithm plots (grouping modes together)
                        algorithms = {}
                        for strategy_key, data in all_pruning_data.items():
                            parts = strategy_key.rsplit("_", 1)
                            if len(parts) == 2 and parts[1] in ["low", "high", "random"]:
                                algo = parts[0]
                            else:
                                algo = strategy_key
                            if algo not in algorithms:
                                algorithms[algo] = {}
                            algorithms[algo][strategy_key] = data

                        for algo, algo_data in algorithms.items():
                            algo_path = pruning_plots_dir / f"pruning_{algo}_comparison{metric_suffix}.png"
                            fig = viz.plot_llm_pruning_comparison(
                                results=algo_data,
                                baseline_ppl=baseline_ppl if eval_metric == "perplexity" else None,
                                baseline_values=baseline_values,
                                metric=eval_metric,
                                title=f"LLM Pruning: {algo.replace('_', ' ').title()} ({eval_metric.replace('_', ' ').title()})",
                                save_path=algo_path,
                                total_params=display_params,  # Use prunable params
                            )
                            plt.close(fig)

                    logger.info(f"Generated pruning visualizations in {plots_dir}")

                except Exception as e:
                    logger.error(f"Failed to generate pruning visualizations: {e}")

        # ------------------------------------------------------------------
        # Mechanism diagnostic figures (supernodes + halo structure)
        # ------------------------------------------------------------------
        if getattr(self.config, "generate_plots", True):
            try:
                from alignment.analysis.visualization.llm_mechanism_plots import (
                    plot_bus_concentration,
                    plot_conditional_halo_ablation,
                    plot_halo_structure,
                    plot_loss_proxy_concentration,
                    plot_lp_vs_magnitude_controls,
                    plot_read_halo_dependence_summary,
                    plot_supernode_halo_summary,
                )

                plots_dir = Path(getattr(self.config, "plots_dir", Path(self.config.log_dir) / "plots"))
                report_dir = plots_dir / "report"
                report_dir.mkdir(parents=True, exist_ok=True)

                # 1) Loss proxy concentration for a representative layer
                rho = float((getattr(self.config, "supernode", {}) or {}).get("core_fraction", 0.01))
                down_layers = sorted([k for k in scar_scores.keys() if "mlp.down_proj" in k])
                if down_layers:
                    # Choose a stable "middle" layer as representative
                    mid_layer = down_layers[len(down_layers) // 2]
                    lp = scar_scores.get(mid_layer, {}).get("scar_loss_proxy")
                    if lp is not None:
                        plot_loss_proxy_concentration(
                            loss_proxy=lp,
                            rho=rho,
                            layer_label=mid_layer,
                            save_path=report_dir / "fig_supernode_distribution.png",
                            dpi=getattr(self.config, "plot_dpi", 300),
                        )

                # 2) Halo structure (global): aggregate across many layers for a cleaner story
                if down_layers:
                    conn_all = []
                    prot_all = []
                    red_all = []
                    halo_all = []
                    super_all = []
                    for ln in down_layers:
                        layer_scores = self.importance_scores.get(ln, {})
                        conn = layer_scores.get("connectivity_score")
                        prot = layer_scores.get("protection_score")
                        red = layer_scores.get("redundancy_to_core")
                        halo_mask = layer_scores.get("halo_mask")
                        super_mask = layer_scores.get("supernode_mask")
                        if conn is None or prot is None or red is None or halo_mask is None or super_mask is None:
                            continue
                        # Ensure consistent shapes
                        try:
                            if conn.numel() == 0 or conn.numel() != prot.numel() or conn.numel() != halo_mask.numel():
                                continue
                            if red.numel() != conn.numel() or super_mask.numel() != conn.numel():
                                continue
                        except Exception:
                            continue

                        conn_all.append(conn.detach().cpu())
                        prot_all.append(prot.detach().cpu())
                        red_all.append(red.detach().cpu())
                        halo_all.append(halo_mask.detach().cpu())
                        super_all.append(super_mask.detach().cpu())

                    if conn_all:
                        conn_cat = torch.cat(conn_all, dim=0)
                        prot_cat = torch.cat(prot_all, dim=0)
                        red_cat = torch.cat(red_all, dim=0)
                        halo_cat = torch.cat(halo_all, dim=0)
                        super_cat = torch.cat(super_all, dim=0)

                        plot_halo_structure(
                            conn=conn_cat,
                            redundancy_to_core=red_cat,
                            protect=prot_cat,
                            super_mask=super_cat,
                            halo_mask=halo_cat,
                            layer_label="All layers (aggregated)",
                            save_path=report_dir / "fig_halo_structure.png",
                            dpi=getattr(self.config, "plot_dpi", 300),
                        )

                # 2b) Halo structure (example layer): keep a representative layer for debugging/supplementary
                if down_layers:
                    mid_layer = down_layers[len(down_layers) // 2]
                    layer_scores = self.importance_scores.get(mid_layer, {})
                    conn = layer_scores.get("connectivity_score")
                    prot = layer_scores.get("protection_score")
                    red = layer_scores.get("redundancy_to_core")
                    halo_mask = layer_scores.get("halo_mask")
                    super_mask = layer_scores.get("supernode_mask")
                    if conn is not None and prot is not None and red is not None and halo_mask is not None and super_mask is not None:
                        plot_halo_structure(
                            conn=conn,
                            redundancy_to_core=red,
                            protect=prot,
                            super_mask=super_mask,
                            halo_mask=halo_mask,
                            layer_label=mid_layer,
                            save_path=report_dir / "fig_halo_structure_layer.png",
                            dpi=getattr(self.config, "plot_dpi", 300),
                        )

                # 3) Supernode mass ratio across layers + halo redundancy summary
                try:
                    halo_agg = (results.get("halo_analysis") or {}).get("aggregate") or {}
                    # Compute top-rho mass ratio per layer from scar_loss_proxy
                    layer_idxs: List[int] = []
                    ratios: List[float] = []
                    for ln in down_layers:
                        lp = scar_scores.get(ln, {}).get("scar_loss_proxy")
                        if lp is None:
                            continue
                        lp_cpu = lp.detach().float().cpu()
                        m = int(lp_cpu.numel())
                        if m <= 0:
                            continue
                        k = max(1, int(round(rho * m)))
                        top = torch.topk(lp_cpu, k=k, largest=True).values
                        denom = float(lp_cpu.sum().item()) if float(lp_cpu.sum().item()) > 0 else 1.0
                        ratio = float(top.sum().item()) / denom
                        try:
                            idx = int(ln.split("layers.")[-1].split(".")[0])
                        except Exception:
                            idx = len(layer_idxs)
                        layer_idxs.append(idx)
                        ratios.append(ratio)

                    if layer_idxs and halo_agg:
                        # Sort by layer index for plotting
                        order = np.argsort(np.asarray(layer_idxs))
                        layer_idxs_sorted = [layer_idxs[i] for i in order]
                        ratios_sorted = [ratios[i] for i in order]
                        plot_supernode_halo_summary(
                            layer_indices=layer_idxs_sorted,
                            top_mass_ratios=ratios_sorted,
                            halo_aggregate=halo_agg,
                            rho=rho,
                            save_path=report_dir / "fig_supernode_analysis.png",
                            dpi=getattr(self.config, "plot_dpi", 300),
                        )
                except Exception as _summary_err:
                    logger.debug(f"Paper summary plot skipped: {_summary_err}")

                # 4) Disentangle LP from simple magnitude controls (representative layer)
                try:
                    if down_layers:
                        mid_layer = down_layers[len(down_layers) // 2]
                        lp = scar_scores.get(mid_layer, {}).get("scar_loss_proxy")
                        ap = scar_scores.get(mid_layer, {}).get("scar_activation_power")
                        if lp is not None and ap is not None:
                            import re

                            module_dict = dict(self.model.named_modules())
                            m = re.search(r"layers\.(\d+)", mid_layer)
                            layer_idx = int(m.group(1)) if m else None
                            up_name = f"model.layers.{layer_idx}.mlp.up_proj" if layer_idx is not None else None
                            gate_name = f"model.layers.{layer_idx}.mlp.gate_proj" if layer_idx is not None else None

                            def _resolve(name: Optional[str]):
                                if not name:
                                    return None
                                if name in module_dict:
                                    return module_dict[name]
                                if name.startswith("model.") and name[len("model.") :] in module_dict:
                                    return module_dict[name[len("model.") :]]
                                alt = "model.model." + name
                                if alt in module_dict:
                                    return module_dict[alt]
                                for k, v in module_dict.items():
                                    if k.endswith(name):
                                        return v
                                return None

                            down_mod = _resolve(mid_layer)
                            up_mod = _resolve(up_name)
                            gate_mod = _resolve(gate_name)

                            dn = None
                            un = None
                            gn = None
                            try:
                                if down_mod is not None and hasattr(down_mod, "weight"):
                                    Wd = down_mod.weight.detach().float()
                                    dn = torch.sqrt(torch.sum(Wd * Wd, dim=0)).detach().cpu()
                            except Exception:
                                dn = None
                            try:
                                if up_mod is not None and hasattr(up_mod, "weight"):
                                    Wu = up_mod.weight.detach().float()
                                    un = torch.sqrt(torch.sum(Wu * Wu, dim=1)).detach().cpu()
                            except Exception:
                                un = None
                            try:
                                if gate_mod is not None and hasattr(gate_mod, "weight"):
                                    Wg = gate_mod.weight.detach().float()
                                    gn = torch.sqrt(torch.sum(Wg * Wg, dim=1)).detach().cpu()
                            except Exception:
                                gn = None

                            # Store an across-layer correlation summary (small; used for summary tables/claims).
                            try:

                                def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
                                    a = np.asarray(a, dtype=np.float64).reshape(-1)
                                    b = np.asarray(b, dtype=np.float64).reshape(-1)
                                    if a.size == 0 or b.size == 0 or a.size != b.size:
                                        return float("nan")
                                    ra = a.argsort().argsort().astype(np.float64)
                                    rb = b.argsort().argsort().astype(np.float64)
                                    ra -= ra.mean()
                                    rb -= rb.mean()
                                    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
                                    rho = float((ra @ rb) / denom)
                                    return rho if np.isfinite(rho) else float("nan")

                                li_list: List[int] = []
                                rho_ap_list: List[float] = []
                                rho_dn_list: List[float] = []
                                rho_un_list: List[float] = []
                                rho_gn_list: List[float] = []

                                eps = 1e-12

                                for ln in down_layers:
                                    lp_t = scar_scores.get(ln, {}).get("scar_loss_proxy")
                                    ap_t = scar_scores.get(ln, {}).get("scar_activation_power")
                                    if lp_t is None or ap_t is None:
                                        continue

                                    lp_np = lp_t.detach().float().cpu().numpy().reshape(-1)
                                    ap_np = ap_t.detach().float().cpu().numpy().reshape(-1)
                                    n = int(min(lp_np.size, ap_np.size))
                                    if n <= 1:
                                        continue

                                    x = np.log10(np.maximum(lp_np[:n], 0.0) + eps)
                                    y_ap = np.log10(np.maximum(ap_np[:n], 0.0) + eps)

                                    m2 = re.search(r"layers\.(\d+)", ln)
                                    li = int(m2.group(1)) if m2 else len(li_list)

                                    # Weight-norm controls (best-effort; can be NaN if modules are sharded/unavailable)
                                    dn_rho = float("nan")
                                    un_rho = float("nan")
                                    gn_rho = float("nan")
                                    try:
                                        down_mod2 = _resolve(ln)
                                        if down_mod2 is not None and hasattr(down_mod2, "weight"):
                                            Wd2 = down_mod2.weight.detach().float()
                                            dn2 = torch.sqrt(torch.sum(Wd2 * Wd2, dim=0)).detach().cpu().numpy().reshape(-1)[:n]
                                            dn_rho = _spearman_np(x, np.log10(np.maximum(dn2, 0.0) + eps))
                                    except Exception:
                                        pass
                                    try:
                                        up_name2 = f"model.layers.{li}.mlp.up_proj"
                                        up_mod2 = _resolve(up_name2)
                                        if up_mod2 is not None and hasattr(up_mod2, "weight"):
                                            Wu2 = up_mod2.weight.detach().float()
                                            un2 = torch.sqrt(torch.sum(Wu2 * Wu2, dim=1)).detach().cpu().numpy().reshape(-1)[:n]
                                            un_rho = _spearman_np(x, np.log10(np.maximum(un2, 0.0) + eps))
                                    except Exception:
                                        pass
                                    try:
                                        gate_name2 = f"model.layers.{li}.mlp.gate_proj"
                                        gate_mod2 = _resolve(gate_name2)
                                        if gate_mod2 is not None and hasattr(gate_mod2, "weight"):
                                            Wg2 = gate_mod2.weight.detach().float()
                                            gn2 = torch.sqrt(torch.sum(Wg2 * Wg2, dim=1)).detach().cpu().numpy().reshape(-1)[:n]
                                            gn_rho = _spearman_np(x, np.log10(np.maximum(gn2, 0.0) + eps))
                                    except Exception:
                                        pass

                                    li_list.append(li)
                                    rho_ap_list.append(_spearman_np(x, y_ap))
                                    rho_dn_list.append(dn_rho)
                                    rho_un_list.append(un_rho)
                                    rho_gn_list.append(gn_rho)

                                if li_list:
                                    order2 = np.argsort(np.asarray(li_list))
                                    li_sorted = [li_list[i] for i in order2]
                                    ap_sorted = [rho_ap_list[i] for i in order2]
                                    dn_sorted = [rho_dn_list[i] for i in order2]
                                    un_sorted = [rho_un_list[i] for i in order2]
                                    gn_sorted = [rho_gn_list[i] for i in order2]

                                    def _summ(vals: List[float]) -> Dict[str, float]:
                                        a = np.asarray(vals, dtype=np.float64)
                                        a = a[np.isfinite(a)]
                                        if a.size == 0:
                                            return {"median": float("nan"), "min": float("nan"), "max": float("nan")}
                                        return {"median": float(np.median(a)), "min": float(np.min(a)), "max": float(np.max(a))}

                                    results["lp_magnitude_controls"] = {
                                        "layer_indices": li_sorted,
                                        "spearman_log_lp_log_activation_power": ap_sorted,
                                        "spearman_log_lp_log_downproj_col_norm": dn_sorted,
                                        "spearman_log_lp_log_upproj_row_norm": un_sorted,
                                        "spearman_log_lp_log_gateproj_row_norm": gn_sorted,
                                        "summary": {
                                            "log_lp_vs_log_activation_power": _summ(ap_sorted),
                                            "log_lp_vs_log_downproj_col_norm": _summ(dn_sorted),
                                            "log_lp_vs_log_upproj_row_norm": _summ(un_sorted),
                                            "log_lp_vs_log_gateproj_row_norm": _summ(gn_sorted),
                                        },
                                    }
                            except Exception as _lp_ctrl_sum_err:
                                logger.debug(f"LP-vs-magnitude summary skipped: {_lp_ctrl_sum_err}")

                            plot_lp_vs_magnitude_controls(
                                loss_proxy=lp,
                                activation_power=ap,
                                downproj_col_norm=dn,
                                upproj_row_norm=un,
                                gateproj_row_norm=gn,
                                layer_label=mid_layer,
                                rho=rho,
                                save_path=report_dir / "fig_lp_vs_magnitude.png",
                                dpi=getattr(self.config, "plot_dpi", 300),
                            )
                except Exception as _lp_ctrl_err:
                    logger.debug(f"LP-vs-magnitude figure skipped: {_lp_ctrl_err}")

                # 5) Bus concentration: low-dimensional write support (supernodes vs random baseline)
                try:
                    import re

                    module_dict = dict(self.model.named_modules())

                    def _resolve(name: str):
                        if name in module_dict:
                            return module_dict[name]
                        if name.startswith("model.") and name[len("model.") :] in module_dict:
                            return module_dict[name[len("model.") :]]
                        alt = "model.model." + name
                        if alt in module_dict:
                            return module_dict[alt]
                        for k, v in module_dict.items():
                            if k.endswith(name):
                                return v
                        return None

                    rng = np.random.default_rng(0)
                    layer_idx_list: List[int] = []
                    deff_super: List[float] = []
                    deff_rand: List[float] = []
                    curves: Dict[int, Dict[str, Any]] = {}

                    show_set: set = set()
                    if down_layers:
                        show_set = {down_layers[0], down_layers[len(down_layers) // 2], down_layers[-1]}

                    def _d_eff(vec: np.ndarray) -> float:
                        v = np.asarray(vec, dtype=np.float64).reshape(-1)
                        v = np.maximum(v, 0.0)
                        s = float(v.sum())
                        if not np.isfinite(s) or s <= 0:
                            return 0.0
                        p = v / s
                        p = p[p > 0]
                        H = -float(np.sum(p * np.log(p)))
                        return float(np.exp(H))

                    for ln in down_layers:
                        m = re.search(r"layers\.(\d+)", ln)
                        li = int(m.group(1)) if m else None
                        if li is None:
                            continue

                        lp = scar_scores.get(ln, {}).get("scar_loss_proxy")
                        if lp is None:
                            continue
                        lp_cpu = lp.detach().float().cpu()
                        m_int = int(lp_cpu.numel())
                        if m_int <= 0:
                            continue

                        num_super = max(1, int(round(float(rho) * float(m_int))))
                        super_idx = torch.topk(lp_cpu, k=num_super, largest=True).indices.to(dtype=torch.long)

                        down_mod = _resolve(ln)
                        if down_mod is None or not hasattr(down_mod, "weight"):
                            continue

                        W = down_mod.weight.detach()
                        a = torch.abs(W.index_select(dim=1, index=super_idx.to(device=W.device))).sum(dim=1).float().cpu().numpy()

                        rand_idx_np = rng.choice(m_int, size=num_super, replace=False)
                        rand_idx = torch.as_tensor(rand_idx_np, dtype=torch.long, device=W.device)
                        a_r = torch.abs(W.index_select(dim=1, index=rand_idx)).sum(dim=1).float().cpu().numpy()

                        layer_idx_list.append(li)
                        deff_super.append(_d_eff(a))
                        deff_rand.append(_d_eff(a_r))

                        if ln in show_set:
                            aa = np.sort(a.astype(np.float64))[::-1]
                            bb = np.sort(a_r.astype(np.float64))[::-1]
                            denom_a = float(aa.sum()) if float(aa.sum()) > 0 else 1.0
                            denom_b = float(bb.sum()) if float(bb.sum()) > 0 else 1.0
                            cum_a = np.cumsum(aa) / denom_a
                            cum_b = np.cumsum(bb) / denom_b
                            frac = (np.arange(aa.size) + 1) / float(max(1, aa.size))
                            curves[li] = {"frac": frac, "cum_super": cum_a, "cum_rand": cum_b}

                    if layer_idx_list:
                        order = np.argsort(np.asarray(layer_idx_list))
                        layer_idx_sorted = [layer_idx_list[i] for i in order]
                        deff_super_sorted = [deff_super[i] for i in order]
                        deff_rand_sorted = [deff_rand[i] for i in order]
                        results["bus_concentration"] = {
                            "layer_indices": layer_idx_sorted,
                            "d_eff_super": deff_super_sorted,
                            "d_eff_random": deff_rand_sorted,
                        }
                        plot_bus_concentration(
                            layer_indices=layer_idx_sorted,
                            d_eff_super=deff_super_sorted,
                            d_eff_random=deff_rand_sorted,
                            curves=curves,
                            save_path=report_dir / "fig_bus_concentration.png",
                            dpi=getattr(self.config, "plot_dpi", 300),
                        )
                except Exception as _bus_err:
                    logger.debug(f"Bus concentration figure skipped: {_bus_err}")

                # 6) Read-halo dependence summary (if computed during supernode analysis)
                try:
                    sn = results.get("supernode_analysis") or {}
                    layer_idx_list: List[int] = []
                    rho_list: List[float] = []
                    mh_list: List[float] = []
                    mr_list: List[float] = []

                    for _ln, rec in sn.items():
                        if not isinstance(rec, dict):
                            continue
                        rh = rec.get("next_layer_read_halo") or {}
                        if not isinstance(rh, dict):
                            continue
                        dep = rh.get("dependence_u")
                        if not isinstance(dep, dict):
                            continue
                        try:
                            li = int(rh.get("target_layer_idx"))
                        except Exception:
                            continue
                        try:
                            rr = float(dep.get("spearman_readconn_vs_mean_abs_delta_u", float("nan")))
                        except Exception:
                            rr = float("nan")
                        mabs = dep.get("mean_abs_delta_u") or {}
                        try:
                            mh = float(mabs.get("read_halo"))
                            mr = float(mabs.get("random"))
                        except Exception:
                            continue
                        if not (np.isfinite(rr) and np.isfinite(mh) and np.isfinite(mr)):
                            continue
                        layer_idx_list.append(li)
                        rho_list.append(rr)
                        mh_list.append(mh)
                        mr_list.append(mr)

                    if layer_idx_list:
                        order = np.argsort(np.asarray(layer_idx_list))
                        layer_idx_sorted = [layer_idx_list[i] for i in order]
                        rho_sorted = [rho_list[i] for i in order]
                        mh_sorted = [mh_list[i] for i in order]
                        mr_sorted = [mr_list[i] for i in order]
                        results["read_halo_dependence"] = {
                            "layer_indices": layer_idx_sorted,
                            "spearman_readconn_vs_mean_abs_delta_u": rho_sorted,
                            "mean_abs_delta_u_read_halo": mh_sorted,
                            "mean_abs_delta_u_random": mr_sorted,
                        }
                        plot_read_halo_dependence_summary(
                            layer_indices=layer_idx_sorted,
                            spearman_rho=rho_sorted,
                            read_halo_mean_abs_delta_u=mh_sorted,
                            random_mean_abs_delta_u=mr_sorted,
                            save_path=report_dir / "fig_read_halo_dependence.png",
                            dpi=getattr(self.config, "plot_dpi", 300),
                        )
                except Exception as _rh_dep_err:
                    logger.debug(f"Read-halo dependence summary skipped: {_rh_dep_err}")

                # 7) Conditional halo ablation (if computed)
                try:
                    ca = results.get("conditional_halo_ablation") or {}
                    layers_rec = ca.get("layers") if isinstance(ca, dict) else None
                    if isinstance(layers_rec, list) and layers_rec:
                        layer_idx_list: List[int] = []
                        dh: List[float] = []
                        dm: List[float] = []
                        ds: List[float] = []
                        db: List[float] = []

                        for rec in layers_rec:
                            if not isinstance(rec, dict):
                                continue
                            try:
                                li = int(rec.get("layer_idx"))
                            except Exception:
                                continue
                            dn = rec.get("delta_nll") or {}
                            if not isinstance(dn, dict):
                                continue
                            try:
                                v_h = float(dn.get("halo_subset"))
                                v_m = float(dn.get("matched_non_halo_subset"))
                                v_s = float(dn.get("supernodes"))
                                v_b = float(dn.get("supernodes_plus_halo"))
                            except Exception:
                                continue
                            if not (np.isfinite(v_h) and np.isfinite(v_m) and np.isfinite(v_s) and np.isfinite(v_b)):
                                continue
                            layer_idx_list.append(li)
                            dh.append(v_h)
                            dm.append(v_m)
                            ds.append(v_s)
                            db.append(v_b)

                        if layer_idx_list:
                            order = np.argsort(np.asarray(layer_idx_list))
                            layer_idx_sorted = [layer_idx_list[i] for i in order]
                            dh_sorted = [dh[i] for i in order]
                            dm_sorted = [dm[i] for i in order]
                            ds_sorted = [ds[i] for i in order]
                            db_sorted = [db[i] for i in order]

                            plot_conditional_halo_ablation(
                                layer_indices=layer_idx_sorted,
                                delta_nll_halo=dh_sorted,
                                delta_nll_matched=dm_sorted,
                                delta_nll_supernodes=ds_sorted,
                                delta_nll_halo_plus_supernodes=db_sorted,
                                save_path=report_dir / "fig_halo_conditional_ablation.png",
                                dpi=getattr(self.config, "plot_dpi", 300),
                            )
                except Exception as _ca_plot_err:
                    logger.debug(f"Conditional ablation plot skipped: {_ca_plot_err}")

            except Exception as e:
                logger.warning(f"Failed to generate mechanism figures: {e}")

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
                f"plot_neuron_output_weights_histogram: neuron_index {neuron_index} " f"out of range for layer '{layer_name}' with width {W.shape[1]}"
            )
            return {}

        outgoing = W[:, neuron_index]
        magnitudes = outgoing.abs()
        k = min(5, magnitudes.numel())
        top_idxs, _ = torch.topk(magnitudes, k=k)
        outgoing[top_idxs]

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

    def compute_scar_optimal(
        self,
        scar_scores: Dict[str, Dict[str, Any]],
        num_validation_samples: int = 32,
        sparsity: float = 0.3,
        search_granularity: int = 5,
        plots_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Compute SCAR-optimal: learned weighted combination of SCAR components.

        This performs a grid search over weights for:
        - Loss Proxy (LP)
        - Activation Power
        - Taylor (first-order sensitivity)
        - Protection score (from halo analysis)

        The optimal weights are found by minimizing perplexity on a validation set.

        Args:
            scar_scores: Pre-computed SCAR scores
            num_validation_samples: Samples for validation PPL
            sparsity: Sparsity level for grid search (default 30%)
            search_granularity: Number of weight values to try (5 = [0, 0.25, 0.5, 0.75, 1])
            plots_dir: Directory to save analysis plots

        Returns:
            Dict with optimal weights, per-layer weights, and final scores
        """
        import itertools
        import re

        logger.info("=" * 60)
        logger.info("Computing SCAR-optimal: Learned Component Weights")
        logger.info("=" * 60)

        # Weight values to search
        weight_values = [i / (search_granularity - 1) for i in range(search_granularity)]
        logger.info(f"Weight values: {weight_values}")

        # Get available components per layer
        layer_names = [ln for ln in scar_scores.keys() if "mlp.down_proj" in ln]
        if not layer_names:
            logger.warning("No SCAR scores found")
            return {}

        # Check which components are available
        sample_layer = layer_names[0]
        sample_metrics = scar_scores[sample_layer]
        available_components = []
        for comp in ["scar_loss_proxy", "scar_activation_power", "scar_taylor", "scar_curvature"]:
            if comp in sample_metrics:
                available_components.append(comp)

        # Also check importance_scores for protection
        layer_imp = self.importance_scores.get(sample_layer.replace("model.layers", "model.model.layers"), {})
        if "supernode_protection_score" in layer_imp or "protection_score" in layer_imp:
            available_components.append("protection")

        logger.info(f"Available components: {available_components}")

        # Generate weight combinations (normalized to sum to 1)
        n_components = len(available_components)
        weight_combos = []
        for combo in itertools.product(weight_values, repeat=n_components):
            if sum(combo) > 0:  # Avoid all-zero
                normalized = tuple(w / sum(combo) for w in combo)
                if normalized not in weight_combos:
                    weight_combos.append(normalized)

        logger.info(f"Testing {len(weight_combos)} weight combinations")

        # Get validation data
        val_texts = []
        if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
            val_texts = list(self.dataset.texts)[:num_validation_samples]
        if not val_texts:
            logger.warning("No validation texts available")
            return {}

        # Prepare model
        device = next(self.model.parameters()).device

        # Quick PPL evaluation function
        def quick_ppl(scores_dict, amount_to_prune: float) -> float:
            """
            Quick PPL evaluation for SCAR-optimal grid search.

            Notes:
            - This is intentionally lightweight (few samples, short context).
            - We prune only FFN `down_proj` *columns* according to the provided per-channel scores.
            """
            try:
                module_dict = dict(self.model.named_modules())

                # Apply pruning masks temporarily (store/restore weights)
                original_weights: Dict[str, torch.Tensor] = {}

                for layer_name, scores in scores_dict.items():
                    if "down_proj" not in layer_name:
                        continue

                    module_path = layer_name.replace("model.layers", "model.model.layers")
                    module = module_dict.get(module_path)
                    if module is None or not hasattr(module, "weight"):
                        continue

                    s = scores.detach().to(device=module.weight.device, dtype=torch.float32).flatten()
                    if s.numel() == 0:
                        continue

                    k = int(float(amount_to_prune) * float(s.numel()))
                    if k <= 0:
                        continue

                    # Prune LOW scores (keep high-scoring channels)
                    _, idx = torch.topk(s, k, largest=False)
                    keep = torch.ones(s.numel(), dtype=torch.bool, device=s.device)
                    keep[idx] = False  # False = prune

                    # Save & apply: zero out pruned *columns*
                    original_weights[module_path] = module.weight.data.clone()
                    module.weight.data[:, ~keep] = 0

                if not original_weights:
                    return float("inf")

                # Compute PPL
                total_loss = 0.0
                total_tokens = 0
                self.model.eval()
                with torch.no_grad():
                    for text in val_texts[:8]:
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                        total_loss += float(outputs.loss.item()) * int(inputs["input_ids"].numel())
                        total_tokens += int(inputs["input_ids"].numel())

                ppl = float(np.exp(total_loss / max(total_tokens, 1)))
                return ppl
            except Exception as e:
                logger.warning(f"PPL eval failed: {e}")
                return float("inf")
            finally:
                # Restore weights
                try:
                    module_dict = dict(self.model.named_modules())
                    for name, weight in original_weights.items():
                        module = module_dict.get(name)
                        if module is not None and hasattr(module, "weight"):
                            module.weight.data = weight
                except Exception:
                    pass

        # Grid search
        best_ppl = float("inf")
        best_weights = None
        results_log = []

        for i, weights in enumerate(weight_combos):
            if i % 10 == 0:
                logger.info(f"Testing combination {i+1}/{len(weight_combos)}...")

            # Compute combined scores
            combined_scores = {}
            for layer_name in layer_names:
                layer_metrics = scar_scores[layer_name]
                layer_imp = self.importance_scores.get(layer_name.replace("model.layers", "model.model.layers"), {})

                # Get component tensors
                components = []
                for comp in available_components:
                    if comp == "protection":
                        val = layer_imp.get("supernode_protection_score") or layer_imp.get("protection_score")
                    else:
                        val = layer_metrics.get(comp)

                    if val is None:
                        components.append(None)
                    elif isinstance(val, dict) and "scores" in val:
                        components.append(torch.tensor(val["scores"]))
                    elif torch.is_tensor(val):
                        components.append(val.float().cpu())
                    else:
                        components.append(None)

                # Skip if any component missing
                if any(c is None for c in components):
                    continue

                # Normalize each component to [0, 1]
                normalized = []
                for c in components:
                    c_min, c_max = c.min(), c.max()
                    if c_max > c_min:
                        normalized.append((c - c_min) / (c_max - c_min))
                    else:
                        normalized.append(torch.ones_like(c))

                # Weighted combination
                combined = sum(w * n for w, n in zip(weights, normalized))
                combined_scores[layer_name] = combined

            if not combined_scores:
                continue

            # Evaluate
            ppl = quick_ppl(combined_scores, sparsity)
            results_log.append((weights, ppl))

            if ppl < best_ppl:
                best_ppl = ppl
                best_weights = weights
                logger.info(f"  New best: weights={[f'{w:.2f}' for w in weights]}, PPL={ppl:.2f}")

        # Store optimal weights
        weight_dict = {comp: w for comp, w in zip(available_components, best_weights)} if best_weights else {}

        logger.info("\n" + "=" * 60)
        logger.info("SCAR-optimal Results")
        logger.info("=" * 60)
        logger.info(f"Best weights: {weight_dict}")
        logger.info(f"Best PPL at {sparsity*100:.0f}% sparsity: {best_ppl:.2f}")

        # Compute final optimal scores
        optimal_scores = {}
        for layer_name in layer_names:
            layer_metrics = scar_scores[layer_name]
            layer_imp = self.importance_scores.get(layer_name.replace("model.layers", "model.model.layers"), {})

            components = []
            for comp in available_components:
                if comp == "protection":
                    val = layer_imp.get("supernode_protection_score") or layer_imp.get("protection_score")
                else:
                    val = layer_metrics.get(comp)

                if val is None:
                    continue
                elif isinstance(val, dict) and "scores" in val:
                    components.append(torch.tensor(val["scores"]))
                elif torch.is_tensor(val):
                    components.append(val.float().cpu())

            if len(components) == len(available_components) and best_weights:
                # Normalize and combine
                normalized = []
                for c in components:
                    c_min, c_max = c.min(), c.max()
                    if c_max > c_min:
                        normalized.append((c - c_min) / (c_max - c_min))
                    else:
                        normalized.append(torch.ones_like(c))

                combined = sum(w * n for w, n in zip(best_weights, normalized))
                optimal_scores[layer_name] = combined

                # Store in importance_scores for *all* FFN projections in this layer, so pruning can see it.
                try:
                    m = re.search(r"layers\.(\d+)\\.mlp", layer_name)
                    layer_idx = int(m.group(1)) if m else None
                except Exception:
                    layer_idx = None

                # Default: store on the down_proj key (legacy)
                imp_key = layer_name.replace("model.layers", "model.model.layers")
                if imp_key not in self.importance_scores:
                    self.importance_scores[imp_key] = {}
                self.importance_scores[imp_key]["scar_optimal"] = combined

                if layer_idx is not None:
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        k = f"model.model.layers.{layer_idx}.mlp.{proj}"
                        if k not in self.importance_scores:
                            self.importance_scores[k] = {}
                        self.importance_scores[k]["scar_optimal"] = combined

        # Save plot if requested
        if plots_dir and results_log:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ppls = [r[1] for r in results_log if r[1] < 1000]
            ax.hist(ppls, bins=30, alpha=0.7, color="blue")
            ax.axvline(x=best_ppl, color="red", linestyle="--", linewidth=2, label=f"Best: {best_ppl:.1f}")
            ax.set_xlabel("Perplexity", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("SCAR-optimal Grid Search Results", fontsize=14)
            ax.legend()

            fig.tight_layout()
            fig.savefig(plots_dir / "scar_optimal_search.png", dpi=150)
            plt.close(fig)
            logger.info(f"Saved plot: {plots_dir / 'scar_optimal_search.png'}")

        return {
            "optimal_weights": weight_dict,
            "best_ppl": best_ppl,
            "sparsity": sparsity,
            "components": available_components,
            "search_results": results_log,
            "optimal_scores": optimal_scores,
        }

    def compute_random_supernode_ablation(
        self,
        scar_scores: Dict[str, Dict[str, Any]],
        supernode_fraction: float = 0.01,
        num_trials: int = 5,
        sparsity: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Ablation: What if we used RANDOM supernodes instead of LP-identified ones?

        This tests whether correct supernode identification matters, or if
        any sparse set of "protected" channels works equally well.

        Args:
            scar_scores: Pre-computed SCAR scores (for comparison)
            supernode_fraction: Fraction to treat as supernodes
            num_trials: Number of random trials
            sparsity: Sparsity level for evaluation

        Returns:
            Dict describing the injected pruning metrics and the sampled indices.

        Notes:
            This function *injects* additional precomputed per-channel pruning scores into
            `self.importance_scores` under synthetic metric names (returned in
            `results["pruning_metrics"]`). The standard pruning loop can then evaluate these
            metrics and write PPL into `results["pruning_results"]`.
        """
        import re

        logger.info("=" * 60)
        logger.info("Random Supernode Ablation")
        logger.info("=" * 60)
        logger.info("Testing whether LP-based supernode identification matters")
        logger.info(f"Comparing LP-supernodes vs {num_trials} random supernode trials")

        layer_names = [ln for ln in scar_scores.keys() if "mlp.down_proj" in ln]
        if not layer_names:
            return {}

        # Map layer index -> projection module keys in importance_scores (so pruning can see the metric everywhere).
        layer_to_proj_keys: Dict[int, List[str]] = {}
        for k in self.importance_scores.keys():
            m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", k)
            if m:
                layer_to_proj_keys.setdefault(int(m.group(1)), []).append(k)

        # Parse LP tensors per layer
        lp_tensors: Dict[str, torch.Tensor] = {}
        for layer_name in layer_names:
            layer_metrics = scar_scores.get(layer_name) or {}
            lp = layer_metrics.get("scar_loss_proxy")
            if isinstance(lp, dict) and "scores" in lp:
                lp_tensor = torch.tensor(lp["scores"], dtype=torch.float32)
            elif torch.is_tensor(lp):
                lp_tensor = lp.float().detach().cpu()
            else:
                continue
            if lp_tensor.numel() > 0:
                lp_tensors[layer_name] = lp_tensor

        if not lp_tensors:
            logger.warning("No LP scores found")
            return {}

        intermediate_dim = next(iter(lp_tensors.values())).numel()
        num_supernodes = max(1, int(supernode_fraction * intermediate_dim))
        logger.info(f"Intermediate dim: {intermediate_dim}, supernodes per layer: {num_supernodes}")

        base_seed = int(getattr(self.config, "seed", 0) or 0)
        lp_metric = "random_supernode_ablation_lp"
        random_metrics = [f"random_supernode_ablation_random_{t}" for t in range(int(num_trials))]
        pruning_metrics = [lp_metric] + random_metrics

        results: Dict[str, Any] = {
            "target_sparsity": float(sparsity),
            "supernode_fraction": float(supernode_fraction),
            "num_trials": int(num_trials),
            "num_supernodes": int(num_supernodes),
            "seed": base_seed,
            "lp_metric": lp_metric,
            "random_metrics": random_metrics,
            "pruning_metrics": pruning_metrics,
            "lp_indices": {},
            "random_indices": [],
            "overlap": {},
        }

        def _store_metric(layer_idx: int, metric_name: str, scores: torch.Tensor) -> None:
            keys = layer_to_proj_keys.get(layer_idx) or []
            for k in keys:
                if k not in self.importance_scores:
                    self.importance_scores[k] = {}
                self.importance_scores[k][metric_name] = scores

        # LP-supernode protection metric
        for layer_name, lp_tensor in lp_tensors.items():
            m = re.search(r"layers\.(\d+)\.mlp", layer_name)
            if not m:
                continue
            layer_idx = int(m.group(1))

            _, top_idx = torch.topk(lp_tensor, num_supernodes)
            results["lp_indices"][layer_name] = top_idx.tolist()

            protection = lp_tensor.clone()
            if protection.numel() > 0:
                protection[top_idx] = protection.max() * 2
            _store_metric(layer_idx, lp_metric, protection)

        # Random trials (deterministic from seed, trial, layer_idx)
        for trial in range(int(num_trials)):
            metric_name = random_metrics[trial]
            trial_entry: Dict[str, Any] = {"trial": int(trial), "seed": int(base_seed + 100000 * (trial + 1)), "indices": {}}

            for layer_name, lp_tensor in lp_tensors.items():
                m = re.search(r"layers\.(\d+)\.mlp", layer_name)
                if not m:
                    continue
                layer_idx = int(m.group(1))

                g = torch.Generator(device="cpu")
                g.manual_seed(base_seed + 100000 * (trial + 1) + layer_idx)
                random_idx = torch.randperm(intermediate_dim, generator=g)[:num_supernodes]
                trial_entry["indices"][layer_name] = random_idx.tolist()

                protection = lp_tensor.clone()
                if protection.numel() > 0:
                    protection[random_idx] = protection.max() * 2
                _store_metric(layer_idx, metric_name, protection)

            results["random_indices"].append(trial_entry)

        # Overlap stats (LP vs random per layer)
        try:
            overlap_by_layer: Dict[str, Any] = {}
            for layer_name in layer_names:
                lp_idx = results["lp_indices"].get(layer_name)
                if not lp_idx:
                    continue
                lp_set = set(lp_idx)
                overlaps: List[float] = []
                for tr in results["random_indices"]:
                    ridx = (tr.get("indices") or {}).get(layer_name)
                    if not ridx:
                        continue
                    overlaps.append(len(lp_set & set(ridx)) / float(num_supernodes))
                if overlaps:
                    overlap_by_layer[layer_name] = {
                        "mean_overlap_frac": float(np.mean(overlaps)),
                        "std_overlap_frac": float(np.std(overlaps)),
                        "expected_overlap_frac": float(num_supernodes / float(intermediate_dim)),
                    }
            results["overlap"] = overlap_by_layer
        except Exception as e:
            logger.warning(f"Overlap analysis failed: {e}")

        logger.info("Random supernode ablation metrics injected into importance_scores.")
        logger.info(f"Injected pruning metrics: {pruning_metrics}")

        return results

    def compute_mean_replacement_control(
        self,
        scar_scores: Dict[str, Dict[str, Any]],
        *,
        supernode_fraction: float = 0.01,
        num_eval_texts: int = 64,
        max_length: int = 512,
        num_random_trials: int = 5,
    ) -> Dict[str, Any]:
        """
        Mean-replacement control experiment.

        Tests whether supernodes are functionally important by replacing their activations
        with per-channel mean values and measuring the loss impact.

        Interventions:
        1. Baseline: no replacement
        2. LP supernodes replaced with mean
        3. Activation supernodes replaced with mean
        4. Random channels (same size) replaced with mean (control)

        Args:
            scar_scores: Pre-computed SCAR scores with 'scar_loss_proxy' and 'scar_activation_power'
            supernode_fraction: Fraction of channels to treat as supernodes (default 1%)
            num_eval_texts: Number of evaluation texts
            max_length: Maximum sequence length
            num_random_trials: Number of random replacement trials

        Returns:
            Dict with baseline loss, LP supernode loss, activation supernode loss,
            random replacement mean/std, per-layer statistics
        """
        logger.info("=" * 60)
        logger.info("Mean-Replacement Control Experiment")
        logger.info("=" * 60)

        device = torch.device(self.config.device)
        model_dtype = getattr(torch, self.config.model_config.get("torch_dtype", "float32"))

        # Get evaluation texts
        eval_texts: List[str] = []
        if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
            eval_texts = list(self.dataset.texts)[:num_eval_texts]
        else:
            try:
                from datasets import load_dataset

                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                eval_texts = [t for t in ds["text"] if t.strip()][:num_eval_texts]
            except Exception as e:
                logger.error(f"Failed to load evaluation texts: {e}")
                return {}

        if not eval_texts:
            logger.error("No evaluation texts available for mean-replacement control")
            return {}

        # Extract LP and activation supernodes per layer
        layer_supernodes: Dict[str, Dict[str, np.ndarray]] = {}
        for layer_name, layer_data in scar_scores.items():
            if "mlp.down_proj" not in layer_name:
                continue

            lp = layer_data.get("scar_loss_proxy")
            act = layer_data.get("scar_activation_power")

            if lp is None or act is None:
                continue

            if torch.is_tensor(lp):
                lp = lp.cpu().numpy()
            if torch.is_tensor(act):
                act = act.cpu().numpy()

            n = len(lp)
            k = max(1, int(supernode_fraction * n))

            lp_indices = np.argsort(lp)[-k:]
            act_indices = np.argsort(act)[-k:]

            layer_supernodes[layer_name] = {
                "lp": lp_indices,
                "act": act_indices,
                "n_channels": n,
                "k": k,
            }

        if not layer_supernodes:
            logger.error("No supernode data found in scar_scores")
            return {}

        logger.info(f"Found {len(layer_supernodes)} layers with supernodes")
        sample_layer = next(iter(layer_supernodes.values()))
        logger.info(f"Channels per layer: {sample_layer['n_channels']}, supernodes: {sample_layer['k']}")

        # Get underlying HF model
        hf_model: nn.Module = self.model
        if hasattr(hf_model, "model"):
            hf_model = getattr(hf_model, "model")

        def compute_loss_with_replacement(
            replacement_indices: Optional[Dict[str, np.ndarray]],
            mean_values: Dict[str, torch.Tensor],
        ) -> float:
            """Compute mean loss when replacing specified channels with their means."""
            hooks = []

            if replacement_indices is not None:
                for layer_name, module in hf_model.named_modules():
                    if layer_name not in replacement_indices:
                        continue
                    indices = replacement_indices[layer_name]
                    means = mean_values.get(layer_name)
                    if means is None:
                        continue

                    def make_hook(idx: np.ndarray, mv: torch.Tensor):
                        def hook(mod, inp, out):
                            if not inp or inp[0] is None:
                                return
                            u = inp[0]
                            # Replace selected channels with mean
                            u_modified = u.clone()
                            u_modified[..., idx] = mv[idx].to(u.device, u.dtype)
                            return (u_modified,) + inp[1:] if len(inp) > 1 else (u_modified,)

                        return hook

                    h = module.register_forward_pre_hook(make_hook(indices, means))
                    hooks.append(h)

            total_loss = 0.0
            total_tokens = 0

            try:
                self.model.eval()
                with torch.no_grad():
                    for text in eval_texts:
                        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                        input_ids = enc["input_ids"].to(device)
                        if input_ids.size(1) < 2:
                            continue

                        labels = input_ids.clone()
                        labels[:, 0] = -100
                        n_valid = int((labels != -100).sum().item())
                        if n_valid <= 0:
                            continue

                        with torch.autocast(device_type=str(device).split(":")[0], dtype=model_dtype):
                            outputs = self.model(input_ids, labels=labels)
                            loss = outputs.loss

                        total_loss += loss.item() * n_valid
                        total_tokens += n_valid
            finally:
                for h in hooks:
                    h.remove()

            return total_loss / total_tokens if total_tokens > 0 else float("inf")

        # Step 1: Compute per-channel means from calibration
        logger.info("Computing per-channel activation means...")
        mean_values: Dict[str, torch.Tensor] = {}
        count_values: Dict[str, int] = {}
        hooks = []

        for layer_name, module in hf_model.named_modules():
            if layer_name not in layer_supernodes:
                continue
            n_ch = layer_supernodes[layer_name]["n_channels"]
            mean_values[layer_name] = torch.zeros(n_ch, device="cpu", dtype=torch.float32)
            count_values[layer_name] = 0

            def make_mean_hook(name: str, n: int):
                def hook(mod, inp, out):
                    if not inp or inp[0] is None:
                        return
                    u = inp[0].detach().float()
                    if u.ndim > 2:
                        u = u.reshape(-1, u.shape[-1])
                    mean_values[name] += u.sum(dim=0).cpu()
                    count_values[name] += u.shape[0]

                return hook

            h = module.register_forward_hook(make_mean_hook(layer_name, n_ch))
            hooks.append(h)

        # Forward pass to accumulate means
        self.model.eval()
        with torch.no_grad():
            for text in eval_texts[:32]:  # Use subset for mean computation
                enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = enc["input_ids"].to(device)
                if input_ids.size(1) < 2:
                    continue
                with torch.autocast(device_type=str(device).split(":")[0], dtype=model_dtype):
                    _ = self.model(input_ids)

        for h in hooks:
            h.remove()

        # Finalize means
        for name in mean_values:
            if count_values[name] > 0:
                mean_values[name] /= count_values[name]

        # Step 2: Baseline (no replacement)
        logger.info("Computing baseline loss...")
        baseline_loss = compute_loss_with_replacement(None, mean_values)
        logger.info(f"Baseline loss: {baseline_loss:.4f}")

        # Step 3: LP supernode replacement
        logger.info("Computing LP supernode replacement loss...")
        lp_indices = {name: data["lp"] for name, data in layer_supernodes.items()}
        lp_loss = compute_loss_with_replacement(lp_indices, mean_values)
        logger.info(f"LP supernode replacement loss: {lp_loss:.4f}")

        # Step 4: Activation supernode replacement
        logger.info("Computing activation supernode replacement loss...")
        act_indices = {name: data["act"] for name, data in layer_supernodes.items()}
        act_loss = compute_loss_with_replacement(act_indices, mean_values)
        logger.info(f"Activation supernode replacement loss: {act_loss:.4f}")

        # Step 5: Random replacement trials
        logger.info(f"Computing {num_random_trials} random replacement trials...")
        random_losses = []
        base_seed = int(getattr(self.config, "seed", 42) or 42)

        for trial in range(num_random_trials):
            random_indices = {}
            for name, data in layer_supernodes.items():
                g = torch.Generator()
                g.manual_seed(base_seed + trial * 1000 + hash(name) % 10000)
                n_ch = data["n_channels"]
                k = data["k"]
                random_indices[name] = torch.randperm(n_ch, generator=g)[:k].numpy()

            trial_loss = compute_loss_with_replacement(random_indices, mean_values)
            random_losses.append(trial_loss)
            logger.info(f"  Trial {trial + 1}: {trial_loss:.4f}")

        random_mean = float(np.mean(random_losses))
        random_std = float(np.std(random_losses))
        logger.info(f"Random replacement mean: {random_mean:.4f} +/- {random_std:.4f}")

        results = {
            "supernode_fraction": float(supernode_fraction),
            "num_eval_texts": int(num_eval_texts),
            "max_length": int(max_length),
            "num_random_trials": int(num_random_trials),
            "baseline_loss": float(baseline_loss),
            "lp_supernode_loss": float(lp_loss),
            "activation_supernode_loss": float(act_loss),
            "random_replacement": {
                "mean": float(random_mean),
                "std": float(random_std),
                "trials": [float(x) for x in random_losses],
            },
            "lp_vs_baseline_increase": float(lp_loss - baseline_loss),
            "act_vs_baseline_increase": float(act_loss - baseline_loss),
            "random_vs_baseline_increase": float(random_mean - baseline_loss),
        }

        logger.info("=" * 60)
        logger.info("Mean-Replacement Control Results Summary")
        logger.info("=" * 60)
        logger.info(f"Baseline: {baseline_loss:.4f}")
        logger.info(f"LP supernodes: {lp_loss:.4f} (+{lp_loss - baseline_loss:.4f})")
        logger.info(f"Act supernodes: {act_loss:.4f} (+{act_loss - baseline_loss:.4f})")
        logger.info(f"Random: {random_mean:.4f} +/- {random_std:.4f} (+{random_mean - baseline_loss:.4f})")

        return results

    def compute_lp_activation_analysis(
        self,
        scar_scores: Dict[str, Dict[str, Any]],
        *,
        supernode_fraction: float = 0.01,
        percentiles: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute LP vs Activation analysis: correlation by percentile and supernode overlap.

        This analyzes the relationship between LP (loss proxy) and activation power:
        1. Spearman correlation between log(LP) and log(activation) per layer
        2. Correlation restricted to top X% by activation power
        3. Jaccard overlap between LP-defined and activation-defined supernodes

        Args:
            scar_scores: Pre-computed SCAR scores with 'scar_loss_proxy' and 'scar_activation_power'
            supernode_fraction: Fraction for supernode definition (default 1%)
            percentiles: Percentiles to compute correlation for (default [100, 99, 95, 90, 75, 50, 25, 10, 5, 1])

        Returns:
            Dict with per-layer and summary statistics
        """
        from scipy.stats import spearmanr

        logger.info("=" * 60)
        logger.info("LP vs Activation Analysis")
        logger.info("=" * 60)

        if percentiles is None:
            percentiles = [100, 99, 95, 90, 75, 50, 25, 10, 5, 1]

        results: Dict[str, Any] = {
            "supernode_fraction": float(supernode_fraction),
            "percentiles": percentiles,
            "per_layer": {},
            "summary": {},
        }

        all_correlations: Dict[int, List[float]] = {p: [] for p in percentiles}
        all_jaccard: List[float] = []

        for layer_name, layer_data in scar_scores.items():
            if "mlp.down_proj" not in layer_name:
                continue

            lp = layer_data.get("scar_loss_proxy")
            act = layer_data.get("scar_activation_power")

            if lp is None or act is None:
                continue

            if torch.is_tensor(lp):
                lp = lp.cpu().numpy().astype(np.float64)
            else:
                lp = np.array(lp, dtype=np.float64)

            if torch.is_tensor(act):
                act = act.cpu().numpy().astype(np.float64)
            else:
                act = np.array(act, dtype=np.float64)

            n = len(lp)
            if n < 10:
                continue

            # Log transform (handle zeros)
            eps = 1e-12
            log_lp = np.log(np.maximum(lp, eps))
            log_act = np.log(np.maximum(act, eps))

            # Correlation by percentile (top X% by activation)
            layer_corr: Dict[int, float] = {}
            for pct in percentiles:
                if pct >= 100:
                    subset_mask = np.ones(n, dtype=bool)
                else:
                    threshold = np.percentile(act, 100 - pct)
                    subset_mask = act >= threshold

                if subset_mask.sum() < 3:
                    layer_corr[pct] = float("nan")
                    continue

                try:
                    rho, _ = spearmanr(log_lp[subset_mask], log_act[subset_mask])
                    layer_corr[pct] = float(rho) if rho is not None else float("nan")
                except Exception:
                    layer_corr[pct] = float("nan")

                if not np.isnan(layer_corr[pct]):
                    all_correlations[pct].append(layer_corr[pct])

            # Supernode overlap (Jaccard)
            k = max(1, int(supernode_fraction * n))
            lp_supernodes = set(np.argsort(lp)[-k:].tolist())
            act_supernodes = set(np.argsort(act)[-k:].tolist())

            intersection = len(lp_supernodes & act_supernodes)
            union = len(lp_supernodes | act_supernodes)
            jaccard = intersection / union if union > 0 else 0.0
            all_jaccard.append(jaccard)

            results["per_layer"][layer_name] = {
                "n_channels": int(n),
                "correlation_by_percentile": layer_corr,
                "jaccard_supernodes": float(jaccard),
            }

        # Summary statistics
        summary_corr: Dict[str, Dict[str, float]] = {}
        for pct in percentiles:
            vals = all_correlations[pct]
            if vals:
                summary_corr[str(pct)] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
            else:
                summary_corr[str(pct)] = {"mean": float("nan"), "std": float("nan")}

        results["summary"] = {
            "correlation_by_percentile": summary_corr,
            "jaccard_supernodes": {
                "mean": float(np.mean(all_jaccard)) if all_jaccard else float("nan"),
                "std": float(np.std(all_jaccard)) if all_jaccard else float("nan"),
            },
        }

        # Log summary
        logger.info(f"Analyzed {len(results['per_layer'])} layers")
        if "100" in summary_corr:
            logger.info(f"Full correlation (log LP vs log Act): {summary_corr['100']['mean']:.3f} +/- {summary_corr['100']['std']:.3f}")
        if "90" in summary_corr:
            logger.info(f"Top 90% by activation: {summary_corr['90']['mean']:.3f} +/- {summary_corr['90']['std']:.3f}")
        if all_jaccard:
            logger.info(f"Supernode Jaccard overlap: {np.mean(all_jaccard)*100:.1f}% +/- {np.std(all_jaccard)*100:.1f}%")

        return results

    def compute_supernode_hit_rate_sweep(
        self,
        scar_scores: Dict[str, Dict[str, Any]],
        *,
        supernode_fraction: float = 0.01,
        sparsity: float = 0.5,
        hit_rates: Optional[List[float]] = None,
        num_trials: int = 3,
        seed: Optional[int] = None,
        prefix: str = "supernode_hit_rate_sweep",
    ) -> Dict[str, Any]:
        """
        Dose-response control: random FFN channel pruning masks conditioned on a target
        *supernode hit-rate* (fraction of LP supernodes pruned).

        This constructs synthetic per-channel pruning scores (stored in `self.importance_scores`)
        such that structured pruning at `sparsity` prunes:
          - ~hit_rate * (num_supernodes) channels from the supernode set, and
          - the remaining pruned channels from non-supernodes, per layer.

        The standard pruning loop can then evaluate perplexity/benchmarks for each synthetic
        metric name, producing a clean causal curve (hit-rate -> degradation) without confounds
        from comparing only named baselines.

        Notes:
          - This is *per-layer* conditioning (same target hit-rate in each FFN layer).
          - The synthetic metric names are prefixed so `_should_protect_supernodes_for_metric`
            will not apply core protection, even if enabled globally.
        """
        import re

        if hit_rates is None:
            hit_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
        # Sanitize hit rates
        hit_rates = [float(max(0.0, min(1.0, hr))) for hr in hit_rates]

        layer_names = [ln for ln in scar_scores.keys() if "mlp.down_proj" in ln]
        if not layer_names:
            return {}

        # Map layer index -> projection module keys in importance_scores (so pruning can see the metric everywhere).
        layer_to_proj_keys: Dict[int, List[str]] = {}
        for k in self.importance_scores.keys():
            m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", k)
            if m:
                layer_to_proj_keys.setdefault(int(m.group(1)), []).append(k)

        # Parse LP tensors per layer (used for supernode identification)
        lp_tensors: Dict[str, torch.Tensor] = {}
        for layer_name in layer_names:
            layer_metrics = scar_scores.get(layer_name) or {}
            lp = layer_metrics.get("scar_loss_proxy")
            if isinstance(lp, dict) and "scores" in lp:
                lp_tensor = torch.tensor(lp["scores"], dtype=torch.float32)
            elif torch.is_tensor(lp):
                lp_tensor = lp.float().detach().cpu()
            else:
                continue
            if lp_tensor.numel() > 0:
                lp_tensors[layer_name] = lp_tensor

        if not lp_tensors:
            logger.warning("Hit-rate sweep: no LP scores found; cannot identify supernodes")
            return {}

        base_seed = int(seed if seed is not None else getattr(self.config, "seed", 0) or 0)

        results: Dict[str, Any] = {
            "target_sparsity": float(sparsity),
            "supernode_fraction": float(supernode_fraction),
            "hit_rates": hit_rates,
            "num_trials": int(num_trials),
            "seed": int(base_seed),
            "prefix": str(prefix),
            "pruning_metrics": [],
            "targets": [],  # one entry per generated metric
        }

        def _store_metric(layer_idx: int, metric_name: str, scores: torch.Tensor) -> None:
            keys = layer_to_proj_keys.get(layer_idx) or []
            for k in keys:
                if k not in self.importance_scores:
                    self.importance_scores[k] = {}
                self.importance_scores[k][metric_name] = scores

        # Precompute supernode indices per layer (top by LP)
        super_idx_by_layer: Dict[str, torch.Tensor] = {}
        non_super_idx_by_layer: Dict[str, torch.Tensor] = {}
        num_super_by_layer: Dict[str, int] = {}
        for layer_name, lp_tensor in lp_tensors.items():
            m = lp_tensor.numel()
            num_super = max(1, int(round(supernode_fraction * m)))
            _, top_idx = torch.topk(lp_tensor, num_super)
            super_idx_by_layer[layer_name] = top_idx
            num_super_by_layer[layer_name] = int(num_super)

            super_mask = torch.zeros(m, dtype=torch.bool)
            super_mask[top_idx] = True
            non_idx = (~super_mask).nonzero(as_tuple=True)[0]
            non_super_idx_by_layer[layer_name] = non_idx

        # Generate metrics
        for hr in hit_rates:
            hr_tag = int(round(100.0 * hr))
            for trial in range(int(num_trials)):
                metric_name = f"{prefix}_hr{hr_tag:02d}_t{trial}"
                results["pruning_metrics"].append(metric_name)
                results["targets"].append({"metric": metric_name, "hit_rate": float(hr), "trial": int(trial)})

                for layer_name, lp_tensor in lp_tensors.items():
                    m = re.search(r"layers\.(\d+)\.mlp", layer_name)
                    if not m:
                        continue
                    layer_idx = int(m.group(1))

                    dim = int(lp_tensor.numel())
                    num_to_prune = int(round(float(sparsity) * float(dim)))
                    num_to_prune = max(0, min(num_to_prune, dim - 1))  # keep at least 1 channel

                    super_idx = super_idx_by_layer[layer_name]
                    non_idx = non_super_idx_by_layer[layer_name]
                    num_super = int(num_super_by_layer[layer_name])

                    n_super_prune = int(round(float(hr) * float(num_super)))
                    n_super_prune = max(0, min(n_super_prune, num_super, num_to_prune))
                    n_non_prune = max(0, num_to_prune - n_super_prune)
                    if n_non_prune > int(non_idx.numel()):
                        # Should not happen for small supernode_fraction, but keep robust.
                        n_non_prune = int(non_idx.numel())
                        n_super_prune = max(0, num_to_prune - n_non_prune)

                    # Deterministic RNG per (hit-rate, trial, layer_idx)
                    g = torch.Generator(device="cpu")
                    g.manual_seed(base_seed + 1000000 * (hr_tag + 1) + 10000 * (trial + 1) + layer_idx)

                    # Sample pruned indices without replacement
                    pruned_super = super_idx[torch.randperm(num_super, generator=g)[:n_super_prune]] if n_super_prune > 0 else None
                    pruned_non = non_idx[torch.randperm(int(non_idx.numel()), generator=g)[:n_non_prune]] if n_non_prune > 0 else None
                    if pruned_super is None and pruned_non is None:
                        prune_idx = torch.empty(0, dtype=torch.long)
                    elif pruned_super is None:
                        prune_idx = pruned_non
                    elif pruned_non is None:
                        prune_idx = pruned_super
                    else:
                        prune_idx = torch.cat([pruned_super, pruned_non], dim=0)

                    # Construct synthetic pruning scores: pruned channels get low scores.
                    scores = torch.ones(dim, dtype=torch.float32)
                    if prune_idx.numel() > 0:
                        scores[prune_idx] = 0.0
                    # Tiny noise to avoid tie-edge cases in topk selection
                    scores = scores + (1e-6 * torch.rand(dim, generator=g, dtype=torch.float32))

                    _store_metric(layer_idx, metric_name, scores)

        return results

    def compute_conditional_halo_ablation(
        self,
        *,
        scar_scores: Dict[str, Dict[str, Any]],
        supernode_fraction: float = 0.01,
        halo_fraction: float = 0.10,
        layer_stride: int = 4,
        layer_indices: Optional[List[int]] = None,
        num_texts: int = 16,
        max_length: int = 256,
        match_bins: int = 10,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Conditional causal test for the mechanistic story:

        For each selected layer ℓ (FFN `down_proj`):
        - Define supernodes M_ℓ as top-ρ by LP (loss proxy).
        - Define write-halo H_ℓ as top-η (by Conn) among non-supernodes.
        - Compare ΔNLL when ablating:
            (i)  a random K-sized subset of H_ℓ (supernodes intact)
            (ii) a matched K-sized subset of non-halo channels (supernodes intact)
            (iii) supernodes M_ℓ
            (iv) supernodes M_ℓ plus the halo subset

        This is designed to show that halo membership predicts *conditional redundancy*:
        halo ablation is small given supernodes intact, while supernode ablation is large.
        """
        import re
        from contextlib import contextmanager

        logger.info("=" * 60)
        logger.info("Conditional Halo Ablation (causal redundancy probe)")
        logger.info("=" * 60)
        logger.info(f"  supernode_fraction (rho): {float(supernode_fraction) * 100:.2f}%")
        logger.info(f"  halo_fraction (eta): {float(halo_fraction) * 100:.2f}%")
        logger.info(f"  num_texts: {int(num_texts)}, max_length: {int(max_length)}")

        # ------------------------------------------------------------------
        # Build a small held-out text set (prefer WikiText-2 test; fallback to calibration texts)
        # ------------------------------------------------------------------
        eval_texts: List[str] = []
        llm_cfg = getattr(self.config, "llm", {}) or {}
        try:
            from datasets import load_dataset

            subset = str(llm_cfg.get("wikitext_subset", "wikitext-2-raw-v1"))
            ds = load_dataset("wikitext", subset, split="test")
            texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
            rng = np.random.default_rng(int(seed))
            rng.shuffle(texts)
            eval_texts = texts[: max(1, int(num_texts))]
            logger.info(f"  Using WikiText test lines: subset={subset}, n={len(eval_texts)}")
        except Exception:
            if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
                eval_texts = [t for t in list(self.dataset.texts) if isinstance(t, str) and t.strip()][: max(1, int(num_texts))]
                logger.info(f"  Using calibration texts fallback: n={len(eval_texts)}")

        if not eval_texts:
            logger.warning("No evaluation texts available; skipping conditional halo ablation.")
            return {"error": "no_evaluation_texts"}

        tokenized: List[Dict[str, torch.Tensor]] = []
        for t in eval_texts:
            toks = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=int(max_length),
                padding=False,
            )
            tokenized.append(toks)

        device = torch.device(getattr(self.config, "device", "cuda"))

        @torch.no_grad()
        def _eval_loss() -> float:
            total_loss = 0.0
            total_tokens = 0
            self.model.eval()
            for toks in tokenized:
                batch = {k: v.to(device) for k, v in toks.items()}
                input_ids = batch.get("input_ids")
                if input_ids is None:
                    continue
                try:
                    out = self.model(**batch, labels=input_ids)
                    loss = float(out.loss.item())
                except Exception:
                    continue
                n = int(input_ids.numel())
                total_loss += loss * max(1, n)
                total_tokens += max(1, n)
            return total_loss / max(1, total_tokens)

        module_dict = dict(self.model.named_modules())

        def _resolve(name: str):
            if name in module_dict:
                return module_dict[name]
            if name.startswith("model.") and name[len("model.") :] in module_dict:
                return module_dict[name[len("model.") :]]
            alt = "model.model." + name
            if alt in module_dict:
                return module_dict[alt]
            for k, v in module_dict.items():
                if k.endswith(name):
                    return v
            return None

        def _lookup_layer_scores(layer_name: str) -> Dict[str, Any]:
            # importance_scores keys can vary (model.layers vs model.model.layers, etc.)
            for key in (
                layer_name,
                layer_name.replace("model.layers.", "model.model.layers."),
                layer_name.replace("model.model.layers.", "model.layers."),
                layer_name.replace("model.", ""),
            ):
                rec = self.importance_scores.get(key)
                if isinstance(rec, dict) and rec:
                    return rec
            return {}

        @contextmanager
        def _ablate_downproj_inputs(layer_name: str, indices: np.ndarray):
            mod = _resolve(layer_name)
            if mod is None:
                raise ValueError(f"could not resolve module: {layer_name}")
            if indices is None or len(indices) == 0:
                yield
                return
            try:
                idx_device = mod.weight.device  # type: ignore[attr-defined]
            except Exception:
                idx_device = next(mod.parameters()).device
            idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long, device=idx_device)

            def pre_hook(_m: nn.Module, inputs: Tuple[torch.Tensor, ...]):
                if not inputs or inputs[0] is None:
                    return inputs
                u = inputs[0]
                y = u.clone()
                y.index_fill_(-1, idx, 0.0)
                return (y,) + tuple(inputs[1:])

            h = mod.register_forward_pre_hook(pre_hook)
            try:
                yield
            finally:
                h.remove()

        baseline_loss = _eval_loss()
        baseline_ppl = float(np.exp(baseline_loss))

        # Select layers to analyze
        down_layers = sorted([k for k in scar_scores.keys() if "mlp.down_proj" in k])
        layer_recs: List[Dict[str, Any]] = []

        # Parse available layer indices
        parsed: List[Tuple[int, str]] = []
        for ln in down_layers:
            m = re.search(r"layers\.(\d+)", ln)
            if m:
                parsed.append((int(m.group(1)), ln))
        parsed.sort(key=lambda x: x[0])

        if layer_indices is not None:
            wanted = set(int(x) for x in layer_indices)
            parsed = [p for p in parsed if p[0] in wanted]
        else:
            stride = max(1, int(layer_stride))
            parsed = [p for p in parsed if (p[0] % stride) == 0]

        np.random.default_rng(int(seed))

        for li, ln in parsed:
            lp = scar_scores.get(ln, {}).get("scar_loss_proxy")
            if lp is None:
                continue
            lp_cpu = lp.detach().float().cpu().numpy().reshape(-1)
            m_int = int(lp_cpu.size)
            if m_int <= 0:
                continue

            # Connectivity score from SCAR-Conn computation
            layer_scores = _lookup_layer_scores(ln)
            conn = layer_scores.get("connectivity_score")
            if conn is None or not torch.is_tensor(conn) or int(conn.numel()) != m_int:
                continue
            conn_np = conn.detach().float().cpu().numpy().reshape(-1)

            num_super = max(1, int(round(float(supernode_fraction) * float(m_int))))
            super_idx = np.argsort(lp_cpu)[::-1][:num_super].astype(np.int64)
            super_mask = np.zeros(m_int, dtype=bool)
            super_mask[super_idx] = True

            eligible = np.where(~super_mask)[0]
            if eligible.size == 0:
                continue

            # Halo: top-eta by Conn among non-supernodes
            num_halo = max(1, int(round(float(halo_fraction) * float(m_int))))
            num_halo = int(min(num_halo, eligible.size))
            elig_conn = conn_np[eligible]
            halo_order = eligible[np.argsort(elig_conn)[::-1]]
            halo_idx = halo_order[:num_halo].astype(np.int64)

            halo_set = set(int(x) for x in halo_idx.tolist())
            non_halo_pool = np.asarray([i for i in eligible.tolist() if int(i) not in halo_set], dtype=np.int64)
            if non_halo_pool.size == 0:
                continue

            # Ablate K channels (default: K = |M|)
            K = int(min(num_super, halo_idx.size, non_halo_pool.size))
            if K <= 0:
                continue

            rng_layer = np.random.default_rng(int(seed) + 1000 * int(li))
            halo_subset = rng_layer.choice(halo_idx, size=K, replace=False).astype(np.int64)

            # LP-quantile matched non-halo subset
            pool_lp = lp_cpu[non_halo_pool]
            # Robust binning
            bins = max(2, int(match_bins))
            edges = np.quantile(pool_lp, np.linspace(0.0, 1.0, bins + 1))
            edges[0] -= 1e-12
            edges[-1] += 1e-12
            pool_bin = np.clip(np.digitize(pool_lp, edges[1:-1], right=True), 0, bins - 1)
            halo_bin = np.clip(np.digitize(lp_cpu[halo_subset], edges[1:-1], right=True), 0, bins - 1)

            matched: List[int] = []
            used: set = set()
            for b in range(bins):
                need = int(np.sum(halo_bin == b))
                if need <= 0:
                    continue
                cand = non_halo_pool[pool_bin == b]
                cand = np.asarray([int(x) for x in cand.tolist() if int(x) not in used], dtype=np.int64)
                if cand.size >= need:
                    pick = rng_layer.choice(cand, size=need, replace=False)
                else:
                    pick = cand
                    rem = need - int(cand.size)
                    rest = np.asarray(
                        [int(x) for x in non_halo_pool.tolist() if int(x) not in used and int(x) not in set(pick.tolist())], dtype=np.int64
                    )
                    if rest.size > 0:
                        pick2 = rng_layer.choice(rest, size=min(rem, int(rest.size)), replace=False)
                        pick = np.concatenate([pick, pick2])
                for x in pick.tolist():
                    used.add(int(x))
                matched.extend([int(x) for x in pick.tolist()])

            # If matching underfilled (rare), top up randomly.
            if len(matched) < K:
                rest = np.asarray([int(x) for x in non_halo_pool.tolist() if int(x) not in set(matched)], dtype=np.int64)
                if rest.size > 0:
                    fill = rng_layer.choice(rest, size=min(K - len(matched), int(rest.size)), replace=False)
                    matched.extend([int(x) for x in fill.tolist()])
            matched = matched[:K]
            matched_np = np.asarray(matched, dtype=np.int64)

            # Evaluate interventions
            with _ablate_downproj_inputs(ln, halo_subset):
                loss_halo = _eval_loss()
            with _ablate_downproj_inputs(ln, matched_np):
                loss_matched = _eval_loss()
            with _ablate_downproj_inputs(ln, super_idx):
                loss_super = _eval_loss()
            both = np.unique(np.concatenate([super_idx, halo_subset]).astype(np.int64))
            with _ablate_downproj_inputs(ln, both):
                loss_both = _eval_loss()

            layer_recs.append(
                {
                    "layer": ln,
                    "layer_idx": int(li),
                    "K": int(K),
                    "sets": {
                        "num_supernodes": int(num_super),
                        "num_halo": int(num_halo),
                        "halo_subset": halo_subset.tolist(),
                        "matched_non_halo_subset": matched_np.tolist(),
                    },
                    "losses": {
                        "baseline": float(baseline_loss),
                        "halo_subset": float(loss_halo),
                        "matched_non_halo_subset": float(loss_matched),
                        "supernodes": float(loss_super),
                        "supernodes_plus_halo": float(loss_both),
                    },
                    "delta_nll": {
                        "halo_subset": float(loss_halo - baseline_loss),
                        "matched_non_halo_subset": float(loss_matched - baseline_loss),
                        "supernodes": float(loss_super - baseline_loss),
                        "supernodes_plus_halo": float(loss_both - baseline_loss),
                    },
                }
            )

        layer_recs.sort(key=lambda r: int(r.get("layer_idx", 0)))
        logger.info(f"Conditional halo ablation complete for {len(layer_recs)} layers.")

        # Aggregate summary stats (small; used for summary tables/claims).
        gaps: List[float] = []
        dn_halo: List[float] = []
        dn_matched: List[float] = []
        dn_super: List[float] = []
        dn_both: List[float] = []
        for rec in layer_recs:
            dn = rec.get("delta_nll") or {}
            try:
                h = float(dn.get("halo_subset"))
                m = float(dn.get("matched_non_halo_subset"))
                s = float(dn.get("supernodes"))
                b = float(dn.get("supernodes_plus_halo"))
            except Exception:
                continue
            if not (np.isfinite(h) and np.isfinite(m) and np.isfinite(s) and np.isfinite(b)):
                continue
            dn_halo.append(h)
            dn_matched.append(m)
            dn_super.append(s)
            dn_both.append(b)
            gaps.append(m - h)

        def _summ(vals: List[float]) -> Dict[str, float]:
            a = np.asarray(vals, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
            return {
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
            }

        return {
            "baseline_loss": float(baseline_loss),
            "baseline_ppl": float(baseline_ppl),
            "supernode_fraction": float(supernode_fraction),
            "halo_fraction": float(halo_fraction),
            "num_texts": int(len(eval_texts)),
            "max_length": int(max_length),
            "match_bins": int(match_bins),
            "summary": {
                "delta_nll_halo_subset": _summ(dn_halo),
                "delta_nll_matched_non_halo_subset": _summ(dn_matched),
                "delta_nll_supernodes": _summ(dn_super),
                "delta_nll_supernodes_plus_halo": _summ(dn_both),
                "gap_matched_minus_halo": _summ(gaps),
                "frac_layers_where_halo_less_than_matched": float(np.mean(np.asarray(gaps) > 0.0)) if gaps else float("nan"),
            },
            "layers": layer_recs,
        }

    def compute_lp_ablation_validation(
        self,
        *,
        scar_scores: Dict[str, Dict[str, Any]],
        layer_stride: int = 8,
        layer_indices: Optional[List[int]] = None,
        num_texts: int = 8,
        max_length: int = 256,
        num_channels: int = 128,
        quantile_bins: int = 8,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Validate the LP proxy against *true* loss change from single-channel ablation.

        For each selected layer ℓ (FFN `down_proj`), sample channels spanning the LP range
        (via LP-quantile bins), ablate each channel i by setting u_i=0, and measure ΔNLL.

        This produces a direct empirical calibration of LP as a measurement instrument.
        """
        import math
        import re
        from contextlib import contextmanager

        logger.info("=" * 60)
        logger.info("LP Ablation Validation (LP vs true Δloss)")
        logger.info("=" * 60)
        logger.info(f"  num_texts: {int(num_texts)}, max_length: {int(max_length)}")
        logger.info(f"  num_channels/layer: {int(num_channels)}, quantile_bins: {int(quantile_bins)}")

        # ------------------------------------------------------------------
        # Build a small held-out text set (prefer WikiText-2 test; fallback to calibration texts)
        # ------------------------------------------------------------------
        eval_texts: List[str] = []
        llm_cfg = getattr(self.config, "llm", {}) or {}
        try:
            from datasets import load_dataset

            subset = str(llm_cfg.get("wikitext_subset", "wikitext-2-raw-v1"))
            ds = load_dataset("wikitext", subset, split="test")
            texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
            rng = np.random.default_rng(int(seed))
            rng.shuffle(texts)
            eval_texts = texts[: max(1, int(num_texts))]
            logger.info(f"  Using WikiText test lines: subset={subset}, n={len(eval_texts)}")
        except Exception:
            if hasattr(self, "dataset") and hasattr(self.dataset, "texts"):
                eval_texts = [t for t in list(self.dataset.texts) if isinstance(t, str) and t.strip()][: max(1, int(num_texts))]
                logger.info(f"  Using calibration texts fallback: n={len(eval_texts)}")

        if not eval_texts:
            logger.warning("No evaluation texts available; skipping LP ablation validation.")
            return {"error": "no_evaluation_texts"}

        tokenized: List[Dict[str, torch.Tensor]] = []
        for t in eval_texts:
            toks = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=int(max_length),
                padding=False,
            )
            tokenized.append(toks)

        device = torch.device(getattr(self.config, "device", "cuda"))

        @torch.no_grad()
        def _eval_loss() -> float:
            total_loss = 0.0
            total_tokens = 0
            self.model.eval()
            for toks in tokenized:
                batch = {k: v.to(device) for k, v in toks.items()}
                input_ids = batch.get("input_ids")
                if input_ids is None:
                    continue
                try:
                    out = self.model(**batch, labels=input_ids)
                    loss = float(out.loss.item())
                except Exception:
                    continue
                n = int(input_ids.numel())
                total_loss += loss * max(1, n)
                total_tokens += max(1, n)
            return total_loss / max(1, total_tokens)

        module_dict = dict(self.model.named_modules())

        def _resolve(name: str):
            if name in module_dict:
                return module_dict[name]
            if name.startswith("model.") and name[len("model.") :] in module_dict:
                return module_dict[name[len("model.") :]]
            alt = "model.model." + name
            if alt in module_dict:
                return module_dict[alt]
            for k, v in module_dict.items():
                if k.endswith(name):
                    return v
            return None

        @contextmanager
        def _ablate_downproj_inputs(layer_name: str, indices: np.ndarray):
            mod = _resolve(layer_name)
            if mod is None:
                raise ValueError(f"could not resolve module: {layer_name}")
            if indices is None or len(indices) == 0:
                yield
                return
            try:
                idx_device = mod.weight.device  # type: ignore[attr-defined]
            except Exception:
                idx_device = next(mod.parameters()).device
            idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long, device=idx_device)

            def pre_hook(_m: nn.Module, inputs: Tuple[torch.Tensor, ...]):
                if not inputs or inputs[0] is None:
                    return inputs
                u = inputs[0]
                y = u.clone()
                y.index_fill_(-1, idx, 0.0)
                return (y,) + tuple(inputs[1:])

            h = mod.register_forward_pre_hook(pre_hook)
            try:
                yield
            finally:
                h.remove()

        baseline_loss = _eval_loss()
        baseline_ppl = float(np.exp(baseline_loss))

        # Select layers to analyze
        down_layers = sorted([k for k in scar_scores.keys() if "mlp.down_proj" in k])
        parsed: List[Tuple[int, str]] = []
        for ln in down_layers:
            m = re.search(r"layers\.(\d+)", ln)
            if m:
                parsed.append((int(m.group(1)), ln))
        parsed.sort(key=lambda x: x[0])

        if layer_indices is not None:
            wanted = set(int(x) for x in layer_indices)
            parsed = [p for p in parsed if p[0] in wanted]
        else:
            stride = max(1, int(layer_stride))
            parsed = [p for p in parsed if (p[0] % stride) == 0]

        def _spearman(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            if a.size < 3 or b.size != a.size:
                return float("nan")
            ra = a.argsort().argsort().astype(np.float64)
            rb = b.argsort().argsort().astype(np.float64)
            ra -= ra.mean()
            rb -= rb.mean()
            denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
            rho = float((ra @ rb) / denom)
            return rho if np.isfinite(rho) else float("nan")

        rng0 = np.random.default_rng(int(seed))
        layer_recs: List[Dict[str, Any]] = []

        for li, ln in parsed:
            lp = scar_scores.get(ln, {}).get("scar_loss_proxy")
            if lp is None or not torch.is_tensor(lp):
                continue
            lp_cpu = lp.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
            lp_cpu = np.where(np.isfinite(lp_cpu) & (lp_cpu > 0.0), lp_cpu, 0.0)
            m_int = int(lp_cpu.size)
            if m_int <= 0:
                continue

            n = int(min(max(1, int(num_channels)), m_int))
            bins = max(2, int(quantile_bins))

            nz = np.where(lp_cpu > 0.0)[0]
            if nz.size < 3:
                continue
            log_lp = np.log10(lp_cpu[nz])
            edges = np.quantile(log_lp, np.linspace(0.0, 1.0, bins + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            bin_id = np.clip(np.digitize(log_lp, edges[1:-1], right=True), 0, bins - 1)

            per_bin = max(1, int(math.ceil(n / float(bins))))
            chosen: List[int] = []
            used: set = set()
            for b in range(bins):
                cand = nz[bin_id == b]
                if cand.size == 0:
                    continue
                take = min(per_bin, int(cand.size))
                pick = rng0.choice(cand, size=take, replace=False)
                for x in pick.tolist():
                    used.add(int(x))
                chosen.extend([int(x) for x in pick.tolist()])
            if len(chosen) < n:
                rest = np.asarray([int(x) for x in nz.tolist() if int(x) not in used], dtype=np.int64)
                if rest.size > 0:
                    pick = rng0.choice(rest, size=min(n - len(chosen), int(rest.size)), replace=False)
                    chosen.extend([int(x) for x in pick.tolist()])
            chosen_np = np.asarray(chosen[:n], dtype=np.int64)

            logger.info(f"  Layer {li}: evaluating {int(chosen_np.size)} single-channel ablations...")
            deltas: List[float] = []
            for k, idx in enumerate(chosen_np.tolist()):
                with _ablate_downproj_inputs(ln, np.asarray([int(idx)], dtype=np.int64)):
                    loss_i = _eval_loss()
                deltas.append(float(loss_i - baseline_loss))
                if (k + 1) % 25 == 0:
                    logger.info(f"    progress: {k+1}/{int(chosen_np.size)}")

            lp_sel = lp_cpu[chosen_np].astype(np.float64)
            dn_sel = np.asarray(deltas, dtype=np.float64)

            mask = np.isfinite(lp_sel) & np.isfinite(dn_sel) & (lp_sel > 0.0) & (dn_sel > 0.0)
            rho_loglog = _spearman(np.log10(lp_sel[mask]), np.log10(dn_sel[mask])) if int(np.sum(mask)) >= 3 else float("nan")
            rho_raw = _spearman(lp_sel[mask], dn_sel[mask]) if int(np.sum(mask)) >= 3 else float("nan")

            layer_recs.append(
                {
                    "layer": ln,
                    "layer_idx": int(li),
                    "num_channels": int(chosen_np.size),
                    "indices": chosen_np.tolist(),
                    "lp": lp_sel.tolist(),
                    "delta_nll": dn_sel.tolist(),
                    "spearman_loglog": float(rho_loglog) if np.isfinite(rho_loglog) else float("nan"),
                    "spearman_raw": float(rho_raw) if np.isfinite(rho_raw) else float("nan"),
                }
            )

        layer_recs.sort(key=lambda r: int(r.get("layer_idx", 0)))
        rhos = [float(r.get("spearman_loglog")) for r in layer_recs if isinstance(r, dict)]
        rhos = [r for r in rhos if np.isfinite(r)]

        summary = {
            "spearman_loglog": {
                "mean": float(np.mean(np.asarray(rhos))) if rhos else float("nan"),
                "median": float(np.median(np.asarray(rhos))) if rhos else float("nan"),
                "min": float(np.min(np.asarray(rhos))) if rhos else float("nan"),
                "max": float(np.max(np.asarray(rhos))) if rhos else float("nan"),
            }
        }

        return {
            "baseline_loss": float(baseline_loss),
            "baseline_ppl": float(baseline_ppl),
            "num_texts": int(len(eval_texts)),
            "max_length": int(max_length),
            "num_channels": int(num_channels),
            "quantile_bins": int(quantile_bins),
            "summary": summary,
            "layers": layer_recs,
        }
