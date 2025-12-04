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
from alignment.pruning.strategies.llm_baselines import WandaPruning, SparseGPTPruning
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

    # =========================================================================
    # NVIDIA MINITRON-COMPATIBLE FEW-SHOT SETTINGS
    # Reference: https://arxiv.org/abs/2408.11796
    # =========================================================================
    # These are the official few-shot settings used by NVIDIA Minitron:
    NVIDIA_FEWSHOT_SETTINGS = {
        "accuracy_mmlu": 5,           # MMLU: 5-shot
        "accuracy_hellaswag": 10,     # HellaSwag: 10-shot
        "accuracy_arc_challenge": 25, # ARC-Challenge: 25-shot
        "accuracy_arc_easy": 25,      # ARC-Easy: 25-shot (same as Challenge)
        "accuracy_winogrande": 5,     # WinoGrande: 5-shot
        "accuracy_gsm8k": 5,          # GSM8k: 5-shot with chain-of-thought
        "accuracy_truthfulqa": 0,     # TruthfulQA: 0-shot
        "accuracy_mbpp": 0,           # MBPP: 0-shot
        "accuracy_humaneval": 0,      # HumanEval: 0-shot
        "accuracy_piqa": 0,           # PIQA: 0-shot
        "accuracy_boolq": 0,          # BoolQ: 0-shot
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
        
        results = {}
        
        for metric in metrics:
            num_fewshot = fewshot_settings.get(metric, 0)
            try:
                if metric == "perplexity":
                    results["perplexity"] = self.evaluate_perplexity(
                        dataset=getattr(self.config, "evaluation_dataset", "wikitext"),
                        num_samples=num_samples
                    )
                elif metric == "loss":
                    # Cross-entropy loss = ln(perplexity)
                    ppl = self.evaluate_perplexity(
                        dataset=getattr(self.config, "evaluation_dataset", "wikitext"),
                        num_samples=num_samples
                    )
                    results["loss"] = np.log(ppl)
                elif metric == "bits_per_byte":
                    # Bits per byte = log2(perplexity) / avg_chars_per_token
                    ppl = self.evaluate_perplexity(
                        dataset=getattr(self.config, "evaluation_dataset", "wikitext"),
                        num_samples=num_samples
                    )
                    # Approximate: assume ~4 characters per token on average
                    results["bits_per_byte"] = np.log2(ppl) / 4.0
                elif metric == "accuracy_hellaswag":
                    results["accuracy_hellaswag"] = self._evaluate_hellaswag(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_arc_easy":
                    results["accuracy_arc_easy"] = self._evaluate_arc_easy(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_arc_challenge":
                    results["accuracy_arc_challenge"] = self._evaluate_arc_challenge(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_piqa":
                    results["accuracy_piqa"] = self._evaluate_piqa(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_boolq":
                    results["accuracy_boolq"] = self._evaluate_boolq(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_winogrande":
                    results["accuracy_winogrande"] = self._evaluate_winogrande(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_truthfulqa":
                    results["accuracy_truthfulqa"] = self._evaluate_truthfulqa(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_mmlu":
                    results["accuracy_mmlu"] = self._evaluate_mmlu(
                        num_samples=num_samples, num_fewshot=num_fewshot
                    )
                elif metric == "accuracy_gsm8k":
                    results["accuracy_gsm8k"] = self._evaluate_gsm8k(
                        num_samples=num_samples, 
                        num_fewshot=num_fewshot,
                        use_chain_of_thought=use_chain_of_thought
                    )
                elif metric == "accuracy_mbpp":
                    results["accuracy_mbpp"] = self._evaluate_mbpp(num_samples=num_samples)
                elif metric == "accuracy_humaneval":
                    results["accuracy_humaneval"] = self._evaluate_humaneval(num_samples=num_samples)
                elif metric == "normalized_perplexity":
                    # Normalized to 0-100 scale (100 = best = PPL of 1)
                    ppl = self.evaluate_perplexity(
                        dataset=getattr(self.config, "evaluation_dataset", "wikitext"),
                        num_samples=num_samples
                    )
                    # Use exponential decay: score = 100 * exp(-0.01 * (ppl - 1))
                    results["normalized_perplexity"] = 100 * np.exp(-0.01 * (ppl - 1))
                else:
                    logger.warning(f"Unknown evaluation metric: {metric}")
            except Exception as e:
                logger.error(f"Failed to evaluate metric '{metric}': {e}")
                results[metric] = None
        
        return results

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
        logger.info(f"Evaluating {shot_str} accuracy on MMLU ({num_samples} samples)...")
        
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
        device = torch.device(self.config.device)
        choice_labels = ["A", "B", "C", "D"]
        
        # Calculate samples per subject
        samples_per_subject = max(1, num_samples // len(subjects))
        
        with torch.no_grad():
            for subject in subjects:
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
                    for i, ex in enumerate(dev_dataset):
                        if i >= num_fewshot:
                            break
                        q = ex["question"]
                        choices = ex["choices"]
                        answer_idx = ex["answer"]
                        # Format: The following are multiple choice questions...
                        choices_str = "\n".join([f"{choice_labels[j]}) {c}" for j, c in enumerate(choices)])
                        correct_label = choice_labels[answer_idx]
                        fewshot_prompt += f"Question: {q}\n{choices_str}\nAnswer: {correct_label}\n\n"
                
                subject_correct = 0
                subject_total = 0
                
                for i, example in enumerate(dataset):
                    if subject_total >= samples_per_subject:
                        break
                    
                    try:
                        question = example["question"]
                        choices = example["choices"]
                        answer_idx = example["answer"]  # 0-indexed
                        
                        # Score each choice
                        scores = []
                        for j, choice in enumerate(choices):
                            # Format: Question: ... Answer: A) choice
                            if num_fewshot > 0:
                                choices_str = "\n".join([f"{choice_labels[k]}) {c}" for k, c in enumerate(choices)])
                                text = f"{fewshot_prompt}Question: {question}\n{choices_str}\nAnswer: {choice_labels[j]}"
                            else:
                                text = f"Question: {question}\nAnswer: {choice_labels[j]}) {choice}"
                            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            
                            outputs = self.model(**inputs)
                            logits = outputs.logits
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = inputs["input_ids"][..., 1:].contiguous()
                            
                            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            scores.append(-loss.item())
                        
                        predicted = np.argmax(scores)
                        if predicted == answer_idx:
                            correct += 1
                            subject_correct += 1
                        total += 1
                        subject_total += 1
                        
                    except Exception as e:
                        logger.warning(f"Error on MMLU {subject} sample {i}: {e}")
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
        device = torch.device(self.config.device)
        
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
                    
                    # Score each ending
                    scores = []
                    for ending in endings:
                        if num_fewshot > 0:
                            text = f"{fewshot_prompt}Context: {ctx}\nEnding: {ending}"
                        else:
                            text = f"{ctx} {ending}"
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scores.append(-loss.item())
                    
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
        device = torch.device(self.config.device)
        
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
                    
                    # Score each choice
                    scores = []
                    for choice_text in choice_texts:
                        if num_fewshot > 0:
                            text = f"{fewshot_prompt}Question: {question}\nAnswer: {choice_text}"
                        else:
                            text = f"Question: {question}\nAnswer: {choice_text}"
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scores.append(-loss.item())
                    
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
        device = torch.device(self.config.device)
        
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break
                
                try:
                    goal = example["goal"]
                    sol1 = example["sol1"]
                    sol2 = example["sol2"]
                    label = example["label"]  # 0 or 1
                    
                    # Score each solution
                    scores = []
                    for sol in [sol1, sol2]:
                        if num_fewshot > 0:
                            text = f"{fewshot_prompt}Goal: {goal}\nSolution: {sol}"
                        else:
                            text = f"Goal: {goal}\nSolution: {sol}"
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scores.append(-loss.item())
                    
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
        device = torch.device(self.config.device)
        
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break
                
                try:
                    question = example["question"]
                    passage = example["passage"]
                    answer = example["answer"]  # True or False
                    
                    # Score "Yes" vs "No" completions
                    scores = []
                    for response in ["Yes", "No"]:
                        if num_fewshot > 0:
                            text = f"{fewshot_prompt}Passage: {passage}\nQuestion: {question}\nAnswer: {response}"
                        else:
                            text = f"Passage: {passage}\nQuestion: {question}\nAnswer: {response}"
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scores.append(-loss.item())
                    
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
        device = torch.device(self.config.device)
        
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if total >= num_samples:
                    break
                
                try:
                    sentence = example["sentence"]
                    option1 = example["option1"]
                    option2 = example["option2"]
                    answer = int(example["answer"]) - 1  # Convert 1/2 to 0/1
                    
                    # Replace _ with each option and score
                    scores = []
                    for option in [option1, option2]:
                        completed = sentence.replace("_", option)
                        if num_fewshot > 0:
                            text = f"{fewshot_prompt}Sentence: {completed}"
                        else:
                            text = completed
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scores.append(-loss.item())  # Higher = better
                    
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
        device = torch.device(self.config.device)
        
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
                    
                    # Score each choice
                    scores = []
                    for choice_text in choice_texts:
                        if num_fewshot > 0:
                            text = f"{fewshot_prompt}Question: {question}\nAnswer: {choice_text}"
                        else:
                            text = f"Question: {question}\nAnswer: {choice_text}"
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs["input_ids"][..., 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scores.append(-loss.item())
                    
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
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
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
            match = re.search(r'####\s*([0-9,\-\.]+)', text)
            if match:
                return match.group(1).replace(',', '')
            # Try to find last number in text
            numbers = re.findall(r'[\-]?[0-9,]+\.?[0-9]*', text)
            if numbers:
                return numbers[-1].replace(',', '')
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
                    
                    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
                    
                    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    
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
                    
                    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    
                    # Combine prompt and generated code
                    # Stop at first function definition that's not the target
                    lines = generated.split('\n')
                    code_lines = []
                    for line in lines:
                        if line.strip().startswith('def ') and entry_point not in line:
                            break
                        code_lines.append(line)
                    
                    full_code = prompt + '\n'.join(code_lines)
                    
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
            strategies: List of baseline strategies to compute. Default: ["wanda", "sparsegpt"]
            num_calibration_samples: Number of samples for calibration
            
        Returns:
            Dict mapping layer names to {strategy_name: scores_tensor}
        """
        if strategies is None:
            # Check which baseline strategies are configured
            pruning_strategies = getattr(self.config, "pruning_strategies", [])
            strategies = [s for s in pruning_strategies if s in ["wanda", "sparsegpt"]]
        
        if not strategies:
            logger.info("No baseline pruning strategies (wanda/sparsegpt) configured, skipping.")
            return {}
        
        logger.info(f"Computing baseline pruning scores for: {strategies}")
        
        # Get calibration dataloader
        try:
            from alignment.dataops.datasets.text_datasets import WikiTextDataset
            from torch.utils.data import DataLoader
            
            tokenizer = getattr(self, 'tokenizer', None)
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
        
        # Track which layers to compute scores for (MLP layers from tracked_layers)
        target_layers = []
        for layer_name in self.importance_scores.keys():
            # Check if it's an MLP layer
            if any(mlp_pattern in layer_name for mlp_pattern in ["up_proj", "gate_proj", "down_proj", "fc", "mlp"]):
                target_layers.append(layer_name)
        
        if not target_layers:
            logger.warning("No target layers found for baseline pruning scores")
            return {}
        
        logger.info(f"Computing baseline scores for {len(target_layers)} layers")
        
        # Compute Wanda scores
        if "wanda" in strategies:
            logger.info("Calibrating Wanda pruning strategy...")
            try:
                wanda = WandaPruning(num_calibration_samples=num_calibration_samples)
                wanda.calibrate(model, calib_dataloader, device=str(device))
                
                # Compute scores for each tracked layer
                for layer_name in target_layers:
                    # Find the module
                    module = None
                    for name, mod in model.named_modules():
                        if name == layer_name or layer_name.endswith(name) or name.endswith(layer_name.split('.')[-1]):
                            if isinstance(mod, nn.Linear):
                                module = mod
                                break
                    
                    if module is not None:
                        try:
                            # Get structured scores (per output neuron)
                            scores = wanda.get_structured_scores(module, layer_name=layer_name, dim=0)
                            
                            # Store in importance_scores
                            if layer_name not in self.importance_scores:
                                self.importance_scores[layer_name] = {}
                            self.importance_scores[layer_name]["wanda"] = scores
                            
                            if layer_name not in results:
                                results[layer_name] = {}
                            results[layer_name]["wanda"] = scores
                            
                            logger.debug(f"Wanda scores for {layer_name}: shape {scores.shape}, mean {scores.mean():.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to compute Wanda scores for {layer_name}: {e}")
                
                logger.info(f"Wanda: computed scores for {len([k for k in results if 'wanda' in results.get(k, {})])} layers")
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
                
                # Compute scores for each tracked layer
                for layer_name in target_layers:
                    # Find the module
                    module = None
                    for name, mod in model.named_modules():
                        if name == layer_name or layer_name.endswith(name) or name.endswith(layer_name.split('.')[-1]):
                            if isinstance(mod, nn.Linear):
                                module = mod
                                break
                    
                    if module is not None:
                        try:
                            # Get structured scores (per output neuron)
                            scores = sparsegpt.get_structured_scores(module, layer_name=layer_name, dim=0)
                            
                            # Store in importance_scores
                            if layer_name not in self.importance_scores:
                                self.importance_scores[layer_name] = {}
                            self.importance_scores[layer_name]["sparsegpt"] = scores
                            
                            if layer_name not in results:
                                results[layer_name] = {}
                            results[layer_name]["sparsegpt"] = scores
                            
                            logger.debug(f"SparseGPT scores for {layer_name}: shape {scores.shape}, mean {scores.mean():.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to compute SparseGPT scores for {layer_name}: {e}")
                
                logger.info(f"SparseGPT: computed scores for {len([k for k in results if 'sparsegpt' in results.get(k, {})])} layers")
            except Exception as e:
                logger.error(f"SparseGPT calibration failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
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
                import fnmatch
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
                    threshold_value=follower_vals[num_followers-1].item(),
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
        from scipy import stats
        import matplotlib.pyplot as plt
        
        if metrics is None:
            metrics = [
                'scar_activation_power', 
                'scar_loss_proxy', 
                'scar_taylor',
                'rayleigh_quotient', 
                'gaussian_mi_analytic', 
                'activation_l2_norm'
            ]
        
        logger.info(f"Analyzing supernode robustness:")
        logger.info(f"  - Supernode fraction: top {supernode_fraction*100:.1f}%")
        logger.info(f"  - Bootstrap samples: {num_bootstrap_samples}")
        logger.info(f"  - Metrics to compare: {metrics}")
        
        if plots_dir is None:
            plots_dir = Path(getattr(self.config, "plots_dir", "./plots"))
        plots_dir = Path(plots_dir)
        robustness_dir = plots_dir / "supernode_robustness"
        robustness_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model and device
        device = next(self.model.parameters()).device
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
                elif metric_name in self.importance_scores.get(layer_name, {}):
                    # Pre-computed importance scores
                    metric_scores_layer[metric_name] = self.importance_scores[layer_name][metric_name].float().cpu()
                else:
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
            
            if len(metric_scores_layer) < 2:
                logger.warning(f"    Only {len(metric_scores_layer)} metrics available, need at least 2 for comparison")
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
            layer_suffix = layer_name.replace('.', '_')
            
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
            summary = {
                "num_layers_analyzed": len(results),
                "avg_jaccard_across_metrics": np.mean([
                    np.mean(np.array(r["jaccard_matrix"])[np.triu_indices(len(r["metrics_analyzed"]), k=1)])
                    for r in results.values() if "jaccard_matrix" in r
                ]),
                "avg_spearman_across_metrics": np.mean([
                    np.mean(np.array(r["spearman_matrix"])[np.triu_indices(len(r["metrics_analyzed"]), k=1)])
                    for r in results.values() if "spearman_matrix" in r
                ]),
                "avg_highly_stable_fraction": np.mean([
                    r["num_highly_stable"] / r["num_supernodes"]
                    for r in results.values() if "num_highly_stable" in r
                ]),
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
                    batch_texts = texts[i:i+batch_size]
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
        """Compute Rayleigh Quotient for a specific layer."""
        device = next(self.model.parameters()).device
        
        # Get weight matrix
        hf_model = self.model
        if hasattr(hf_model, "model"):
            hf_model = hf_model.model
        
        weight = None
        for name, module in hf_model.named_modules():
            if name == layer_name or name.endswith(layer_name):
                if hasattr(module, "weight"):
                    weight = module.weight.data.float()  # [hidden_dim, intermediate_dim]
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
                    batch_texts = texts[i:i+batch_size]
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
                
                # Compute covariance
                mean = all_acts.mean(dim=0, keepdim=True)
                centered = all_acts - mean
                cov = (centered.T @ centered) / (all_acts.shape[0] - 1)
                
                # weight: [hidden_dim, intermediate_dim]
                # Each column is a neuron's outgoing weights
                # RQ = w^T Σ w / w^T w
                weight_t = weight.T.to(device)  # [intermediate_dim, hidden_dim]
                
                # w^T Σ w for each neuron
                w_cov = weight_t @ cov  # [intermediate_dim, intermediate_dim]
                w_cov_w = torch.sum(w_cov * weight_t, dim=1)  # [intermediate_dim]
                
                # w^T w for each neuron
                w_w = torch.sum(weight_t ** 2, dim=1)  # [intermediate_dim]
                
                rq = w_cov_w / (w_w + 1e-10)
                return rq.cpu()
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
                color='green',
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
                color='purple',
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
            all_labels = ['High'] * len(high_influence_values) + ['Low'] * len(low_influence_values)
            
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

    def compute_directed_redundancy(
        self,
        scar_scores: Dict[str, Dict[str, torch.Tensor]],
        supernode_fraction: float = 0.01,
        halo_fraction: float = 0.10,
        num_samples: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute DIRECTED REDUNDANCY: redundancy among neurons in the "halo" connected to supernodes.
        
        This is the key novel contribution for SCAR pruning:
        1. Identify supernodes (top neurons by activation power/loss proxy)
        2. Find "halo" neurons: those with large weight connections TO supernodes
        3. Compute pairwise redundancy WITHIN the halo
        4. Neurons in the halo that are highly redundant with each other are safe to prune
        5. Neurons with low redundancy (unique information) should be protected
        
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
            rho_sq = corr_matrix ** 2
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
            DirectedRedundancy(i→j) = |weight_ij| × R²(activation_i → activation_j)
        
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
        logger.info(f"Computing directed redundancy from supernodes to downstream neurons...")
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
            # Compute Directed Redundancy: R²(supernode_i → output_j) × |weight_ij|
            # =====================================================================
            
            # For efficiency, compute correlations in batch
            # Center the activations
            supernode_centered = supernode_acts - supernode_acts.mean(dim=0, keepdim=True)
            output_centered = all_output - all_output.mean(dim=0, keepdim=True)
            
            # Compute variance of each supernode
            supernode_var = (supernode_centered ** 2).sum(dim=0) / (N - 1)  # [num_supernodes]
            output_var = (output_centered ** 2).sum(dim=0) / (N - 1)  # [hidden_dim]
            
            # Compute covariance between supernodes and outputs
            # cov_matrix[i, j] = cov(supernode_i, output_j)
            cov_matrix = (supernode_centered.T @ output_centered) / (N - 1)  # [num_supernodes, hidden_dim]
            
            # Compute R² (coefficient of determination)
            # R²(i→j) = cov(i,j)² / (var(i) × var(j))
            denom = (supernode_var.unsqueeze(1) * output_var.unsqueeze(0) + 1e-8)
            r_squared = (cov_matrix ** 2) / denom  # [num_supernodes, hidden_dim]
            
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
            
            # Create "supernode protection score" - higher = more important, harder to prune
            # All neurons get a base score based on their activation (L2 norm)
            # Then supernodes get a large boost
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
            protection_scores = base_protection.clone()
            protection_scores[supernode_indices] = max_base * 10 + supernode_total_influence
            layer_scores["supernode_protection_score"] = protection_scores
            
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
        Compute supernode-connectivity based pruning score.
        
        Algorithm:
        1. For each layer, identify supernodes (top neurons by activation power)
        2. Compute weight connections from each neuron to supernodes in next layer
        3. Partition neurons into:
           - High-connectivity: Strong weights to supernodes → compute redundancy, prune redundant
           - Low-connectivity: Weak weights to supernodes → low importance, can prune
        4. Create composite pruning score:
           - Low connectivity neurons get low score (safe to prune)
           - High connectivity neurons: score = base_importance - redundancy_penalty
        
        This captures the insight that neurons weakly connected to important supernodes
        are likely less important, while among strongly connected neurons, the redundant
        ones are safer to prune.
        
        Args:
            scar_scores: SCAR scores dictionary with supernode metrics
            supernode_fraction: Fraction of neurons considered supernodes
            high_connectivity_fraction: Fraction of neurons considered "high connectivity"
            redundancy_weight: Weight for redundancy penalty in high-connectivity group
            num_samples: Calibration samples for redundancy computation
            plots_dir: Directory to save analysis plots
            
        Returns:
            Dictionary with pruning scores and analysis per layer
        """
        logger.info("Computing supernode-connectivity based pruning score...")
        logger.info(f"  Supernode fraction: {supernode_fraction*100:.1f}%")
        logger.info(f"  High-connectivity fraction: {high_connectivity_fraction*100:.1f}%")
        
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
            logger.warning("No calibration texts available")
            return {}
        
        # Setup plots directory
        if plots_dir:
            plots_dir = Path(plots_dir)
            scatter_dir = plots_dir / "scatter"
            scatter_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all layer names with SCAR scores
        layer_names = [ln for ln in scar_scores.keys() if "mlp.down_proj" in ln]
        
        for idx, layer_name in enumerate(layer_names):
            layer_metrics = scar_scores[layer_name]
            
            # Get supernode identification metric
            # Note: Use explicit None checks to avoid tensor boolean ambiguity
            supernode_metric = layer_metrics.get("scar_activation_power")
            if supernode_metric is None:
                supernode_metric = layer_metrics.get("scar_loss_proxy")
            if supernode_metric is None:
                continue
            
            supernode_metric = supernode_metric.float().cpu()
            num_neurons = supernode_metric.numel()  # intermediate_dim
            
            # Identify supernodes in THIS layer
            num_supernodes = max(1, int(supernode_fraction * num_neurons))
            _, sorted_indices = torch.sort(supernode_metric, descending=True)
            supernode_indices = sorted_indices[:num_supernodes]
            supernode_mask = torch.zeros(num_neurons, dtype=torch.bool)
            supernode_mask[supernode_indices] = True
            
            # Get weights from this layer's neurons to NEXT layer
            # down_proj: [hidden_dim, intermediate_dim] - current layer output
            # next layer's up_proj/gate_proj: [intermediate_dim', hidden_dim] - receives from hidden
            
            # For simplicity, we use the current layer's down_proj weights
            # to estimate connectivity importance (how much each neuron contributes to output)
            down_proj_weight = None
            for name, module in hf_model.named_modules():
                if name == layer_name or (name.endswith("mlp.down_proj") and name in layer_name):
                    if hasattr(module, "weight"):
                        down_proj_weight = module.weight.detach().float().cpu()
                        break
            
            if down_proj_weight is None:
                continue
            
            hidden_dim, intermediate_dim = down_proj_weight.shape
            
            # Compute connectivity score: how much each neuron (column) contributes
            # to outputs that go to supernodes in the NEXT layer
            # For within-layer analysis: use total weight magnitude per neuron
            neuron_output_magnitude = down_proj_weight.abs().sum(dim=0)  # [intermediate_dim]
            
            # If we have next layer info, use supernode indices from next layer
            # For now, use supernode influence from THIS layer
            # Supernodes have high activation, so neurons with large weights TO those outputs matter
            supernode_influence = down_proj_weight[:, supernode_indices].abs().sum(dim=1)  # [hidden_dim]
            
            # For each neuron in intermediate_dim, compute its "supernode connectivity"
            # = sum of |weights| to hidden dimensions that have high supernode_influence
            num_high_influence = max(1, int(high_connectivity_fraction * hidden_dim))
            _, high_influence_hidden = torch.topk(supernode_influence, num_high_influence)
            
            # Connectivity score: how much does each intermediate neuron contribute to 
            # hidden dimensions that strongly connect to supernodes
            connectivity_score = down_proj_weight[high_influence_hidden, :].abs().sum(dim=0)  # [intermediate_dim]
            
            # Partition into high and low connectivity
            num_high_conn = max(1, int(high_connectivity_fraction * intermediate_dim))
            _, conn_sorted = torch.sort(connectivity_score, descending=True)
            high_conn_indices = conn_sorted[:num_high_conn]
            low_conn_indices = conn_sorted[num_high_conn:]
            
            high_conn_mask = torch.zeros(intermediate_dim, dtype=torch.bool)
            high_conn_mask[high_conn_indices] = True
            
            logger.info(f"  {layer_name}:")
            logger.info(f"    {num_supernodes} supernodes, {len(high_conn_indices)} high-connectivity neurons")
            
            # Compute redundancy among high-connectivity neurons
            # Capture activations for redundancy computation
            activations: List[torch.Tensor] = []
            
            def capture_hook(module, inputs, outputs):
                if inputs and inputs[0] is not None:
                    inp = inputs[0].detach().float()
                    if inp.ndim == 3:
                        inp = inp.reshape(-1, inp.shape[-1])
                    activations.append(inp.cpu())
            
            hook_handle = None
            for name, module in hf_model.named_modules():
                if name == layer_name or (name.endswith("mlp.down_proj") and name in layer_name):
                    hook_handle = module.register_forward_hook(capture_hook)
                    break
            
            if hook_handle:
                self.model.eval()
                with torch.no_grad():
                    for text in calibration_texts:
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                        try:
                            self.model(**inputs)
                        except Exception:
                            pass
                hook_handle.remove()
            
            # Compute redundancy for high-connectivity neurons
            redundancy_scores = torch.zeros(intermediate_dim)
            
            if activations:
                all_acts = torch.cat(activations, dim=0)  # [N, intermediate_dim]
                high_acts = all_acts[:, high_conn_indices]  # [N, num_high_conn]
                
                # Compute pairwise correlation among high-connectivity neurons
                high_centered = high_acts - high_acts.mean(dim=0, keepdim=True)
                high_cov = (high_centered.T @ high_centered) / (high_acts.shape[0] - 1)
                high_std = torch.sqrt(torch.diag(high_cov) + 1e-8)
                high_corr = high_cov / (high_std.unsqueeze(0) * high_std.unsqueeze(1) + 1e-8)
                high_corr = torch.clamp(high_corr, -1, 1)
                high_corr.fill_diagonal_(0)
                
                # Per-neuron redundancy = mean |correlation| with others
                per_neuron_redundancy = high_corr.abs().mean(dim=1)
                redundancy_scores[high_conn_indices] = per_neuron_redundancy
                
                logger.info(f"    High-connectivity redundancy: mean={per_neuron_redundancy.mean():.4f}, max={per_neuron_redundancy.max():.4f}")
            
            # Build composite pruning score
            # Low connectivity → low score (safe to prune)
            # High connectivity with high redundancy → lower score (redundant, can prune)
            # High connectivity with low redundancy → high score (important, protect)
            
            # Normalize connectivity to [0, 1]
            conn_normalized = connectivity_score.clone()
            if conn_normalized.max() > conn_normalized.min():
                conn_normalized = (conn_normalized - conn_normalized.min()) / (conn_normalized.max() - conn_normalized.min())
            
            # Base importance from activation power
            base_importance = supernode_metric.clone()
            if base_importance.max() > base_importance.min():
                base_importance = (base_importance - base_importance.min()) / (base_importance.max() - base_importance.min())
            
            # Composite score:
            # - Start with base importance (activation power)
            # - Multiply by connectivity (low connectivity = low score)
            # - Subtract redundancy penalty for high-connectivity neurons
            pruning_score = base_importance * (0.5 + 0.5 * conn_normalized)  # [0, 1]
            pruning_score[high_conn_mask] = pruning_score[high_conn_mask] - redundancy_weight * redundancy_scores[high_conn_mask]
            
            # Supernodes always get high protection
            pruning_score[supernode_mask] = pruning_score.max() + 1.0
            
            # Store in importance_scores
            layer_scores = self.importance_scores.get(layer_name, {})
            layer_scores["supernode_connectivity_score"] = pruning_score
            layer_scores["connectivity_score"] = connectivity_score
            layer_scores["redundancy_in_high_conn"] = redundancy_scores
            layer_scores["high_connectivity_mask"] = high_conn_mask
            self.importance_scores[layer_name] = layer_scores
            
            # Generate scatter plots
            if plots_dir and activations:
                import matplotlib.pyplot as plt
                viz = UnifiedVisualizer()
                
                try:
                    # Scatter 1: Connectivity vs Base Importance
                    fig, ax = plt.subplots(figsize=(10, 8))
                    colors = ['#e74c3c' if supernode_mask[i] else '#3498db' if high_conn_mask[i] else '#95a5a6' 
                              for i in range(intermediate_dim)]
                    ax.scatter(conn_normalized.numpy(), base_importance.numpy(), c=colors, alpha=0.5, s=10)
                    ax.set_xlabel("Connectivity Score (normalized)")
                    ax.set_ylabel("Base Importance (activation power)")
                    ax.set_title(f"Connectivity vs Importance\n{layer_name}")
                    ax.legend(handles=[
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Supernode'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', label='High-Connectivity'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', label='Low-Connectivity'),
                    ])
                    ax.grid(True, alpha=0.3)
                    fig.savefig(scatter_dir / f"connectivity_vs_importance_{layer_name.replace('.', '_')}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    
                    # Scatter 2: Redundancy vs Connectivity (for high-connectivity only)
                    if high_conn_mask.sum() > 0:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        hc_conn = conn_normalized[high_conn_mask].numpy()
                        hc_red = redundancy_scores[high_conn_mask].numpy()
                        ax.scatter(hc_conn, hc_red, alpha=0.5, s=20, c='#3498db')
                        ax.set_xlabel("Connectivity Score")
                        ax.set_ylabel("Redundancy Score")
                        ax.set_title(f"Redundancy vs Connectivity (High-Conn Neurons)\n{layer_name}")
                        ax.grid(True, alpha=0.3)
                        fig.savefig(scatter_dir / f"redundancy_vs_connectivity_{layer_name.replace('.', '_')}.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                    
                    # Scatter 3: Final Pruning Score vs Base Importance
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter(base_importance.numpy(), pruning_score.numpy(), c=colors, alpha=0.5, s=10)
                    ax.set_xlabel("Base Importance (activation power)")
                    ax.set_ylabel("Final Pruning Score")
                    ax.set_title(f"Pruning Score vs Base Importance\n{layer_name}")
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.savefig(scatter_dir / f"pruning_score_vs_importance_{layer_name.replace('.', '_')}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    
                except Exception as e:
                    logger.warning(f"    Failed to generate scatter plots: {e}")
            
            results[layer_name] = {
                "num_supernodes": num_supernodes,
                "num_high_connectivity": len(high_conn_indices),
                "num_low_connectivity": len(low_conn_indices),
                "mean_redundancy_high_conn": float(redundancy_scores[high_conn_mask].mean().item()) if high_conn_mask.sum() > 0 else 0,
                "pruning_score_range": [float(pruning_score.min().item()), float(pruning_score.max().item())],
            }
        
        logger.info(f"Computed supernode-connectivity pruning score for {len(results)} layers")
        return results
    
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
        
        # For SCAR metrics, baseline methods (wanda, sparsegpt), and other pre-computed scores,
        # use PrecomputedScorePruning since they're not in the metric registry
        precomputed_metrics = [
            # SCAR metrics (computed by SCAR analysis)
            "scar_loss_proxy", "scar_activation_power", "scar_taylor", "scar_curvature",
            # Supernode/connectivity metrics
            "directed_redundancy", "supernode_protection_score", "supernode_connectivity_score",
            # LLM baseline methods (computed by compute_baseline_pruning_scores)
            "wanda", "sparsegpt",
        ]
        if metric in precomputed_metrics:
            from alignment.pruning.base import PrecomputedScorePruning
            pruner = PrecomputedScorePruning(config=config)
        else:
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

            supernode_cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
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
            # Order matters: try most specific first
            possible_paths = [
                f"model.model.layers.{layer_idx}.mlp",  # HFCausalLM wrapper with nested model
                f"model.layers.{layer_idx}.mlp",         # Direct HF model (LlamaForCausalLM)
                f"layers.{layer_idx}.mlp",               # Inner LlamaModel
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
                if not all(hasattr(mlp_module, attr) for attr in ['gate_proj', 'up_proj', 'down_proj']):
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

        supernode_cfg = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
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
                        
                        for metric_name in scar_metric_list + ["activation_l2_norm", "rayleigh_quotient", "gaussian_mi_analytic", "average_redundancy"]:
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
                                        xlabel=metric_name.replace('_', ' ').title(),
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
                            comparison_metrics = supernode_cfg.get("compute_metrics", 
                                ["activation_l2_norm", "rayleigh_quotient", "scar_activation_power", "scar_loss_proxy"])
                            
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
                                        axes[1].set_title(f"Distribution Comparison")
                                        axes[1].grid(True, alpha=0.3)
                                        
                                        # Add statistics annotation
                                        stats_text = f"Supernode: μ={sn_arr.mean():.4f}, σ={sn_arr.std():.4f}\nNon-SN: μ={non_sn_arr.mean():.4f}, σ={non_sn_arr.std():.4f}"
                                        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                                                     verticalalignment='top', fontsize=9, 
                                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                                        
                                        plt.tight_layout()
                                        fig.savefig(supernode_dir / f"supernode_comparison_{metric_name}.png", dpi=300, bbox_inches="tight")
                                        plt.close(fig)
                                        logger.info(f"Generated supernode comparison plot for {metric_name}")
                                    except Exception as cmp_err:
                                        logger.warning(f"Failed to generate supernode comparison for {metric_name}: {cmp_err}")
                                        
                    except Exception as viz_err:
                        logger.error(f"Failed to generate SCAR visualizations: {viz_err}")

                    # Run supernode connection analysis
                    try:
                        supernode_config = getattr(self.config, "supernode", {}) or getattr(self.config, "supernode_config", {}) or {}
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
                        except Exception as conn_err:
                            logger.error(f"Failed supernode-connectivity computation: {conn_err}")
                            import traceback
                            logger.error(traceback.format_exc())

                    # Supernode Robustness Analysis
                    # Analyzes consistency of supernode identification across metrics and batches
                    robustness_config = getattr(self.config, "supernode_robustness", {}) or {}
                    if robustness_config.get("enabled", False):
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

        # Compute baseline pruning scores (Wanda, SparseGPT) if configured
        # This runs OUTSIDE the SCAR metrics block so it can work independently
        baseline_scores: Dict[str, Any] = {}
        pruning_strategies = getattr(self.config, "pruning_strategies", None) or []
        baseline_strategies = [s for s in pruning_strategies if s in ["wanda", "sparsegpt"]]
        logger.info(f"Checking baseline strategies: pruning_strategies={pruning_strategies}, baseline_strategies={baseline_strategies}")
        if baseline_strategies:
            try:
                baseline_scores = self.compute_baseline_pruning_scores(
                    strategies=baseline_strategies,
                    num_calibration_samples=getattr(self.config, "scar_num_samples", 128),
                )
                logger.info(f"Computed baseline pruning scores for {len(baseline_scores)} layers")
            except Exception as base_err:
                logger.error(f"Failed baseline pruning score computation: {base_err}")
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

        if self.config.do_perplexity_computation:
            baseline_ppl = self.evaluate_perplexity(dataset=self.config.evaluation_dataset, num_samples=self.config.evaluation_num_samples)
            results["evaluation"]["baseline_perplexity"] = baseline_ppl


        if self.config.do_pruning_experiments:
            sparsity_levels = self.config.pruning_amounts
            
            # Get pruning strategies from config - these are now the metrics from metrics.enabled
            pruning_strategies = getattr(self.config, "pruning_strategies", None)
            if not pruning_strategies:
                # Fallback to single metric for backward compatibility
                pruning_strategies = [self.config.pruning_alignment_metric]
            
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
                has_metric_scores = any(
                    metric in layer_scores 
                    for layer_scores in self.importance_scores.values()
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
                            eval_results = self.evaluate_multiple_metrics(
                                metrics=eval_metrics,
                                num_samples=self.config.evaluation_num_samples
                            )
                            
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
                            }
                        else:
                            pruning_data["perplexities"].append(None)
                    
                    # Restore original weights after all sparsity levels for this strategy/mode
                    restore_weights()
                    
                    # Store in unified format for visualization
                    # Filter out None perplexities
                    valid_data = [
                        (s, p) for s, p in zip(pruning_data["sparsities"], pruning_data["perplexities"]) 
                        if p is not None
                    ]
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
                                    v for s, v in zip(pruning_data["sparsities"], pruning_data[eval_metric])
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
