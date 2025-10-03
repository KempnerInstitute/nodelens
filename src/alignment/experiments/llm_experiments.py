"""
LLM-specific experiment classes for alignment analysis and pruning.

This module extends the general experiment framework to support
Large Language Model experiments including:
- Neuron importance analysis
- Structured pruning of MLP/attention layers
- Perplexity evaluation
- Multi-metric comparison
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging

from .base import BaseExperiment
from ..models.wrappers_transformer import TransformerWrapper
from ..metrics import get_metric
from ..pruning import AlignmentPruning, PruningConfig
from ..training.base import BaseTrainer

logger = logging.getLogger(__name__)


class LLMAlignmentExperiment(BaseExperiment):
    """
    Experiment for analyzing neuron alignment in Large Language Models.
    
    This experiment:
    1. Loads an LLM (via HuggingFace)
    2. Computes importance scores for neurons using alignment metrics
    3. Optionally prunes neurons
    4. Evaluates performance (perplexity, etc.)
    
    Example config:
        experiment:
          name: "llama3_alignment_analysis"
          type: "llm_alignment"
        
        model:
          name: "hf_causal_lm"
          model_id: "meta-llama/Meta-Llama-3-8B-Instruct"
          torch_dtype: "bfloat16"
        
        wrapper:
          name: "transformer_wrapper"
          tracked_layers:
            - "model.layers.*.mlp"  # Supports wildcards
        
        alignment:
          metrics: ["rayleigh_quotient", "mutual_information_gaussian"]
        
        pruning:
          enabled: true
          algorithms: ["alignment"]
          sparsity_levels: [0.1, 0.2, 0.3]
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tokenizer = None
        self.importance_scores = {}
        self.pruning_masks = {}
        self.evaluation_results = {}
    
    def setup(self):
        """Setup experiment components."""
        logger.info("Setting up LLM experiment...")
        
        # Load model and tokenizer together
        self._load_model_and_tokenizer()
        
        # Setup metrics
        self._setup_metrics()
        
        # Setup pruning if enabled
        if self.config.get('pruning', {}).get('enabled', False):
            self._setup_pruning()
    
    def _load_model_and_tokenizer(self):
        """Load LLM model and tokenizer."""
        from transformers import AutoTokenizer
        
        model_config = self.config.get('model', {})
        model_id = model_config.get('model_id')
        
        if not model_id:
            raise ValueError("model_id must be specified for LLM experiments")
        
        logger.info(f"Loading tokenizer for {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model using parent class method
        super().setup()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def _expand_layer_patterns(self, patterns: List[str], model: nn.Module) -> List[str]:
        """
        Expand layer patterns with wildcards to actual layer names.
        
        Supports patterns like:
        - "model.layers.*.mlp" -> ["model.layers.0.mlp", "model.layers.1.mlp", ...]
        - "model.layers.[0-15].self_attn" -> first 16 attention layers
        """
        import re
        
        expanded = []
        all_names = [name for name, _ in model.named_modules()]
        
        for pattern in patterns:
            if '*' in pattern:
                # Convert glob pattern to regex
                regex_pattern = pattern.replace('.', r'\.').replace('*', r'\d+')
                regex = re.compile(regex_pattern)
                expanded.extend([name for name in all_names if regex.match(name)])
            elif '[' in pattern and ']' in pattern:
                # Range pattern like [0-15]
                import re
                match = re.search(r'\[(\d+)-(\d+)\]', pattern)
                if match:
                    start, end = int(match.group(1)), int(match.group(2))
                    base_pattern = pattern[:match.start()] + '{}' + pattern[match.end():]
                    expanded.extend([base_pattern.format(i) for i in range(start, end + 1)])
            else:
                # Exact match
                if pattern in all_names:
                    expanded.append(pattern)
        
        return expanded
    
    def compute_importance_scores(
        self,
        calibration_texts: Optional[List[str]] = None,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for all tracked layers.
        
        Args:
            calibration_texts: Optional list of texts for calibration
            num_samples: Number of samples to use for importance computation
            
        Returns:
            Dictionary mapping layer names to importance score tensors
        """
        logger.info("Computing importance scores...")
        
        if calibration_texts is None:
            calibration_texts = [
                "The quick brown fox jumps over the lazy dog. " * 20
            ]
        
        self.model.eval()
        
        # Accumulate activations from multiple samples
        all_activations = {}
        
        for text in calibration_texts[:num_samples]:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs, activations = self.model.forward_with_activations(**inputs)
            
            # Accumulate activations
            for key, value in activations.items():
                if key not in all_activations:
                    all_activations[key] = []
                all_activations[key].append(value)
        
        # Average activations if multiple samples
        if len(calibration_texts) > 1:
            all_activations = {
                key: torch.cat(values, dim=0) 
                for key, values in all_activations.items()
            }
        else:
            all_activations = {
                key: values[0] 
                for key, values in all_activations.items()
            }
        
        # Compute importance for each layer
        alignment_config = self.config.get('alignment', {})
        metric_names = alignment_config.get('metrics', ['rayleigh_quotient'])
        
        for layer_name in self.model.tracked_layers:
            logger.info(f"Computing scores for {layer_name}")
            
            layer_module = dict(self.model._model.named_modules())[layer_name]
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
                    MetricClass = get_metric(metric_name)
                    metric = MetricClass()
                    scores = metric.compute(inputs=layer_inputs, weights=weight)
                    layer_scores[metric_name] = scores
                    
                    logger.debug(
                        f"  {metric_name}: "
                        f"mean={scores.mean().item():.6f}, "
                        f"std={scores.std().item():.6f}"
                    )
                except Exception as e:
                    logger.error(f"Error computing {metric_name} for {layer_name}: {e}")
                    continue
            
            self.importance_scores[layer_name] = layer_scores
        
        return self.importance_scores
    
    def _get_layer_weights(self, layer_module: nn.Module) -> Optional[torch.Tensor]:
        """Get the appropriate weight tensor from a layer module."""
        # For MLP layers
        if hasattr(layer_module, 'gate_proj'):
            return layer_module.gate_proj.weight
        elif hasattr(layer_module, 'up_proj'):
            return layer_module.up_proj.weight
        elif hasattr(layer_module, 'fc1'):
            return layer_module.fc1.weight
        # For attention layers
        elif hasattr(layer_module, 'q_proj'):
            return layer_module.q_proj.weight
        # Generic
        elif hasattr(layer_module, 'weight'):
            return layer_module.weight
        
        logger.warning(f"No suitable weight tensor found for {layer_module}")
        return None
    
    def apply_pruning(
        self,
        sparsity: float = 0.2,
        metric: str = 'rayleigh_quotient',
        mode: str = 'low'
    ) -> Dict[str, torch.Tensor]:
        """
        Apply pruning to the model based on importance scores.
        
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
        
        config = PruningConfig(
            amount=sparsity,
            structured=True,
            pruning_mode=mode
        )
        
        pruner = AlignmentPruning(metric=metric, config=config)
        
        masks = {}
        for layer_name in self.importance_scores.keys():
            if metric not in self.importance_scores[layer_name]:
                continue
            
            scores = self.importance_scores[layer_name][metric]
            layer_module = dict(self.model._model.named_modules())[layer_name]
            
            # Get target module for pruning
            if hasattr(layer_module, 'gate_proj'):
                target = layer_module.gate_proj
            elif hasattr(layer_module, 'up_proj'):
                target = layer_module.up_proj
            else:
                continue
            
            try:
                mask = pruner.create_pruning_mask(scores)
                pruner.apply_pruning(target, mask)
                masks[layer_name] = mask
                
                sparsity_achieved = (mask == 0).float().mean().item()
                logger.info(f"  {layer_name}: {sparsity_achieved:.2%} sparsity")
            except Exception as e:
                logger.error(f"Error pruning {layer_name}: {e}")
        
        self.pruning_masks = masks
        return masks
    
    def evaluate_perplexity(
        self,
        dataset: str = 'wikitext',
        split: str = 'test',
        num_samples: int = 100
    ) -> float:
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
        dataset_obj = load_text_dataset(
            dataset,
            self.tokenizer,
            split=split,
            max_samples=num_samples
        )
        
        # Compute perplexity
        self.model.eval()
        nlls = []
        total_length = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataset_obj):
                if i >= num_samples:
                    break
                
                input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)
                
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
    
    def run(self) -> Dict[str, Any]:
        """Run the complete LLM experiment."""
        logger.info("Running LLM alignment experiment...")
        
        results = {
            'config': self.config,
            'importance_scores': {},
            'pruning_results': {},
            'evaluation': {}
        }
        
        # Compute importance scores
        calibration_config = self.config.get('importance_computation', {})
        scores = self.compute_importance_scores(
            num_samples=calibration_config.get('num_samples', 1)
        )
        
        # Save scores summary
        for layer_name, layer_scores in scores.items():
            results['importance_scores'][layer_name] = {
                metric: {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max())
                }
                for metric, scores in layer_scores.items()
            }
        
        # Evaluate baseline
        eval_config = self.config.get('evaluation', {})
        if eval_config.get('compute_perplexity', False):
            baseline_ppl = self.evaluate_perplexity(
                dataset=eval_config.get('dataset', 'wikitext'),
                num_samples=eval_config.get('num_samples', 100)
            )
            results['evaluation']['baseline_perplexity'] = baseline_ppl
        
        # Apply pruning if enabled
        pruning_config = self.config.get('pruning', {})
        if pruning_config.get('enabled', False):
            sparsity_levels = pruning_config.get('sparsity_levels', [0.2])
            metric = pruning_config.get('alignment_metric', 'rayleigh_quotient')
            
            for sparsity in sparsity_levels:
                masks = self.apply_pruning(sparsity=sparsity, metric=metric)
                
                # Evaluate pruned model
                if eval_config.get('compute_perplexity', False):
                    pruned_ppl = self.evaluate_perplexity(
                        dataset=eval_config.get('dataset', 'wikitext'),
                        num_samples=eval_config.get('num_samples', 100)
                    )
                    
                    results['pruning_results'][f'sparsity_{sparsity}'] = {
                        'perplexity': pruned_ppl,
                        'sparsity': sparsity,
                        'num_pruned_layers': len(masks)
                    }
        
        return results


# Register experiment type
try:
    from ..core.registry import register_experiment
    register_experiment("llm_alignment", LLMAlignmentExperiment)
except ImportError:
    pass

