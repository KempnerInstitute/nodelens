"""
Global dropout experiment for analyzing model alignment under dropout.

This module implements experiments that apply the same dropout rate globally
across all layers and track changes in alignment metrics.
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from alignment.core.registry import register_experiment
from alignment.experiments.base import BaseExperiment, ExperimentConfig
from alignment.experiments.config_components import PruningConfig
from alignment.experiments.training_utils import (
    convert_training_history,
    create_experiment_trainer,
    train_with_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class GlobalDropoutConfig(ExperimentConfig):
    """Configuration for global dropout experiment."""
    
    # Dropout configuration
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    dropout_structure: str = 'random'  # 'random', 'magnitude', 'gradient'
    dropout_mode: str = 'scaled'  # 'scaled' or 'unscaled'
    
    # Pruning configuration (when using structured dropout)
    pruning_mode: str = 'global_joint'  # 'global_joint', 'layer_wise', etc.
    pruning_strategy: str = 'low'  # 'low', 'high', 'random'
    exclude_classification_layer: bool = True
    
    # Training configuration
    train_before_dropout: bool = True
    training_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Evaluation
    num_samples: int = 1000
    apply_to_layers: Optional[List[str]] = None
    eval_batches: Optional[int] = None


@register_experiment("global_dropout")
@register_experiment("progressive_dropout")  # Backward compatibility
class GlobalDropoutExperiment(BaseExperiment):
    """
    Experiment for applying global dropout to analyze alignment changes.
    
    This experiment:
    1. Trains a model (if configured)
    2. Applies the same dropout rate globally across all layers
    3. Tracks alignment metrics at each dropout level
    4. Analyzes how dropout affects model structure
    """
    
    def __init__(self, config: GlobalDropoutConfig):
        """
        Initialize global dropout experiment.
        """
        super().__init__(config)
        
        # Results storage
        self.dropout_results = {
            'dropout_rates': self.config.dropout_rates,
            'metrics_by_rate': {},
            'layer_statistics': {}
        }
        
    def _train_model(self) -> Dict[str, Any]:
        """Train the model if configured."""
        if not self.config.train_before_dropout:
            logger.info("Skipping initial training")
            return {}
            
        logger.info(f"Training model for {self.config.training_epochs} epochs")
        
        # Create trainer using the unified interface
        trainer = create_experiment_trainer(
            self.model,
            asdict(self.config),
            device=self.config.device
        )
        
        # Train with metrics
        history = train_with_metrics(
            trainer,
            self.data_loader,
            val_loader=None,
            compute_accuracy=True
        )
        
        # Log final metrics
        if history['train_loss']:
            final_metrics = {
                "train_loss": history['train_loss'][-1],
                "train_accuracy": history['train_metrics'][-1].get('accuracy', 0.0)
            }
            self.log_metrics(len(history['train_loss']) - 1, final_metrics)
        
        return convert_training_history(history)
    
    def run(self, models=None, dataset=None, **kwargs) -> Dict[str, Any]:
        """
        Run the global dropout experiment.
        
        Returns:
            Experiment results including metrics at each dropout rate
        """
        logger.info("Starting global dropout experiment")
        
        # Collect initial model statistics
        self._collect_initial_statistics()
        
        # Get data samples for evaluation
        eval_data = self._get_evaluation_data()
        
        # Test each dropout rate
        for dropout_rate in self.config.dropout_rates:
            logger.info(f"Testing dropout rate: {dropout_rate}")
            
            # Create dropout masks for this rate
            dropout_masks = self._create_dropout_masks(dropout_rate)
            
            # Apply dropout to model
            if dropout_masks:
                self.wrapped_model.apply_structured_dropout(
                    dropout_masks, 
                    mode="multiplicative", 
                    permanent=False
                )
            
            # Compute metrics with dropout
            metrics = self._evaluate_with_dropout(eval_data, dropout_rate)
            
            # Store results
            self.dropout_results['metrics_by_rate'][dropout_rate] = metrics
            
            # Log progress
            self.log_metrics(
                step=int(dropout_rate * 100),  # Use dropout % as step
                metrics=self._flatten_metrics(metrics, dropout_rate)
            )
            
            # Restore original weights
            if dropout_masks:
                self.wrapped_model.restore_weights()
            
            # Optional: Save checkpoint
            if self.config.checkpoint_interval > 0:
                self.save_checkpoint(
                    step=int(dropout_rate * 100),
                    metrics=metrics
                )
        
        # Analyze results
        self._analyze_dropout_effects()
        
        # Save final results
        self.results.update(self.dropout_results)
        self.save_results()
        
        return self.results
    
    def _collect_initial_statistics(self):
        """Collect statistics about the initial model."""
        logger.info("Collecting initial model statistics")
        
        weights = self.wrapped_model.get_layer_weights()
        
        for layer_name, weight in weights.items():
            if weight is None:
                continue
            
            # Compute weight statistics
            stats = {
                'shape': list(weight.shape),
                'num_parameters': weight.numel(),
                'mean': weight.mean().item(),
                'std': weight.std().item(),
                'min': weight.min().item(),
                'max': weight.max().item(),
                'sparsity': (weight == 0).float().mean().item()
            }
            
            # Compute norms
            stats['l1_norm'] = weight.abs().sum().item()
            stats['l2_norm'] = weight.pow(2).sum().sqrt().item()
            
            self.dropout_results['layer_statistics'][layer_name] = stats
    
    def _create_dropout_masks(self, dropout_rate: float) -> Dict[str, torch.Tensor]:
        """
        Create dropout masks for specified layers.
        
        Args:
            dropout_rate: Fraction of units to drop
            
        Returns:
            Dictionary mapping layer names to binary masks
        """
        if dropout_rate == 0.0:
            return {}
            
        dropout_masks = {}
        layers_to_apply = self.config.apply_to_layers or self.wrapped_model.tracked_layers
        
        for layer_name in layers_to_apply:
            layer_info = self.wrapped_model.get_layer_info(layer_name)
            
            if 'weight_shape' not in layer_info:
                continue
                
            # Get number of units based on layer type
            if layer_info['type'] == 'Linear':
                num_units = layer_info['out_features']
            elif layer_info['type'] in ['Conv2d', 'Conv1d']:
                num_units = layer_info['out_channels']
            else:
                continue
            
            # Create mask based on dropout structure
            if self.config.dropout_structure == 'random':
                # Random dropout
                mask = torch.rand(num_units) > dropout_rate
            elif self.config.dropout_structure == 'magnitude':
                # Magnitude-based dropout (keep high magnitude units)
                weights = self.wrapped_model.get_layer_weights([layer_name])[layer_name]
                magnitudes = weights.abs().sum(dim=tuple(range(1, weights.ndim)))
                threshold = torch.quantile(magnitudes, dropout_rate)
                mask = magnitudes > threshold
            elif self.config.dropout_structure == 'gradient':
                # Gradient-based dropout (requires gradients)
                # For now, fallback to random
                mask = torch.rand(num_units) > dropout_rate
            else:
                raise ValueError(f"Unknown dropout structure: {self.config.dropout_structure}")
            
            dropout_masks[layer_name] = mask.float().to(self.config.device)
            
        return dropout_masks
    
    def _get_evaluation_data(self) -> List[torch.Tensor]:
        """Get data samples for evaluation."""
        eval_data = []
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            inputs = inputs.to(self.config.device)
            eval_data.append(inputs)
            
            total_samples += inputs.size(0)
            if total_samples >= self.config.num_samples:
                break
        
        # Concatenate and trim to exact number
        eval_data = torch.cat(eval_data, dim=0)[:self.config.num_samples]
        
        logger.info(f"Collected {eval_data.size(0)} samples for evaluation")
        return eval_data
    
    def _evaluate_with_dropout(
        self,
        eval_data: torch.Tensor,
        dropout_rate: float
    ) -> Dict[str, Any]:
        """
        Evaluate metrics with current dropout settings.
        
        Args:
            eval_data: Data to evaluate on
            dropout_rate: Current dropout rate
            
        Returns:
            Dictionary of metrics
        """
        # Set model to eval mode (but keep dropout active via context manager)
        self.model.eval()
        
        all_metrics = {}
        batch_size = min(self.config.batch_size, eval_data.size(0))
        
        # Process in batches
        for i in range(0, eval_data.size(0), batch_size):
            batch = eval_data[i:i + batch_size]
            
            # Compute metrics for batch
            with torch.no_grad():
                batch_metrics = self.compute_metrics(batch)
            
            # Accumulate metrics
            for metric_name, layer_results in batch_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = {}
                
                for layer_name, value in layer_results.items():
                    if layer_name not in all_metrics[metric_name]:
                        all_metrics[metric_name][layer_name] = []
                    all_metrics[metric_name][layer_name].append(value)
        
        # Average metrics across batches
        averaged_metrics = {}
        for metric_name, layer_results in all_metrics.items():
            averaged_metrics[metric_name] = {}
            for layer_name, values in layer_results.items():
                averaged_metrics[metric_name][layer_name] = np.mean(values)
        
        # Add additional statistics
        averaged_metrics['_statistics'] = {
            'dropout_rate': dropout_rate,
            'num_samples': eval_data.size(0),
            'effective_sparsity': self._compute_effective_sparsity()
        }
        
        return averaged_metrics
    
    def _compute_effective_sparsity(self) -> Dict[str, float]:
        """Compute effective sparsity after dropout."""
        sparsity = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                # Count zeros (including dropout-induced zeros)
                num_zeros = (weight == 0).float().sum().item()
                total_params = weight.numel()
                sparsity[name] = num_zeros / total_params
        
        return sparsity
    
    def _flatten_metrics(
        self,
        metrics: Dict[str, Any],
        dropout_rate: float
    ) -> Dict[str, float]:
        """Flatten metrics dictionary for logging."""
        flat_metrics = {f'dropout_rate': dropout_rate}
        
        for metric_name, layer_results in metrics.items():
            if metric_name.startswith('_'):
                continue  # Skip internal metrics
            
            for layer_name, value in layer_results.items():
                key = f"{metric_name}/{layer_name}"
                flat_metrics[key] = value
        
        return flat_metrics
    
    def _analyze_dropout_effects(self):
        """Analyze how dropout affects alignment metrics."""
        logger.info("Analyzing dropout effects on alignment")
        
        analysis = {
            'metric_trends': {},
            'layer_sensitivity': {},
            'critical_dropout_rates': {}
        }
        
        # Analyze trends for each metric and layer
        for metric_name in self.metrics.keys():
            analysis['metric_trends'][metric_name] = {}
            
            # Get all layers that have this metric
            all_layers = set()
            for rate_results in self.dropout_results['metrics_by_rate'].values():
                if metric_name in rate_results:
                    all_layers.update(rate_results[metric_name].keys())
            
            for layer_name in all_layers:
                # Collect values across dropout rates
                rates = []
                values = []
                
                for rate, results in self.dropout_results['metrics_by_rate'].items():
                    if metric_name in results and layer_name in results[metric_name]:
                        rates.append(rate)
                        values.append(results[metric_name][layer_name])
                
                if len(values) < 2:
                    continue
                
                # Compute trend statistics
                values = np.array(values)
                rates = np.array(rates)
                
                # Linear regression to find trend
                coeffs = np.polyfit(rates, values, 1)
                slope = coeffs[0]
                
                # Find critical points (large changes)
                if len(values) > 2:
                    diffs = np.diff(values)
                    max_change_idx = np.argmax(np.abs(diffs))
                    critical_rate = rates[max_change_idx + 1]
                else:
                    critical_rate = None
                
                analysis['metric_trends'][metric_name][layer_name] = {
                    'slope': float(slope),
                    'initial_value': float(values[0]),
                    'final_value': float(values[-1]),
                    'percent_change': float((values[-1] - values[0]) / (values[0] + 1e-8) * 100),
                    'critical_dropout_rate': float(critical_rate) if critical_rate else None
                }
        
        # Compute layer sensitivity (how much each layer is affected by dropout)
        for layer_name in self.wrapped_model.tracked_layers:
            sensitivities = []
            
            for metric_name in self.metrics.keys():
                if (metric_name in analysis['metric_trends'] and 
                    layer_name in analysis['metric_trends'][metric_name]):
                    
                    trend = analysis['metric_trends'][metric_name][layer_name]
                    sensitivities.append(abs(trend['percent_change']))
            
            if sensitivities:
                analysis['layer_sensitivity'][layer_name] = {
                    'mean_sensitivity': float(np.mean(sensitivities)),
                    'max_sensitivity': float(np.max(sensitivities)),
                    'sensitivity_scores': sensitivities
                }
        
        self.dropout_results['analysis'] = analysis
        logger.info("Dropout analysis complete") 