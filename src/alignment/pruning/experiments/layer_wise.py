"""
Layer-isolated pruning experiment.

This module implements pruning where each layer is pruned independently
based on its alignment scores, without considering other layers.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json

from alignment.experiments.base import BaseExperiment, ExperimentConfig
from alignment.core.registry import register_experiment
from alignment.models import ModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class LayerIsolatedConfig(ExperimentConfig):
    """Configuration for layer-isolated pruning experiment."""
    
    # Dropout configuration
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"  # "scaled" or "unscaled"
    
    # Pruning configuration
    pruning_metric: str = "rayleigh_quotient"  # Metric to use for pruning decisions
    pruning_strategy: str = "low"  # "low", "high", "random"
    exclude_classification_layer: bool = True
    
    # CNN preprocessing mode
    cnn_mode: str = "unfold"  # "unfold", "patchwise", "batch_patch_combined"
    
    # Training configuration
    train_before_dropout: bool = True
    training_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Evaluation
    eval_batches: Optional[int] = None  # None for full dataset
    num_random_trials: int = 3  # Number of random pruning trials


@register_experiment("layer_isolated_pruning")
class LayerIsolatedPruningExperiment(BaseExperiment):
    """
    Experiment for layer-isolated pruning.
    
    This experiment:
    1. Trains a model (if configured)
    2. Computes alignment scores for each layer independently
    3. Prunes neurons in each layer based on their scores
    4. Evaluates performance at different pruning levels
    """
    
    def __init__(self, config: LayerIsolatedConfig):
        """Initialize layer-isolated pruning experiment."""
        super().__init__(config)
        # config is already handled by parent class
        self.pruning_scores = {}
        self.original_weights = {}
        
    def _compute_layer_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute pruning scores for each layer independently.
        
        Returns:
            Dictionary mapping layer names to score tensors
        """
        logger.info(f"Computing {self.config.pruning_metric} scores for each layer")
        
        layer_scores = {}
        metric = self.metrics[self.config.pruning_metric]
        
        # Evaluate on subset of data if configured
        eval_batches = self.config.eval_batches or len(self.data_loader)
        
        for layer_name in self.wrapped_model.tracked_layers:
            if self.config.exclude_classification_layer and "classifier" in layer_name.lower():
                logger.info(f"Skipping classification layer: {layer_name}")
                continue
                
            scores_list = []
            
            # Compute scores over batches
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                if batch_idx >= eval_batches:
                    break
                    
                inputs = inputs.to(self.config.device)
                
                # Forward pass with activation tracking
                _, activations = self.wrapped_model.forward_with_activations(inputs)
                
                # Get layer-specific data
                layer_inputs = activations.get(f"{layer_name}_input")
                layer_weights = self.wrapped_model.get_layer_weights()[layer_name]
                
                if layer_inputs is None or layer_weights is None:
                    continue
                
                # Preprocess activations based on CNN mode
                preprocessed = self.wrapped_model.preprocess_activations(
                    {f"{layer_name}_input": layer_inputs},
                    mode=self.config.cnn_mode if hasattr(self.config, 'cnn_mode') else None
                )
                layer_inputs = preprocessed.get(f"{layer_name}_input", layer_inputs)
                
                # Compute metric scores
                if hasattr(metric, 'requires_outputs') and metric.requires_outputs:
                    layer_outputs = activations.get(f"{layer_name}_output")
                    if layer_outputs is not None:
                        preprocessed_out = self.wrapped_model.preprocess_activations(
                            {f"{layer_name}_output": layer_outputs},
                            mode=self.config.cnn_mode if hasattr(self.config, 'cnn_mode') else None
                        )
                        layer_outputs = preprocessed_out.get(f"{layer_name}_output", layer_outputs)
                    scores = metric.compute(
                        inputs=layer_inputs,
                        weights=layer_weights,
                        outputs=layer_outputs
                    )
                else:
                    scores = metric.compute(
                        inputs=layer_inputs,
                        weights=layer_weights
                    )
                
                scores_list.append(scores.cpu())
            
            # Aggregate scores across batches
            if scores_list:
                layer_scores[layer_name] = torch.cat(scores_list, dim=0).mean(dim=0)
                logger.info(f"Layer {layer_name}: computed {len(layer_scores[layer_name])} scores")
            else:
                logger.warning(f"No scores computed for layer {layer_name}")
        
        return layer_scores
    
    def _create_dropout_masks(
        self, 
        layer_scores: Dict[str, torch.Tensor], 
        dropout_rate: float
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Create dropout masks for each layer based on scores.
        
        Args:
            layer_scores: Scores for each layer
            dropout_rate: Fraction of neurons to drop
            
        Returns:
            Dictionary with masks for each strategy
        """
        masks = {
            "low": {},
            "high": {},
            "random": {}
        }
        
        for layer_name, scores in layer_scores.items():
            num_neurons = len(scores)
            num_drop = int(num_neurons * dropout_rate)
            
            if num_drop == 0:
                # No dropout for this layer
                for strategy in masks:
                    masks[strategy][layer_name] = torch.ones(num_neurons, dtype=torch.bool)
                continue
            
            # Sort scores to get indices
            sorted_indices = torch.argsort(scores)
            
            # Low scores mask (drop lowest scoring neurons)
            low_mask = torch.ones(num_neurons, dtype=torch.bool)
            low_mask[sorted_indices[:num_drop]] = False
            masks["low"][layer_name] = low_mask
            
            # High scores mask (drop highest scoring neurons)
            high_mask = torch.ones(num_neurons, dtype=torch.bool)
            high_mask[sorted_indices[-num_drop:]] = False
            masks["high"][layer_name] = high_mask
            
            # Random mask
            random_mask = torch.ones(num_neurons, dtype=torch.bool)
            random_indices = torch.randperm(num_neurons)[:num_drop]
            random_mask[random_indices] = False
            masks["random"][layer_name] = random_mask
            
            logger.debug(f"Layer {layer_name}: dropping {num_drop}/{num_neurons} neurons")
        
        return masks
    
    def _apply_layer_masks(self, masks: Dict[str, torch.Tensor]):
        """Apply dropout masks to each layer independently."""
        for layer_name, mask in masks.items():
            layer = self.wrapped_model.get_layer(layer_name)
            if layer is None:
                continue
                
            # Store original weights if not already stored
            if layer_name not in self.original_weights:
                self.original_weights[layer_name] = layer.weight.data.clone()
            
            # Apply mask based on layer type
            if hasattr(layer, 'weight'):
                if len(layer.weight.shape) == 2:  # Linear layer
                    # Mask output neurons
                    layer.weight.data = self.original_weights[layer_name].clone()
                    layer.weight.data[~mask] = 0
                    
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if layer_name + "_bias" not in self.original_weights:
                            self.original_weights[layer_name + "_bias"] = layer.bias.data.clone()
                        layer.bias.data = self.original_weights[layer_name + "_bias"].clone()
                        layer.bias.data[~mask] = 0
                        
                elif len(layer.weight.shape) == 4:  # Conv layer
                    # Mask output channels
                    layer.weight.data = self.original_weights[layer_name].clone()
                    layer.weight.data[~mask] = 0
                    
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if layer_name + "_bias" not in self.original_weights:
                            self.original_weights[layer_name + "_bias"] = layer.bias.data.clone()
                        layer.bias.data = self.original_weights[layer_name + "_bias"].clone()
                        layer.bias.data[~mask] = 0
    
    def _restore_original_weights(self):
        """Restore original weights to all layers."""
        for layer_name, original_weight in self.original_weights.items():
            if "_bias" in layer_name:
                continue
                
            layer = self.wrapped_model.get_layer(layer_name)
            if layer is not None and hasattr(layer, 'weight'):
                layer.weight.data = original_weight.clone()
                
                # Restore bias if exists
                bias_key = layer_name + "_bias"
                if bias_key in self.original_weights and hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data = self.original_weights[bias_key].clone()
    
    def _evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluate model performance.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                if self.config.eval_batches and batch_idx >= self.config.eval_batches:
                    break
                    
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _train_model(self):
        """Train the model if configured."""
        if not self.config.train_before_dropout:
            logger.info("Skipping initial training (train_before_dropout=False)")
            return
            
        logger.info(f"Training model for {self.config.training_epochs} epochs")
        
        # Setup optimizer
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.config.training_epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Log epoch results
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            logger.info(f"Epoch {epoch+1}/{self.config.training_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            
            # Log metrics
            self.log_metrics(epoch, {
                "train_loss": avg_loss,
                "train_accuracy": accuracy
            })
    
    def run(self) -> Dict[str, Any]:
        """
        Run the layer-isolated pruning experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        logger.info("Starting layer-isolated pruning experiment")
        
        # Train model if configured
        self._train_model()
        
        # Compute pruning scores for each layer
        layer_scores = self._compute_layer_scores()
        self.pruning_scores = layer_scores
        
        # Initialize results
        results = {
            "dropout_rates": self.config.dropout_rates,
            "layer_scores": {k: v.tolist() for k, v in layer_scores.items()},
            "accuracies": {
                "low": [],
                "high": [],
                "random": []
            },
            "losses": {
                "low": [],
                "high": [],
                "random": []
            },
            "per_layer_masks": {}
        }
        
        # Evaluate at each dropout rate
        for dropout_rate in self.config.dropout_rates:
            logger.info(f"Evaluating dropout rate: {dropout_rate}")
            
            # Create masks for this dropout rate
            masks = self._create_dropout_masks(layer_scores, dropout_rate)
            
            # Store mask info
            results["per_layer_masks"][f"dropout_{dropout_rate}"] = {
                layer: {
                    "total_neurons": len(mask),
                    "active_neurons": int(mask.sum().item())
                }
                for layer, mask in masks["low"].items()
            }
            
            # Evaluate each strategy
            for strategy in ["low", "high", "random"]:
                if strategy == "random":
                    # Average over multiple random trials
                    trial_losses = []
                    trial_accs = []
                    
                    for trial in range(self.config.num_random_trials):
                        # Create new random masks
                        random_masks = self._create_dropout_masks(layer_scores, dropout_rate)["random"]
                        
                        # Apply masks
                        self._apply_layer_masks(random_masks)
                        
                        # Evaluate
                        loss, acc = self._evaluate_model()
                        trial_losses.append(loss)
                        trial_accs.append(acc)
                        
                        # Restore weights
                        self._restore_original_weights()
                    
                    # Average results
                    avg_loss = np.mean(trial_losses)
                    avg_acc = np.mean(trial_accs)
                    
                else:
                    # Apply masks for this strategy
                    self._apply_layer_masks(masks[strategy])
                    
                    # Evaluate
                    avg_loss, avg_acc = self._evaluate_model()
                    
                    # Restore original weights
                    self._restore_original_weights()
                
                # Store results
                results["losses"][strategy].append(avg_loss)
                results["accuracies"][strategy].append(avg_acc)
                
                logger.info(f"  {strategy}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")
                
                # Log metrics
                self.log_metrics(len(results["losses"][strategy]) - 1, {
                    f"{strategy}_loss": avg_loss,
                    f"{strategy}_accuracy": avg_acc,
                    "dropout_rate": dropout_rate
                })
        
        # Save final results
        self.results.update(results)
        self.save_results()
        
        # Save checkpoint with final state
        self.save_checkpoint(
            step=len(self.config.dropout_rates),
            metrics={"final_results": results}
        )
        
        logger.info("Layer-isolated pruning experiment completed")
        
        return results 