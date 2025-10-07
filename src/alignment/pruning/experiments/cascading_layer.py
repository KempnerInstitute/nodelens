"""
Cascading layer pruning experiment.

This module implements progressive pruning that cascades through layers,
where pruning in earlier layers affects later layers.
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from alignment.core.registry import register_experiment
from alignment.experiments.base import BaseExperiment, ExperimentConfig
from alignment.experiments.training_utils import convert_training_history, create_experiment_trainer, train_with_metrics
from alignment.models import ModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class CascadingConfig(ExperimentConfig):
    """Configuration for cascading layer pruning experiment."""

    # Dropout configuration
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"  # "scaled" or "unscaled"
    cascade_direction: str = "forward"  # "forward" or "backward"

    # Pruning configuration
    pruning_metric: str = "rayleigh_quotient"
    pruning_strategy: str = "low"  # "low", "high", "random"
    exclude_classification_layer: bool = True
    recompute_scores: bool = True  # Whether to recompute scores after each layer pruning

    # CNN preprocessing mode
    cnn_mode: str = "unfold"  # "unfold", "patchwise", "batch_patch_combined"

    # Training configuration
    train_before_dropout: bool = True
    training_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"

    # Evaluation
    eval_batches: Optional[int] = None
    num_random_trials: int = 3


@register_experiment("cascading_layer_pruning")
class CascadingLayerPruningExperiment(BaseExperiment):
    """
    Experiment for cascading layer pruning.

    This experiment:
    1. Trains a model (if configured)
    2. Progressively prunes layers in order (forward or backward)
    3. Optionally recomputes alignment scores after each layer
    4. Evaluates performance at different pruning levels
    """

    def __init__(self, config: CascadingConfig):
        """Initialize cascading layer pruning experiment."""
        super().__init__(config)
        self.original_weights = {}
        self.layer_order = []

    def _get_layer_order(self) -> List[str]:
        """
        Get the order of layers for cascading pruning.

        Returns:
            List of layer names in pruning order
        """
        layers = [
            name for name in self.wrapped_model.tracked_layers if not (self.config.exclude_classification_layer and "classifier" in name.lower())
        ]

        if self.config.cascade_direction == "backward":
            layers = layers[::-1]

        logger.info(f"Layer pruning order ({self.config.cascade_direction}): {layers}")
        return layers

    def _get_layer_type(self, layer_name: str) -> str:
        """Get the type of a layer (linear, conv, etc.)."""
        layer_info = self.wrapped_model.get_layer_info(layer_name)
        return layer_info.get("type", "unknown").lower()

    def _get_appropriate_metric(self, layer_name: str):
        """Get the appropriate metric for a layer based on its type."""
        layer_type = self._get_layer_type(layer_name)

        # Use patchwise RQ for conv layers if using RQ metric
        if self.config.pruning_metric == "rayleigh_quotient" and "conv" in layer_type:
            # Check if patchwise variant exists
            if "rq_patchwise" in self.metrics:
                logger.debug(f"Using patchwise RQ for conv layer {layer_name}")
                return self.metrics["rq_patchwise"]

        # Default to configured metric
        return self.metrics[self.config.pruning_metric]

    def _compute_alignment_scores(self, layer_name: str, active_masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute alignment scores for a specific layer.

        Args:
            layer_name: Name of the layer to compute scores for
            active_masks: Current active masks for all layers

        Returns:
            Alignment scores for the layer
        """
        metric = self._get_appropriate_metric(layer_name)
        scores_list = []

        # Apply current masks if provided
        if active_masks:
            self._apply_masks(active_masks)

        # Compute scores
        eval_batches = self.config.eval_batches or len(self.data_loader)

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            if batch_idx >= eval_batches:
                break

            inputs = inputs.to(self.config.device)

            # Forward pass
            _, activations = self.wrapped_model.forward_with_activations(inputs)

            # Get layer data
            layer_inputs = activations.get(f"{layer_name}_input")
            layer_weights = self.wrapped_model.get_layer_weights()[layer_name]

            if layer_inputs is None or layer_weights is None:
                continue

            # Preprocess activations based on CNN mode
            from alignment.dataops.processing import preprocess_layer_activations

            layer_modules = dict(self.wrapped_model._model.named_modules())
            preprocessed = preprocess_layer_activations(
                {f"{layer_name}_input": layer_inputs}, layer_modules, mode=self.config.cnn_mode if hasattr(self.config, "cnn_mode") else None
            )
            layer_inputs = preprocessed.get(f"{layer_name}_input", layer_inputs)

            # Compute metric
            if hasattr(metric, "requires_outputs") and metric.requires_outputs:
                layer_outputs = activations.get(f"{layer_name}_output")
                if layer_outputs is not None:
                    preprocessed_out = preprocess_layer_activations(
                        {f"{layer_name}_output": layer_outputs},
                        layer_modules,
                        mode=self.config.cnn_mode if hasattr(self.config, "cnn_mode") else None,
                    )
                    layer_outputs = preprocessed_out.get(f"{layer_name}_output", layer_outputs)
                scores = metric.compute(inputs=layer_inputs, weights=layer_weights, outputs=layer_outputs)
            else:
                scores = metric.compute(inputs=layer_inputs, weights=layer_weights)

            scores_list.append(scores.cpu())

        # Restore original weights
        if active_masks:
            self._restore_original_weights()

        # Aggregate scores
        if scores_list:
            # Stack scores from different batches and average across batches
            return torch.stack(scores_list, dim=0).mean(dim=0)
        else:
            logger.warning(f"No scores computed for layer {layer_name}")
            return torch.zeros(1)

    def _create_layer_mask(self, scores: torch.Tensor, dropout_rate: float, strategy: str = "low", layer_name: Optional[str] = None) -> torch.Tensor:
        """
        Create a dropout mask for a single layer.

        Args:
            scores: Alignment scores for the layer
            dropout_rate: Fraction to drop
            strategy: Pruning strategy
            layer_name: Optional layer name to get actual layer size

        Returns:
            Boolean mask (True = keep, False = drop)
        """
        # Get actual layer size if layer name provided
        if layer_name is not None:
            layer = self.wrapped_model.get_layer(layer_name)
            if layer is not None and hasattr(layer, "weight"):
                # Use actual output dimension size
                if len(layer.weight.shape) >= 1:
                    actual_neurons = layer.weight.shape[0]
                    if scores.numel() != actual_neurons:
                        logger.warning(f"Score size ({scores.numel()}) doesn't match layer size ({actual_neurons}). " f"Using layer size for mask.")
                        # Create mask based on actual layer size
                        if scores.numel() == 1:
                            # Single score - apply uniformly
                            return torch.ones(actual_neurons, dtype=torch.bool)
                        else:
                            # Resize scores if possible
                            scores = scores[:actual_neurons] if scores.numel() > actual_neurons else scores

        # Handle scalar scores (0-d tensor)
        if scores.dim() == 0:
            logger.warning("Scores is a scalar, creating single-neuron mask")
            return torch.ones(1, dtype=torch.bool)

        # Get number of neurons
        num_neurons = scores.numel()
        num_drop = int(num_neurons * dropout_rate)

        if num_drop == 0:
            return torch.ones(num_neurons, dtype=torch.bool)

        # Ensure scores is 1D
        scores = scores.flatten()

        if strategy == "low":
            # Drop lowest scoring neurons
            sorted_indices = torch.argsort(scores)
            mask = torch.ones(num_neurons, dtype=torch.bool)
            mask[sorted_indices[:num_drop]] = False
        elif strategy == "high":
            # Drop highest scoring neurons
            sorted_indices = torch.argsort(scores)
            mask = torch.ones(num_neurons, dtype=torch.bool)
            mask[sorted_indices[-num_drop:]] = False
        else:  # random
            mask = torch.ones(num_neurons, dtype=torch.bool)
            random_indices = torch.randperm(num_neurons)[:num_drop]
            mask[random_indices] = False

        return mask

    def _apply_masks(self, masks: Dict[str, torch.Tensor]):
        """Apply dropout masks to layers."""
        for layer_name, mask in masks.items():
            layer = self.wrapped_model.get_layer(layer_name)
            if layer is None:
                continue

            # Store original weights
            if layer_name not in self.original_weights:
                self.original_weights[layer_name] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    self.original_weights[layer_name + "_bias"] = layer.bias.data.clone()

            # Apply mask
            if hasattr(layer, "weight"):
                layer.weight.data = self.original_weights[layer_name].clone()
                if len(layer.weight.shape) == 2:  # Linear layer
                    # Mask output neurons (rows)
                    layer.weight.data[~mask] = 0
                elif len(layer.weight.shape) == 4:  # Conv layer
                    # Mask output channels
                    # Expand mask to match weight dimensions
                    expanded_mask = mask.view(-1, 1, 1, 1).expand_as(layer.weight)
                    layer.weight.data[~expanded_mask] = 0

                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data = self.original_weights[layer_name + "_bias"].clone()
                    layer.bias.data[~mask] = 0

    def _restore_original_weights(self):
        """Restore all layers to original weights."""
        for layer_name, original_weight in self.original_weights.items():
            if "_bias" in layer_name:
                continue

            layer = self.wrapped_model.get_layer(layer_name)
            if layer is not None and hasattr(layer, "weight"):
                layer.weight.data = original_weight.clone()

                bias_key = layer_name + "_bias"
                if bias_key in self.original_weights and hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data = self.original_weights[bias_key].clone()

    def _evaluate_model(self) -> Tuple[float, float]:
        """Evaluate model performance."""
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
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _train_model(self) -> Dict[str, Any]:
        """Train the model if configured."""
        if not self.config.train_before_dropout:
            logger.info("Skipping initial training")
            return {}

        logger.info(f"Training model for {self.config.training_epochs} epochs")

        # Create trainer using the unified interface
        trainer = create_experiment_trainer(self.model, asdict(self.config), device=self.config.device)

        # Train with metrics
        history = train_with_metrics(trainer, self.data_loader, val_loader=None, compute_accuracy=True)  # No validation in original implementation

        # Log final metrics (trainer already logs per-epoch)
        if history["train_loss"]:
            final_metrics = {"train_loss": history["train_loss"][-1], "train_accuracy": history["train_metrics"][-1].get("accuracy", 0.0)}
            self.log_metrics(len(history["train_loss"]) - 1, final_metrics)

        # Return training results
        return convert_training_history(history)

    def _cascading_prune(self, dropout_rate: float, strategy: str = "low") -> Tuple[Dict[str, torch.Tensor], Dict[str, List[float]]]:
        """
        Perform cascading pruning at a specific dropout rate.

        Args:
            dropout_rate: Target dropout rate
            strategy: Pruning strategy

        Returns:
            Tuple of (final masks, layer-wise scores history)
        """
        masks = {}
        scores_history = {}

        # Process layers in order
        for layer_idx, layer_name in enumerate(self.layer_order):
            logger.debug(f"Processing layer {layer_idx+1}/{len(self.layer_order)}: {layer_name}")

            # Compute scores (with previously pruned layers masked)
            if self.config.recompute_scores and masks:
                scores = self._compute_alignment_scores(layer_name, masks)
            else:
                # Use initial scores
                scores = self._compute_alignment_scores(layer_name)

            scores_history[layer_name] = scores.flatten().tolist() if scores.dim() > 0 else [scores.item()]

            # Create mask for this layer
            mask = self._create_layer_mask(scores, dropout_rate, strategy, layer_name)
            masks[layer_name] = mask

            # Log pruning info
            active_neurons = int(mask.sum().item())
            total_neurons = len(mask)
            logger.debug(f"  Layer {layer_name}: {active_neurons}/{total_neurons} neurons active")

        return masks, scores_history

    def run(self) -> Dict[str, Any]:
        """
        Run the cascading layer pruning experiment.

        Returns:
            Dictionary containing experiment results
        """
        logger.info("Starting cascading layer pruning experiment")

        # Train model
        training_results = self._train_model()

        # Get layer order
        self.layer_order = self._get_layer_order()

        # Initialize results
        results = {
            "config": self.config.to_dict(),
            "layer_order": self.layer_order,
            "dropout_rates": self.config.dropout_rates,
            "accuracies": {"low": [], "high": [], "random": []},
            "losses": {"low": [], "high": [], "random": []},
            "layer_scores": {},
            "cascade_masks": {},
            "training_results": training_results,  # Include training results
        }

        # Evaluate at each dropout rate
        for dropout_idx, dropout_rate in enumerate(self.config.dropout_rates):
            logger.info(f"Evaluating dropout rate: {dropout_rate}")

            # Store masks for this dropout rate
            cascade_info = {}

            # Evaluate each strategy
            for strategy in ["low", "high", "random"]:
                if strategy == "random":
                    # Average over multiple trials
                    trial_losses = []
                    trial_accs = []

                    for trial in range(self.config.num_random_trials):
                        # Perform cascading pruning
                        masks, _ = self._cascading_prune(dropout_rate, strategy)

                        # Apply masks
                        self._apply_masks(masks)

                        # Evaluate
                        loss, acc = self._evaluate_model()
                        trial_losses.append(loss)
                        trial_accs.append(acc)

                        # Restore weights
                        self._restore_original_weights()

                    avg_loss = np.mean(trial_losses)
                    avg_acc = np.mean(trial_accs)

                else:
                    # Perform cascading pruning
                    masks, scores_history = self._cascading_prune(dropout_rate, strategy)

                    # Store cascade info
                    if strategy == "low":  # Store detailed info only for one strategy
                        cascade_info = {
                            "masks": {k: v.tolist() for k, v in masks.items()},
                            "scores": scores_history,
                            "active_neurons": {k: int(v.sum().item()) for k, v in masks.items()},
                        }

                    # Apply masks
                    self._apply_masks(masks)

                    # Evaluate
                    avg_loss, avg_acc = self._evaluate_model()

                    # Restore weights
                    self._restore_original_weights()

                # Store results
                results["losses"][strategy].append(avg_loss)
                results["accuracies"][strategy].append(avg_acc)

                logger.info(f"  {strategy}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")

                # Log metrics
                self.log_metrics(
                    dropout_idx * 3 + ["low", "high", "random"].index(strategy),
                    {f"{strategy}_loss": avg_loss, f"{strategy}_accuracy": avg_acc, "dropout_rate": dropout_rate},
                )

            # Store cascade info
            results["cascade_masks"][f"dropout_{dropout_rate}"] = cascade_info

        # Save results
        self.results.update(results)
        self.save_results()

        # Save final checkpoint
        self.save_checkpoint(step=len(self.config.dropout_rates), metrics={"final_results": results})

        logger.info("Cascading layer pruning experiment completed")

        return results
