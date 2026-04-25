"""
Training callbacks for alignment metrics.

Enables computing alignment metrics during training with minimal overhead.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class AlignmentMetricsCallback:
    """
    Lightweight callback to compute alignment metrics during training.

    Piggybacks on the training forward pass with no extra computation.
    Computes metrics every N steps using cached activations.

    Key features:
    - Zero extra forward passes
    - Minimal overhead (~5-10ms per frequency)
    - Automatic activation capture via HookManager
    - Supports any tracker (WandB, TensorBoard, etc.)

    Example:
        >>> from nodelens.training.callbacks import AlignmentMetricsCallback
        >>> callback = AlignmentMetricsCallback(
        ...     metrics={'rq': get_metric('rayleigh_quotient')},
        ...     layers=['conv1', 'fc1'],
        ...     frequency=100
        ... )
        >>>
        >>> # In training loop:
        >>> for batch_idx, (inputs, targets) in enumerate(train_loader):
        ...     outputs = model(inputs)
        ...     loss.backward()
        ...     optimizer.step()
        ...
        ...     # Compute metrics (minimal overhead)
        ...     callback.on_batch_end(wrapper, inputs, targets, global_step)
    """

    def __init__(
        self,
        metrics: Dict[str, Any],
        layers: List[str],
        frequency: int = 100,
        sample_size: Optional[int] = None,
        aggregation: str = "mean",
        tracker: Optional[Any] = None,
        save_history: bool = True,
        track_per_neuron: bool = False,
    ):
        """
        Initialize alignment metrics callback.

        Args:
            metrics: Dict of initialized metrics (e.g., {'rq': RayleighQuotient()})
            layers: Layer names to track
            frequency: Compute metrics every N steps
            sample_size: Max samples to use (None = use full batch)
            aggregation: How to aggregate scores ('mean', 'std', 'both')
            tracker: Optional tracker (WandB, TensorBoard, etc.)
            save_history: Whether to store history for later analysis
            track_per_neuron: If True, also store full per-neuron score tensors (on CPU) in
                `tensor_history` for dynamic scoring / post-hoc analysis. Note: this can be memory-intensive.
        """
        self.metrics = metrics
        self.layers = layers
        self.frequency = frequency
        self.sample_size = sample_size
        self.aggregation = aggregation
        self.tracker = tracker
        self.save_history = save_history
        self.track_per_neuron = track_per_neuron

        # Initialize history storage
        if save_history:
            self.history = {layer: {metric_name: [] for metric_name in metrics} for layer in layers}
            self.step_history = []

        # Optional: store full per-neuron score tensors for each tracked step.
        if track_per_neuron:
            self.tensor_history = {layer: {metric_name: [] for metric_name in metrics} for layer in layers}

        self.step = 0

    def on_batch_end(self, model_wrapper, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, step: Optional[int] = None, **kwargs):
        """
        Compute metrics at the end of a training batch.

        Args:
            model_wrapper: Model wrapper with HookManager
            inputs: Input batch (same as used for forward)
            targets: Target labels (optional)
            step: Global step number (if None, uses internal counter)
            **kwargs: Additional arguments (e.g., for specific metrics)
        """
        # Update step counter
        if step is not None:
            self.step = step
        else:
            self.step += 1

        # Only compute at specified frequency
        if self.step % self.frequency != 0:
            return

        # Record the step for history alignment (e.g., dynamic scoring vs loss curves).
        if self.save_history:
            self.step_history.append(self.step)

        # Sample subset for efficiency (if large batch)
        if self.sample_size is not None and inputs.size(0) > self.sample_size:
            indices = torch.randperm(inputs.size(0))[: self.sample_size]
            inputs_sampled = inputs[indices]
            targets_sampled = targets[indices] if targets is not None else None
        else:
            inputs_sampled = inputs
            targets_sampled = targets

        # Capture activations (reuses existing forward, no extra pass)
        # Use no_grad to avoid interfering with training
        with torch.no_grad():
            try:
                # Use HookManager for safe capture
                outputs, activations = model_wrapper.forward_with_activations(inputs_sampled)
                weights = model_wrapper.get_layer_weights(self.layers)

                # Compute metrics for each layer
                for layer in self.layers:
                    self._compute_layer_metrics(layer, activations, weights, targets_sampled, **kwargs)

            except Exception as e:
                logger.warning(f"Failed to compute alignment metrics at step {self.step}: {e}")

    def _compute_layer_metrics(
        self, layer: str, activations: Dict[str, torch.Tensor], weights: Dict[str, torch.Tensor], targets: Optional[torch.Tensor], **kwargs
    ):
        """Compute metrics for a single layer."""
        # Get layer data
        layer_input = activations.get(f"{layer}_input")
        layer_output = activations.get(f"{layer}_output")
        layer_weights = weights.get(layer)

        if layer_weights is None:
            logger.warning(f"No weights found for layer {layer}")
            return

        # Compute each metric
        for metric_name, metric in self.metrics.items():
            try:
                # Determine what to pass based on metric requirements
                metric_kwargs = {}
                if metric.requires_inputs and layer_input is not None:
                    metric_kwargs["inputs"] = layer_input
                if metric.requires_weights:
                    metric_kwargs["weights"] = layer_weights
                if metric.requires_outputs and layer_output is not None:
                    metric_kwargs["outputs"] = layer_output
                if targets is not None:
                    metric_kwargs["targets"] = targets

                # Compute metric
                scores = metric.compute(**metric_kwargs, **kwargs)

                # Aggregate to scalar
                if self.aggregation == "mean":
                    value = scores.mean().item()
                elif self.aggregation == "std":
                    value = scores.std().item()
                elif self.aggregation == "both":
                    value = {"mean": scores.mean().item(), "std": scores.std().item()}
                else:
                    value = scores.mean().item()

                # Store history
                if self.save_history:
                    if self.aggregation == "both":
                        self.history[layer][metric_name].append(value)
                    else:
                        self.history[layer][metric_name].append(value)

                # Optionally store per-neuron tensors for post-hoc analysis.
                if self.track_per_neuron and hasattr(self, "tensor_history"):
                    try:
                        self.tensor_history[layer][metric_name].append(scores.detach().float().cpu())
                    except Exception as e:
                        logger.warning(f"Failed to store tensor history for {layer}/{metric_name}: {e}")

                # Log to tracker
                if self.tracker:
                    if self.aggregation == "both":
                        self.tracker.log({f"{layer}/{metric_name}_mean": value["mean"], f"{layer}/{metric_name}_std": value["std"]}, step=self.step)
                    else:
                        self.tracker.log({f"{layer}/{metric_name}": value}, step=self.step)

                logger.debug(f"Step {self.step}, {layer}/{metric_name}: {value}")

            except Exception as e:
                logger.warning(f"Failed to compute {metric_name} for {layer}: {e}")

    def get_history(self) -> Dict:
        """Get complete metrics history."""
        if not self.save_history:
            logger.warning("History not saved (save_history=False)")
            return {}

        out = {
            "history": self.history,
            "steps": self.step_history if hasattr(self, "step_history") else list(range(0, self.step + 1, self.frequency)),
        }
        if self.track_per_neuron and hasattr(self, "tensor_history"):
            out["tensor_history"] = self.tensor_history
        return out

    def reset(self):
        """Reset history and step counter."""
        self.step = 0
        if self.save_history:
            self.history = {layer: {metric_name: [] for metric_name in self.metrics} for layer in self.layers}
            self.step_history = []

    def save_history(self, path: str):
        """Save history to file."""
        if not self.save_history:
            logger.warning("No history to save")
            return

        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        history_data = {
            "history": self.history,
            "steps": list(range(0, self.step + 1, self.frequency)),
            "layers": self.layers,
            "metrics": list(self.metrics.keys()),
            "frequency": self.frequency,
        }

        with open(path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Saved alignment history to {path}")

        # Save tensor history separately (binary) if enabled.
        if self.track_per_neuron and hasattr(self, "tensor_history"):
            tensor_path = path.with_name(f"{path.stem}_tensors.pt")
            try:
                torch.save(
                    {
                        "tensor_history": self.tensor_history,
                        "steps": history_data["steps"],
                        "layers": self.layers,
                        "metrics": list(self.metrics.keys()),
                        "frequency": self.frequency,
                    },
                    tensor_path,
                )
                logger.info(f"Saved tensor history to {tensor_path}")
            except Exception as e:
                logger.warning(f"Failed to save tensor history to {tensor_path}: {e}")


def create_alignment_callback(metrics: Dict[str, Any], layers: List[str], **config) -> AlignmentMetricsCallback:
    """
    Factory function for creating alignment callbacks.

    Args:
        metrics: Dictionary of metrics
        layers: Layers to track
        **config: Additional configuration

    Returns:
        Configured AlignmentMetricsCallback
    """
    return AlignmentMetricsCallback(metrics, layers, **config)
