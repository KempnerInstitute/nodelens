"""
General alignment experiment that can perform comprehensive analysis.

This module implements a flexible experiment that can:
- Train models from scratch or use pretrained
- Compute alignment metrics throughout training
- Apply various pruning strategies
- Perform dropout analysis
- Generate comprehensive visualizations
"""

import copy
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from alignment.core.registry import register_experiment
from alignment.dataops.processing import preprocess_layer_activations
from alignment.experiments.base import BaseExperiment, ExperimentConfig
from alignment.metrics.rayleigh.rayleigh_quotient import RayleighQuotient
from alignment.models import ModelWrapper
from alignment.pruning.base import PruningConfig
from alignment.pruning.dependency_aware import DependencyAwarePruning
from alignment.pruning.pipeline import PruningPipelineOptions, run_pruning_pipeline
from alignment.pruning.strategies import MagnitudePruning, RandomPruning
from alignment.services import ActivationCaptureService, MaskOperations

logger = logging.getLogger(__name__)


@dataclass
class GeneralAlignmentConfig(ExperimentConfig):
    """Configuration for general alignment experiment."""

    # Multi-network configuration
    num_networks: int = 1  # Number of networks to train (1 = single network, >1 = parallel)
    parallel_batch_size: Optional[int] = None  # Batch size for parallel training (None = use batch_size)
    use_tensorized_training: bool = True  # Use efficient tensorized ops when possible
    aggregate_metrics: bool = True  # Aggregate metrics across networks
    save_individual_networks: bool = False  # Save each network separately

    # Training configuration
    do_train: bool = True
    training_epochs: int = 100
    learning_rate: float = 0.1
    optimizer: str = "sgd"
    momentum: float = 0.9  # For SGD optimizer
    weight_decay: float = 0.0001  # L2 regularization
    scheduler: str = "cosine"
    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {"T_max": 100, "eta_min": 0})

    # Alignment measurement
    measure_alignment_during_training: bool = True
    alignment_frequency: int = 1  # Measure every N epochs
    alignment_methods: List[str] = field(default_factory=lambda: ["rayleigh_quotient"])
    measure_expected_distribution: bool = True
    distribution_bins: int = 50

    # Progressive dropout analysis
    do_dropout_analysis: bool = True
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"  # "scaled" or "unscaled"
    dropout_pruning_mode: str = "global"  # "global", "per_layer_combined", "per_layer_independent"

    # Pruning experiments
    do_pruning_experiments: bool = True
    pruning_strategies: List[str] = field(default_factory=lambda: ["magnitude", "random", "alignment"])
    pruning_amounts: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    fine_tune_after_pruning: bool = True
    fine_tune_epochs: int = 10
    pruning_selection_mode: str = "low"  # Which weights to prune: "low", "high", "random"
    pruning_alignment_metric: str = "rayleigh_quotient"  # Metric for alignment-based pruning
    pruning_hybrid_alpha: float = 0.5  # Weight for alignment in hybrid pruning (0-1)
    pruning_scope: str = "layer"  # "global" or "layer" - how to select neurons/weights for pruning
    alignment_structured_pruning: bool = True  # Use structured pruning for alignment-based methods

    # Eigenfeature analysis
    do_eigenfeature_analysis: bool = True

    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300

    # Evaluation optimization
    eval_batches: Optional[int] = None  # Limit evaluation to N batches for speed
    use_ultra_parallel_eval: bool = True  # Use ultra-parallel evaluation for pruning
    use_tensorized_pruning: bool = True  # Use tensorized operations for pruning

    # CNN mode
    cnn_mode: str = "unfold"  # "unfold", "patchwise", "batch_patch_combined"

    # Aggregation for layer-wise metrics
    aggregate_alignment: bool = False

    # Results saving
    save_intermediate_results: bool = True
    save_networks: bool = False
    save_checkpoints: bool = False  # Whether to save checkpoints during training


@register_experiment("general_alignment")
class GeneralAlignmentExperiment(BaseExperiment):
    """
    Comprehensive alignment experiment with multiple analysis types.

    REFACTORED (v0.2.0): Now uses services to eliminate redundancy:
    - MaskOperations for mask creation (replaces _create_pruning_mask_tensor logic)
    - preprocess_layer_activations for preprocessing (unified approach)
    - Can optionally use ActivationCaptureService for future enhancements

    NOTE: For new experiments, consider using MasterPruningOrchestrator which
    provides a cleaner API with all modern features.

    This experiment can:
    1. Train networks from scratch or use pretrained
    2. Measure alignment throughout training
    3. Perform progressive dropout analysis
    4. Test various pruning strategies
    5. Analyze eigenfeatures
    6. Generate comprehensive visualizations
    """

    def __init__(self, config: GeneralAlignmentConfig):
        """Initialize general alignment experiment."""
        super().__init__(config)
        self.train_results = {}
        self.test_results = {}
        self.dropout_results = {}
        self.pruning_results = {}
        self.eigenfeature_results = {}

        # Multi-network mode
        self.is_multi_network = config.num_networks > 1
        if self.is_multi_network:
            self.networks = []
            self.wrapped_networks = []
            self._initialize_multiple_networks()
            # Setup parallel processing
            self.num_workers = min(config.num_networks, mp.cpu_count())
            logger.info(f"Multi-network mode: {config.num_networks} networks, {self.num_workers} workers")

    def _initialize_multiple_networks(self):
        """Initialize multiple networks with different seeds."""
        base_seed = self.config.seed

        for i in range(self.config.num_networks):
            # Set unique seed for each network
            seed = base_seed + i
            self._set_seed(seed)

            # Create a new model instance
            # Store original model/wrapped_model temporarily
            original_model = self.model
            original_wrapped = self.wrapped_model

            # Initialize new model (this sets self.model and self.wrapped_model)
            self._initialize_model()

            # Store the new model
            self.networks.append(self.model)
            self.wrapped_networks.append(self.wrapped_model)

            # Restore original references
            self.model = original_model
            self.wrapped_model = original_wrapped

            logger.info(f"Initialized network {i+1}/{self.config.num_networks} with seed {seed}")

        # For multi-network mode, model/wrapped_model will be None
        self.model = None
        self.wrapped_model = None

    def _train_model(self) -> Dict[str, Any]:
        """Train the model and collect alignment metrics."""
        if not self.config.do_train:
            logger.info("Skipping training (do_train=False)")
            return {}

        if self.is_multi_network:
            return self._train_multiple_networks()
        else:
            return self._train_single_network()

    def _train_single_network(self) -> Dict[str, Any]:
        """Train a single network (original implementation)."""
        logger.info(f"Training model for {self.config.training_epochs} epochs")

        # Setup optimizer
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()

        # Training history
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        alignment_history = {method: [] for method in self.config.alignment_methods}

        # Training loop with progress bar
        pbar = tqdm(range(self.config.training_epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            # Train one epoch
            train_loss, train_acc = self._train_epoch(optimizer, criterion)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            val_loss, val_acc = self._evaluate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            # Measure alignment
            if self.config.measure_alignment_during_training and epoch % self.config.alignment_frequency == 0:
                alignment_values = self._measure_alignment()
                for method, values in alignment_values.items():
                    alignment_history[method].append(values)

            # Update progress bar
            pbar.set_postfix(
                loss=f"{train_loss:.3f}",
                acc=f"{train_acc:.1f}%",
                val_acc=f"{val_acc:.1f}%"
            )

            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch, {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        return {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "alignment": alignment_history,
        }

    def _train_multiple_networks(self) -> Dict[str, Any]:
        """Train multiple networks simultaneously (tensorized approach)."""
        logger.info(f"Training {self.config.num_networks} networks for {self.config.training_epochs} epochs")

        # Setup optimizers for all networks
        optimizers = []
        for net in self.networks:
            if self.config.optimizer.lower() == "sgd":
                opt = torch.optim.SGD(net.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            else:
                opt = torch.optim.Adam(net.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            optimizers.append(opt)

        # Setup schedulers
        schedulers = []
        for optimizer in optimizers:
            scheduler = self._setup_scheduler(optimizer)
            schedulers.append(scheduler)

        criterion = nn.CrossEntropyLoss()

        # Training history for all networks
        num_networks = len(self.networks)
        train_losses = torch.zeros((self.config.training_epochs, num_networks))
        train_accs = torch.zeros((self.config.training_epochs, num_networks))
        val_losses = torch.zeros((self.config.training_epochs, num_networks))
        val_accs = torch.zeros((self.config.training_epochs, num_networks))

        alignment_history = {method: [] for method in self.config.alignment_methods}

        # Training loop
        for epoch in range(self.config.training_epochs):
            # Training phase
            epoch_train_losses = []
            epoch_train_accs = []
            num_batches = 0

            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.config.device), target.to(self.config.device)

                # Zero gradients for all optimizers
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # Forward pass through all networks
                outputs = []
                losses = []
                batch_losses = []
                batch_accs = []

                for net in self.networks:
                    output = net(data)
                    outputs.append(output)
                    loss = criterion(output, target)
                    losses.append(loss)

                    # Calculate accuracy
                    pred = output.argmax(dim=1)
                    acc = pred.eq(target).float().mean()
                    batch_losses.append(loss.item())
                    batch_accs.append(acc.item())

                # Backward pass for all networks
                for loss, optimizer in zip(losses, optimizers):
                    loss.backward()
                    optimizer.step()

                epoch_train_losses.append(batch_losses)
                epoch_train_accs.append(batch_accs)
                num_batches += 1

            # Average over batches for this epoch
            epoch_avg_losses = torch.tensor(epoch_train_losses).mean(dim=0)
            epoch_avg_accs = torch.tensor(epoch_train_accs).mean(dim=0)
            train_losses[epoch] = epoch_avg_losses
            train_accs[epoch] = epoch_avg_accs

            # Validation phase
            with torch.no_grad():
                val_batch_losses = []
                val_batch_accs = []

                for data, target in self.data_loader:  # Using train loader as validation
                    data, target = data.to(self.config.device), target.to(self.config.device)

                    batch_val_losses = []
                    batch_val_accs = []

                    for net in self.networks:
                        net.eval()
                        output = net(data)
                        loss = criterion(output, target)
                        pred = output.argmax(dim=1)
                        acc = pred.eq(target).float().mean()

                        batch_val_losses.append(loss.item())
                        batch_val_accs.append(acc.item())
                        net.train()

                    val_batch_losses.append(batch_val_losses)
                    val_batch_accs.append(batch_val_accs)

                epoch_val_losses = torch.tensor(val_batch_losses).mean(dim=0)
                epoch_val_accs = torch.tensor(val_batch_accs).mean(dim=0)
                val_losses[epoch] = epoch_val_losses
                val_accs[epoch] = epoch_val_accs

            # Update schedulers
            for scheduler in schedulers:
                if scheduler is not None:
                    scheduler.step()

            # Measure alignment if configured
            if self.config.measure_alignment_during_training and epoch % self.config.alignment_frequency == 0:
                epoch_alignment = {}
                for method in self.config.alignment_methods:
                    method_values = {}

                    # Get a batch for alignment measurement
                    data, _ = next(iter(self.data_loader))
                    data = data.to(self.config.device)

                    # Measure alignment for each network
                    for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
                        # Set as current model temporarily
                        old_model = self.model
                        old_wrapped = self.wrapped_model
                        self.model = net
                        self.wrapped_model = wrapped_net

                        # Measure alignment
                        alignment_values = self._measure_alignment()

                        # Accumulate values by layer
                        for layer_name, layer_values in alignment_values.get(method, {}).items():
                            if layer_name not in method_values:
                                method_values[layer_name] = []
                            method_values[layer_name].extend(layer_values)

                        # Restore
                        self.model = old_model
                        self.wrapped_model = old_wrapped

                    if method_values:
                        epoch_alignment[method] = method_values

                if epoch_alignment:
                    alignment_history.update(epoch_alignment)

            # Log progress
            avg_train_loss = epoch_avg_losses.mean().item()
            avg_train_acc = epoch_avg_accs.mean().item() * 100
            avg_val_loss = epoch_val_losses.mean().item()
            avg_val_acc = epoch_val_accs.mean().item() * 100

            logger.info(
                f"Epoch {epoch+1}/{self.config.training_epochs}: "
                f"Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.2f}%, "
                f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.2f}%"
            )

            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                # In multi-network mode, we can't save a single model checkpoint
                # Instead, we either skip or save all networks
                if self.config.save_checkpoints:
                    if self.config.save_individual_networks:
                        # Save all networks individually
                        for i, net in enumerate(self.networks):
                            # Temporarily set model for checkpoint saving
                            old_model = self.model
                            self.model = net
                            self.save_checkpoint(
                                epoch,
                                {
                                    "train_loss": avg_train_loss,
                                    "train_acc": avg_train_acc,
                                    "val_loss": avg_val_loss,
                                    "val_acc": avg_val_acc,
                                    "network_id": i,
                                },
                            )
                            self.model = old_model
                    else:
                        # Save just the first network as representative
                        old_model = self.model
                        self.model = self.networks[0]
                        self.save_checkpoint(
                            epoch,
                            {
                                "train_loss": avg_train_loss,
                                "train_acc": avg_train_acc,
                                "val_loss": avg_val_loss,
                                "val_acc": avg_val_acc,
                                "note": "representative_network_from_ensemble",
                            },
                        )
                        self.model = old_model

        # Aggregate results if configured
        if self.config.aggregate_metrics:
            # Average across networks
            return {
                "train_losses": train_losses.mean(dim=1).tolist(),
                "train_accs": train_accs.mean(dim=1).tolist(),
                "val_losses": val_losses.mean(dim=1).tolist(),
                "val_accs": val_accs.mean(dim=1).tolist(),
                "alignment": alignment_history,
            }
        else:
            # Return individual network results
            return {
                "networks": [
                    {
                        "train_losses": train_losses[:, i].tolist(),
                        "train_accs": train_accs[:, i].tolist(),
                        "val_losses": val_losses[:, i].tolist(),
                        "val_accs": val_accs[:, i].tolist(),
                    }
                    for i in range(num_networks)
                ],
                "alignment": alignment_history,
            }

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on config."""
        if self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _setup_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """Setup learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.config.scheduler_config)
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return None

    def _train_epoch(self, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.data_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _preprocess_pruning_inputs(self, layer_inputs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocess layer inputs for pruning, handling CNN unfold for proper RQ computation.
        
        For Conv2d layers, unfolds the input tensor to match weight dimensions:
        [B, C, H, W] -> [B*num_patches, C*kH*kW]
        
        Args:
            layer_inputs_dict: Dictionary mapping layer names to input tensors
            
        Returns:
            Preprocessed inputs dictionary
        """
        processed = {}
        for name, tensor in layer_inputs_dict.items():
            if tensor.ndim == 4:  # Conv2d input: [B, C, H, W]
                # Get the layer module to access kernel parameters
                try:
                    module = dict(self.model.named_modules()).get(name)
                    if module is not None and isinstance(module, torch.nn.Conv2d):
                        # Proper unfold using layer's kernel parameters
                        unfolded = torch.nn.functional.unfold(
                            tensor,
                            kernel_size=module.kernel_size,
                            dilation=module.dilation,
                            padding=module.padding,
                            stride=module.stride,
                        )
                        # [B, C*kH*kW, num_patches] -> [B*num_patches, C*kH*kW]
                        unfolded = unfolded.transpose(1, 2).contiguous()
                        processed[name] = unfolded.view(-1, unfolded.size(2))
                    else:
                        # Fallback: flatten
                        processed[name] = tensor.reshape(tensor.shape[0], -1)
                except Exception as e:
                    logger.debug(f"Unfold failed for {name}: {e}, using flatten")
                    processed[name] = tensor.reshape(tensor.shape[0], -1)
            elif tensor.ndim == 3:  # Conv1d input: [B, C, L]
                # Get the layer module
                try:
                    module = dict(self.model.named_modules()).get(name)
                    if module is not None and isinstance(module, torch.nn.Conv1d):
                        # Unfold for Conv1d
                        unfolded = tensor.unfold(2, module.kernel_size[0], module.stride[0])
                        # [B, C, num_patches, kW] -> [B*num_patches, C*kW]
                        B, C, num_patches, kW = unfolded.shape
                        processed[name] = unfolded.permute(0, 2, 1, 3).reshape(-1, C * kW)
                    else:
                        processed[name] = tensor.reshape(tensor.shape[0], -1)
                except Exception:
                    processed[name] = tensor.reshape(tensor.shape[0], -1)
            else:
                # Linear layers: already 2D or flatten
                if tensor.ndim > 2:
                    processed[name] = tensor.reshape(tensor.shape[0], -1)
                else:
                    processed[name] = tensor
        return processed

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation/test set."""
        if self.is_multi_network:
            # Multi-network evaluation: average across all networks
            return self._evaluate_multi_networks()
        else:
            # Single network evaluation
            return self._evaluate_single_network()

    def _evaluate_single_network(self) -> Tuple[float, float]:
        """Evaluate a single network."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        criterion = nn.CrossEntropyLoss()
        eval_batches = self.config.eval_batches  # Limit evaluation batches (None = all)
        
        # Determine total batches for progress bar
        total_batches = eval_batches if eval_batches is not None else len(self.data_loader)

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.data_loader),
                total=total_batches,
                desc="Evaluating",
                leave=False,
                disable=not logger.isEnabledFor(logging.INFO),
            )
            for batch_idx, (inputs, targets) in pbar:
                if eval_batches is not None and batch_idx >= eval_batches:
                    break
                    
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                num_batches += 1
                
                # Update progress bar with current stats
                pbar.set_postfix(loss=total_loss / num_batches, acc=f"{100.0 * correct / total:.1f}%")

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = 100.0 * correct / max(total, 1)
        
        logger.info(f"Eval: loss={avg_loss:.4f}, acc={accuracy:.2f}%")

        return avg_loss, accuracy

    def _evaluate_multi_networks(self) -> Tuple[float, float]:
        """Evaluate multiple networks and return averaged results."""
        all_losses = []
        all_accuracies = []

        criterion = nn.CrossEntropyLoss()
        eval_batches = self.config.eval_batches  # Limit evaluation batches (None = all)
        total_batches = eval_batches if eval_batches is not None else len(self.data_loader)

        # Set all networks to eval mode
        for net in self.networks:
            net.eval()

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.data_loader),
                total=total_batches,
                desc=f"Evaluating {len(self.networks)} networks",
                leave=False,
            )
            for batch_idx, (inputs, targets) in pbar:
                if eval_batches is not None and batch_idx >= eval_batches:
                    break
                    
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

                batch_losses = []
                batch_accuracies = []

                # Evaluate each network on this batch
                for net in self.networks:
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    accuracy = predicted.eq(targets).float().mean().item() * 100

                    batch_losses.append(loss.item())
                    batch_accuracies.append(accuracy)

                # Store batch averages
                all_losses.append(np.mean(batch_losses))
                all_accuracies.append(np.mean(batch_accuracies))
                
                pbar.set_postfix(loss=np.mean(all_losses), acc=f"{np.mean(all_accuracies):.1f}%")

        # Set networks back to train mode
        for net in self.networks:
            net.train()

        # Return overall averages
        avg_loss = np.mean(all_losses)
        avg_accuracy = np.mean(all_accuracies)

        return avg_loss, avg_accuracy

    def _measure_alignment(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Measure alignment metrics for all layers.

        REFACTORED (v0.2.0): Now uses ActivationCaptureService for cleaner code.
        """
        if self.is_multi_network:
            # Use first network as representative for alignment measurement
            model_to_use = self.networks[0]
            wrapped_model_to_use = self.wrapped_networks[0]
        else:
            model_to_use = self.model
            wrapped_model_to_use = self.wrapped_model

        analysis_opts = getattr(self.config, "analysis_options", {}) or {}
        save_scores = analysis_opts.get("save_scores", True)
        class_conditioned = analysis_opts.get("class_conditioned", False)

        alignment_values = {}

        # Get a batch of data (inputs and targets)
        inputs, targets = next(iter(self.data_loader))
        inputs = inputs.to(self.config.device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.config.device)

        # REFACTORED: Use ActivationCaptureService (eliminates redundancy)
        try:
            capture_service = ActivationCaptureService(wrapped_model_to_use, default_mode=self.config.cnn_mode)

            # Capture and preprocess in one call
            activation_data = capture_service.capture(
                inputs,
                layers=wrapped_model_to_use.tracked_layers,
                include_weights=True,
                preprocess=True,
            )

            # Use captured data
            preprocessed_inputs = activation_data.inputs
            preprocessed_outputs = activation_data.outputs
            weights = activation_data.weights

        except Exception as e:
            # Fallback to manual approach if service fails
            logger.warning(f"ActivationCaptureService failed ({e}), using manual capture")

            # Manual capture (legacy)
            _, activations = wrapped_model_to_use.forward_with_activations(inputs)
            weights = wrapped_model_to_use.get_layer_weights()

            # Manual preprocessing
            from alignment.dataops.processing import preprocess_layer_activations

            layer_modules = dict(wrapped_model_to_use._model.named_modules())

            inputs_to_process = {
                f"{layer_name}_input": activations.get(f"{layer_name}_input")
                for layer_name in wrapped_model_to_use.tracked_layers
                if activations.get(f"{layer_name}_input") is not None
            }
            outputs_to_process = {
                f"{layer_name}_output": activations.get(f"{layer_name}_output")
                for layer_name in wrapped_model_to_use.tracked_layers
                if activations.get(f"{layer_name}_output") is not None
            }

            preprocessed_inputs_raw = preprocess_layer_activations(
                inputs_to_process, layer_modules, mode=self.config.cnn_mode
            )
            preprocessed_outputs_raw = preprocess_layer_activations(
                outputs_to_process, layer_modules, mode=self.config.cnn_mode
            )

            preprocessed_inputs = {
                layer_name: preprocessed_inputs_raw[f"{layer_name}_input"]
                for layer_name in wrapped_model_to_use.tracked_layers
                if f"{layer_name}_input" in preprocessed_inputs_raw
            }
            preprocessed_outputs = {
                layer_name: preprocessed_outputs_raw[f"{layer_name}_output"]
                for layer_name in wrapped_model_to_use.tracked_layers
                if f"{layer_name}_output" in preprocessed_outputs_raw
            }

        # Determine a target representation for synergy-style metrics (e.g., final layer outputs)
        target_outputs = None
        if wrapped_model_to_use.tracked_layers:
            target_layer = wrapped_model_to_use.tracked_layers[-1]
            if target_layer in preprocessed_outputs:
                target_outputs = preprocessed_outputs[target_layer]

        # Compute each metric
        for method in tqdm(self.config.alignment_methods, desc="Computing metrics", leave=False):
            if method not in self.metrics:
                logger.warning(f"Metric {method} not initialized, skipping")
                continue

            metric = self.metrics[method]
            layer_values = {}

            for layer_name in wrapped_model_to_use.tracked_layers:
                if layer_name not in weights:
                    continue

                metric_kwargs = {}

                # Supply inputs/weights/outputs based on metric requirements
                if getattr(metric, "requires_inputs", False):
                    if layer_name not in preprocessed_inputs:
                        continue
                    metric_kwargs["inputs"] = preprocessed_inputs[layer_name]

                if getattr(metric, "requires_weights", False):
                    metric_kwargs["weights"] = weights[layer_name]

                if getattr(metric, "requires_outputs", False):
                    if layer_name not in preprocessed_outputs:
                        continue
                    metric_kwargs["outputs"] = preprocessed_outputs[layer_name]

                # Synergy / redundancy metrics with Gaussian approximations
                if method == "gaussian_pid_synergy_mmi":
                    # Uses layer outputs and a continuous target representation (e.g., final logits)
                    if target_outputs is None:
                        logger.warning(
                            "gaussian_pid_synergy_mmi: target_outputs not available; skipping "
                            f"layer {layer_name}"
                        )
                        continue
                    metric_kwargs["target_outputs"] = target_outputs

                if method == "synergy_gaussian_mmi":
                    # Uses discrete class labels
                    if targets is None:
                        logger.warning(
                            "synergy_gaussian_mmi: targets not available; skipping "
                            f"layer {layer_name}"
                        )
                        continue
                    metric_kwargs["targets"] = targets
                
                # Conditional metrics that need class labels
                if method in ["conditional_rayleigh_quotient", "mi_about_class", 
                              "conditional_mi", "conditional_activation_norm", "delta_rq"]:
                    if targets is not None:
                        metric_kwargs["targets"] = targets

                try:
                    scores = metric.compute(**metric_kwargs)
                    scores_cpu = scores.detach().cpu()
                    if save_scores:
                        layer_values[layer_name] = scores_cpu.tolist()
                    else:
                        layer_values[layer_name] = float(scores_cpu.mean().item())
                except Exception as e:
                    logger.error(f"Error computing {method} for {layer_name}: {e}")

            alignment_values[method] = layer_values

        if class_conditioned:
            rq_metric = self.metrics.get("rayleigh_quotient")
            if rq_metric is None:
                logger.warning("Class-conditioned analysis requested, but 'rayleigh_quotient' metric is not initialized.")
            elif targets is None:
                logger.warning("Class-conditioned analysis requested, but no targets were provided.")
            else:
                cc_values = {}
                delta_values = {}
                for layer_name in wrapped_model_to_use.tracked_layers:
                    if layer_name not in weights or layer_name not in preprocessed_inputs:
                        continue
                    try:
                        cc_result = rq_metric.compute_class_conditioned(
                            inputs=preprocessed_inputs[layer_name],
                            weights=weights[layer_name],
                            targets=targets,
                            return_delta_rq=True,
                        )
                        cond_scores = cc_result["rq_cond"].detach().cpu()
                        delta_scores = cc_result["delta_rq"].detach().cpu()

                        if save_scores:
                            cc_values[layer_name] = cond_scores.tolist()
                            delta_values[layer_name] = delta_scores.tolist()
                        else:
                            cc_values[layer_name] = float(cond_scores.mean().item())
                            delta_values[layer_name] = float(delta_scores.mean().item())
                    except Exception as e:
                        logger.error(f"Error computing class-conditioned RQ for {layer_name}: {e}")

                if cc_values:
                    alignment_values["rayleigh_quotient_class_conditioned"] = cc_values
                if delta_values:
                    alignment_values["rayleigh_quotient_delta"] = delta_values

        return alignment_values

    def _compute_redundancy_matrices(self) -> Dict[str, Any]:
        """
        Compute pairwise redundancy matrices for visualization.
        
        Returns:
            Dictionary mapping layer names to redundancy matrices [N, N]
        """
        # Check if redundancy metric is available
        if "pairwise_redundancy_gaussian" not in self.metrics and "average_redundancy" not in self.metrics:
            logger.debug("No redundancy metrics configured, skipping redundancy matrix computation")
            return {}
        
        try:
            from alignment.metrics.information.pairwise_gaussian import PairwiseRedundancyGaussian
        except ImportError:
            logger.warning("PairwiseRedundancyGaussian not available, skipping redundancy matrices")
            return {}
        
        if self.is_multi_network:
            wrapped_model_to_use = self.wrapped_networks[0]
        else:
            wrapped_model_to_use = self.wrapped_model
        
        if wrapped_model_to_use is None:
            return {}
        
        redundancy_matrices = {}
        
        try:
            # Get a batch of data
            inputs, _ = next(iter(self.data_loader))
            inputs = inputs.to(self.config.device)
            
            # Capture activations
            capture_service = ActivationCaptureService(wrapped_model_to_use, default_mode=self.config.cnn_mode)
            activation_data = capture_service.capture(
                inputs,
                layers=wrapped_model_to_use.tracked_layers,
                include_weights=True,
                preprocess=True,
            )
            
            # Create redundancy metric for matrix computation
            redundancy_metric = PairwiseRedundancyGaussian(
                sampling_strategy="all",  # Compute full matrix
                mode="output_based",  # Faster computation using outputs
            )
            
            # Compute redundancy matrices for each layer
            for layer_name in wrapped_model_to_use.tracked_layers:
                layer_inputs = activation_data.inputs.get(layer_name)
                layer_weights = activation_data.weights.get(layer_name)
                layer_outputs = activation_data.outputs.get(layer_name)
                
                if layer_inputs is None or layer_weights is None:
                    continue
                
                try:
                    # Use compute_pairwise_matrix for full redundancy matrix
                    # Flatten weights for linear algebra
                    if layer_weights.ndim > 2:
                        layer_weights = layer_weights.reshape(layer_weights.shape[0], -1)
                    if layer_inputs.ndim > 2:
                        layer_inputs = layer_inputs.reshape(layer_inputs.shape[0], -1)
                    
                    # Limit matrix size for performance (max 256 neurons)
                    num_neurons = layer_weights.shape[0]
                    if num_neurons > 256:
                        logger.debug(f"Skipping redundancy matrix for {layer_name} ({num_neurons} neurons > 256)")
                        continue
                    
                    matrix = redundancy_metric.compute_pairwise_matrix(layer_inputs, layer_weights)
                    redundancy_matrices[layer_name] = matrix.cpu().numpy()
                    logger.debug(f"Computed redundancy matrix for {layer_name}: shape {matrix.shape}")
                    
                except Exception as e:
                    logger.warning(f"Could not compute redundancy matrix for {layer_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Redundancy matrix computation failed: {e}")
        
        return redundancy_matrices

    def _run_eigenfeature_analysis(self) -> Dict[str, Any]:
        """
        Compute eigenfeature statistics (top eigenvalues / explained variance) for each tracked layer.
        """
        if not getattr(self.config, "do_eigenfeature_analysis", False):
            return {}

        if self.data_loader is None:
            logger.warning("Eigenfeature analysis skipped (no data loader available).")
            return {}

        if self.is_multi_network and self.networks and self.wrapped_networks:
            model_to_use = self.networks[0]
            wrapped_model_to_use = self.wrapped_networks[0]
        else:
            model_to_use = self.model
            wrapped_model_to_use = self.wrapped_model

        if model_to_use is None or wrapped_model_to_use is None:
            logger.warning("Eigenfeature analysis skipped (model not initialized).")
            return {}

        try:
            inputs, _ = next(iter(self.data_loader))
        except StopIteration:
            logger.warning("Eigenfeature analysis skipped (empty data loader).")
            return {}

        inputs = inputs.to(self.config.device)
        eigenfeature_results: Dict[str, Any] = {}

        try:
            capture_service = ActivationCaptureService(wrapped_model_to_use, default_mode=self.config.cnn_mode)
            activation_data = capture_service.capture(inputs, layers=wrapped_model_to_use.tracked_layers, include_weights=False, preprocess=True)
        except Exception as e:
            logger.error(f"Activation capture failed during eigenfeature analysis: {e}")
            return {}

        for layer_name, layer_inputs in activation_data.inputs.items():
            if layer_inputs is None:
                continue
            data = layer_inputs
            if data.dim() > 2:
                data = data.reshape(data.size(0), -1)
            if data.size(0) < 2 or data.size(1) == 0:
                continue

            centered = data - data.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)

            try:
                eigvals, eigvecs = torch.linalg.eigh(cov)
            except RuntimeError as err:
                logger.warning(f"Eigenvalue decomposition failed for layer {layer_name}: {err}")
                continue

            order = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            total_var = torch.clamp(eigvals.sum(), min=1e-12)
            cum_var = torch.cumsum(eigvals, dim=0) / total_var
            top_k = min(10, eigvals.numel())

            eigenfeature_results[layer_name] = {
                "top_eigenvalues": eigvals[:top_k].detach().cpu().tolist(),
                "explained_variance": cum_var[:top_k].detach().cpu().tolist(),
            }

        return eigenfeature_results

    def _dropout_analysis(self) -> Dict[str, Any]:
        """Perform progressive dropout analysis."""
        if not self.config.do_dropout_analysis:
            logger.info("Skipping dropout analysis")
            return {}

        logger.info("Starting progressive dropout analysis")

        if self.is_multi_network:
            return self._dropout_analysis_multi()
        else:
            return self._dropout_analysis_single()

    def _dropout_analysis_single(self) -> Dict[str, Any]:
        """Perform dropout analysis on a single network."""
        # Get initial alignment
        initial_alignment = self._measure_alignment()

        # Results storage
        results = {
            "dropout_rates": self.config.dropout_rates,
            "accuracies": {"low": [], "high": [], "random": []},
            "losses": {"low": [], "high": [], "random": []},
            "alignment_values": {},
        }

        # Test each dropout rate
        for dropout_rate in self.config.dropout_rates:
            logger.info(f"Testing dropout rate: {dropout_rate}")

            for strategy in ["low", "high", "random"]:
                # Apply targeted dropout based on alignment scores
                if strategy == "random":
                    # Average over multiple trials
                    trial_losses = []
                    trial_accs = []

                    for _ in range(3):
                        loss, acc = self._apply_dropout_and_evaluate(dropout_rate, strategy, initial_alignment)
                        trial_losses.append(loss)
                        trial_accs.append(acc)

                    avg_loss = np.mean(trial_losses)
                    avg_acc = np.mean(trial_accs)
                else:
                    avg_loss, avg_acc = self._apply_dropout_and_evaluate(dropout_rate, strategy, initial_alignment)

                results["losses"][strategy].append(avg_loss)
                results["accuracies"][strategy].append(avg_acc)

                logger.info(f"  {strategy}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")

        return results

    def _dropout_analysis_multi(self) -> Dict[str, Any]:
        """Perform dropout analysis on multiple networks."""
        logger.info("Running tensorized dropout analysis on all networks")
        # Create tensorized dropout analysis
        results = self._tensorized_dropout_analysis()

        if self.config.aggregate_metrics:
            return results
        else:
            return {"networks": results}

    def _tensorized_dropout_analysis(self) -> Dict[str, Any]:
        """Tensorized dropout analysis for multiple networks and dropout rates."""
        num_networks = len(self.networks)
        num_rates = len(self.config.dropout_rates)

        # Initialize result tensors
        accuracies = torch.zeros(num_networks, num_rates, 3)  # 3 for low, high, random
        losses = torch.zeros(num_networks, num_rates, 3)

        # Get alignment for all networks
        alignment_results = {}
        for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
            # Set as current model temporarily
            old_model = self.model
            old_wrapped = self.wrapped_model
            self.model = net
            self.wrapped_model = wrapped_net

            alignment_results[i] = self._measure_alignment()

            # Restore
            self.model = old_model
            self.wrapped_model = old_wrapped

        # Process all dropout rates simultaneously
        for rate_idx, dropout_rate in enumerate(self.config.dropout_rates):
            logger.info(f"Processing dropout rate: {dropout_rate * 100:.0f}%")

            # Apply dropout to all networks for each strategy
            for strategy_idx, strategy in enumerate(["low", "high", "random"]):
                # Apply dropout to all networks
                for net_idx, net in enumerate(self.networks):
                    self._apply_tensorized_dropout(net, alignment_results[net_idx], dropout_rate, strategy)

                # Evaluate all networks simultaneously
                batch_losses, batch_accs = self._evaluate_networks_batch()
                accuracies[:, rate_idx, strategy_idx] = batch_accs
                losses[:, rate_idx, strategy_idx] = batch_losses

                # Restore original weights
                for net in self.networks:
                    self._restore_original_weights(net)

        # Average across networks if configured
        if self.config.aggregate_metrics:
            return {
                "dropout_rates": self.config.dropout_rates,
                "accuracies_low": accuracies[:, :, 0].mean(dim=0).tolist(),
                "accuracies_high": accuracies[:, :, 1].mean(dim=0).tolist(),
                "accuracies_random": accuracies[:, :, 2].mean(dim=0).tolist(),
                "losses_low": losses[:, :, 0].mean(dim=0).tolist(),
                "losses_high": losses[:, :, 1].mean(dim=0).tolist(),
                "losses_random": losses[:, :, 2].mean(dim=0).tolist(),
            }
        else:
            return {
                "dropout_rates": self.config.dropout_rates,
                "accuracies_low": accuracies[:, :, 0].tolist(),
                "accuracies_high": accuracies[:, :, 1].tolist(),
                "accuracies_random": accuracies[:, :, 2].tolist(),
                "losses_low": losses[:, :, 0].tolist(),
                "losses_high": losses[:, :, 1].tolist(),
                "losses_random": losses[:, :, 2].tolist(),
            }

    def _apply_tensorized_dropout(self, model: nn.Module, alignment_values: Dict[str, Dict[str, List[float]]], dropout_rate: float, strategy: str):
        """Apply dropout to a model based on alignment values."""
        # Set a unique random seed for this specific combination
        base_seed = getattr(self.config, "seed", 42)
        strategy_seed = base_seed + hash(strategy) % 1000

        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                # Store original weight if not already stored
                if not hasattr(module, "_original_weight"):
                    module.register_buffer("_original_weight", module.weight.data.clone())

                # Get layer weight information
                weight_shape = module.weight.shape
                num_neurons = weight_shape[0]  # Assuming first dimension is output neurons
                num_to_drop = int(dropout_rate * num_neurons)

                if num_to_drop > 0 and num_to_drop < num_neurons:
                    if strategy == "random":
                        # Use network and layer specific seed for random dropout
                        torch.manual_seed(strategy_seed + hash(name) % 1000)
                        indices = torch.randperm(num_neurons)[:num_to_drop]

                    else:  # "low" or "high" - use alignment values
                        # Get alignment values for this layer
                        layer_alignment = alignment_values.get("rayleigh_quotient", {}).get(name, [])

                        if layer_alignment and len(layer_alignment) >= num_neurons:
                            # Convert alignment to tensor and ensure proper length
                            alignment_tensor = torch.tensor(layer_alignment[:num_neurons], dtype=torch.float32)

                            # Add small random noise to break ties and ensure diversity across networks
                            torch.manual_seed(strategy_seed + hash(name) % 1000)
                            noise = torch.randn_like(alignment_tensor) * 1e-6
                            alignment_tensor = alignment_tensor + noise

                            # Create mask based on strategy
                            if strategy == "low":
                                # Drop neurons with LOW alignment (keep high alignment)
                                _, indices = alignment_tensor.topk(num_to_drop, largest=False)
                            elif strategy == "high":
                                # Drop neurons with HIGH alignment (keep low alignment)
                                _, indices = alignment_tensor.topk(num_to_drop, largest=True)
                            else:
                                # This shouldn't happen, but fallback to random
                                indices = torch.randperm(num_neurons)[:num_to_drop]
                        else:
                            # Fallback to magnitude-based selection if no alignment values
                            logger.warning(f"No alignment values for layer {name}, using magnitude-based dropout")
                            weight_magnitudes = module.weight.abs().sum(dim=1)  # Sum across input dimensions

                            if strategy == "low":
                                _, indices = weight_magnitudes.topk(num_to_drop, largest=False)
                            elif strategy == "high":
                                _, indices = weight_magnitudes.topk(num_to_drop, largest=True)
                            else:
                                indices = torch.randperm(num_neurons)[:num_to_drop]

                    # Apply dropout by zeroing selected neurons
                    module.weight.data[indices] = 0

                    # Also zero biases if they exist and create bias mask
                    if hasattr(module, "bias") and module.bias is not None:
                        if not hasattr(module, "_original_bias"):
                            module.register_buffer("_original_bias", module.bias.data.clone())

                        module.bias.data[indices] = 0

                        # Create bias mask for consistency
                        bias_mask = torch.ones_like(module.bias)
                        bias_mask[indices] = 0
                        module.register_buffer("bias_mask", bias_mask)

    def _restore_original_weights(self, model: nn.Module):
        """Restore original weights and biases after dropout."""
        for name, module in model.named_modules():
            if hasattr(module, "_original_weight"):
                module.weight.data = module._original_weight.clone()
                delattr(module, "_original_weight")

            if hasattr(module, "_original_bias"):
                module.bias.data = module._original_bias.clone()
                delattr(module, "_original_bias")

            # Clean up masks
            if hasattr(module, "bias_mask"):
                delattr(module, "bias_mask")

    def _pruning_experiments(self) -> Dict[str, Any]:
        """Perform pruning experiments with various strategies."""
        if not self.config.do_pruning_experiments:
            logger.info("Skipping pruning experiments")
            return {}

        logger.info("Starting pruning experiments")

        if self.is_multi_network:
            return self._pruning_experiments_multi()
        else:
            return self._pruning_experiments_single()

    def _pruning_experiments_single(self) -> Dict[str, Any]:
        """Perform pruning experiments on a single network."""
        # Import pruning utilities
        # Imports moved to top of file

        results = {"strategies": {}, "final_model_performance": {}}

        # Save original model state
        # Deep copy the state dict to avoid reference issues when pruning modifies weights in place
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Get selection modes to test (convert single value to list for consistency)
        selection_modes = self.config.pruning_selection_mode
        if not isinstance(selection_modes, list):
            selection_modes = [selection_modes]

        dependency_aware_enabled = bool(getattr(self.config, "dependency_aware_pruning", False)) and self.config.pruning_scope == "layer"

        for strategy_name in self.config.pruning_strategies:
            logger.info(f"Testing pruning strategy: {strategy_name}")

            # If we have multiple selection modes, test each one
            for selection_mode in selection_modes:
                # Create a key that includes selection mode if testing multiple
                if len(selection_modes) > 1:
                    result_key = f"{strategy_name}_{selection_mode}"
                else:
                    result_key = strategy_name

                strategy_results = {
                    "pruning_amounts": [],
                    "accuracies_before_finetune": [],
                    "losses_before_finetune": [],
                    "accuracies_after_finetune": [],
                    "losses_after_finetune": [],
                    "sparsities": [],
                    "weight_distributions_before": [],
                    "weight_distributions_after": [],
                }

                pbar_amounts = tqdm(
                    self.config.pruning_amounts,
                    desc=f"  {strategy_name}/{selection_mode}",
                    leave=False,
                )
                for amount in pbar_amounts:
                    pbar_amounts.set_postfix(sparsity=f"{amount*100:.0f}%")

                    # Remove any existing pruning masks before resetting
                    for name, module in self.model.named_modules():
                        if hasattr(module, "weight_mask"):
                            delattr(module, "weight_mask")
                        if hasattr(module, "_pruning_hook"):
                            module._pruning_hook.remove()
                            delattr(module, "_pruning_hook")
                        if hasattr(module, "_original_weight"):
                            delattr(module, "_original_weight")

                    # Reset model to original state
                    self.model.load_state_dict(original_state)

                    # Create pruning configuration
                    pruning_config = PruningConfig(
                        amount=amount,
                        pruning_mode=selection_mode,
                        structured=False,  # We handle structured pruning differently for alignment
                        global_pruning=(self.config.pruning_scope == "global"),
                    )

                    if dependency_aware_enabled and not pruning_config.structured:
                        pruning_config.structured = True

                    # For metric-based pruning, override structured setting
                    if self.config.alignment_structured_pruning:
                        pruning_config.structured = True

                    # Import additional strategies if needed
                    strategy = None
                    
                    # Known metric names that can be used for pruning
                    metric_based_strategies = [
                        # Alignment metrics
                        "rayleigh_quotient", "conditional_rayleigh_quotient", "delta_rq",
                        # Activation statistics
                        "activation_l2_norm", "activation_l1_norm", "activation_variance",
                        # Redundancy metrics  
                        "pairwise_redundancy_gaussian", "average_redundancy",
                        # Information-theoretic metrics
                        "mutual_information_gaussian", "gaussian_mi_analytic", "mi_about_class",
                        "synergy_gaussian_mmi",
                        # Composite scores
                        "composite_importance", "alignment_minus_redundancy", "class_informative_score",
                        # Gradient-based
                        "taylor_importance", "taylor_saliency",
                    ]

                    if strategy_name == "magnitude":
                        if pruning_config.global_pruning:
                            from alignment.pruning.strategies import GlobalMagnitudePruning

                            strategy = GlobalMagnitudePruning(config=pruning_config)
                        else:
                            strategy = MagnitudePruning(config=pruning_config)
                    elif strategy_name == "alignment":
                        # Legacy "alignment" keyword - use pruning_alignment_metric
                        from alignment.pruning.strategies import AlignmentPruning, CascadingAlignmentPruning, GlobalAlignmentPruning

                        alignment_metric = getattr(self.config, "pruning_alignment_metric", "rayleigh_quotient")

                        if self.config.pruning_scope == "global":
                            strategy = GlobalAlignmentPruning(metric=alignment_metric, config=pruning_config)
                        elif self.config.pruning_scope == "cascading":
                            pruning_config.structured = True
                            strategy = CascadingAlignmentPruning(
                                metric=alignment_metric, direction=getattr(self.config, "cascading_direction", "forward"), config=pruning_config
                            )
                        else:
                            strategy = AlignmentPruning(metric=alignment_metric, config=pruning_config)
                    elif strategy_name in metric_based_strategies:
                        # Use metric name directly as pruning criterion.
                        # Note: in 'cascading' scope we perform the sequential recomputation in this
                        # experiment loop, so we use the standard AlignmentPruning wrapper (which
                        # forwards outputs/targets kwargs to the metric implementation).
                        from alignment.pruning.strategies import AlignmentPruning, GlobalAlignmentPruning
                        metric_kwargs = {}
                        try:
                            metric_kwargs = (getattr(self.config, "metric_configs", {}) or {}).get(strategy_name, {}) or {}
                        except Exception:
                            metric_kwargs = {}

                        if self.config.pruning_scope == "global":
                            strategy = GlobalAlignmentPruning(metric=strategy_name, config=pruning_config, **metric_kwargs)
                        else:
                            if self.config.pruning_scope == "cascading":
                                pruning_config.structured = True
                            strategy = AlignmentPruning(metric=strategy_name, config=pruning_config, **metric_kwargs)
                    elif strategy_name == "cascading_alignment":
                        # Legacy cascading_alignment handling
                        logger.warning("'cascading_alignment' algorithm is deprecated. Use algorithms=['alignment'] with scope='cascading'")
                        from alignment.pruning.strategies import CascadingAlignmentPruning

                        alignment_metric = getattr(self.config, "pruning_alignment_metric", "rayleigh_quotient")
                        pruning_config.structured = True
                        strategy = CascadingAlignmentPruning(metric=alignment_metric, direction="forward", config=pruning_config)
                    elif strategy_name == "hybrid":
                        from alignment.pruning.strategies import HybridPruning

                        alignment_metric = getattr(self.config, "pruning_alignment_metric", "rayleigh_quotient")
                        alpha = getattr(self.config, "pruning_hybrid_alpha", 0.5)
                        strategy = HybridPruning(alignment_metric=alignment_metric, alpha=alpha, config=pruning_config)
                    elif strategy_name == "gradient":
                        logger.warning("Gradient pruning is not suitable for post-training pruning on converged models")
                        from alignment.pruning.strategies import GradientPruning

                        strategy = GradientPruning(config=pruning_config)
                    elif strategy_name == "fisher":
                        logger.warning("Fisher pruning is not suitable for post-training pruning on converged models")
                        from alignment.pruning.strategies import FisherPruning

                        strategy = FisherPruning(config=pruning_config)
                    elif strategy_name == "random":
                        from alignment.pruning.strategies import RandomPruning

                        strategy = RandomPruning(config=pruning_config)
                    else:
                        logger.warning(f"Unsupported pruning strategy: {strategy_name}")
                        continue

                    # Inputs/outputs/targets used by metric-based pruning and (optionally) gradient-based pruning.
                    layer_inputs_dict = {}
                    layer_outputs_dict = {}
                    layer_output_grads_dict = {}
                    sample_targets = None
                    sample_inputs = None

                    # Weight-gradient-based pruning strategies (populate module.weight.grad).
                    needs_weight_grads = strategy_name in {"gradient", "fisher"}
                    needs_layer_inputs = (strategy_name == "alignment") or (strategy_name == "hybrid") or (strategy_name in metric_based_strategies)
                    needs_layer_outputs = needs_layer_inputs  # capture outputs alongside inputs
                    needs_sample_batch = needs_layer_inputs or needs_weight_grads

                    # Some metric-based pruning criteria (e.g., taylor_saliency) require per-layer
                    # output gradients dL/d(output). We capture those via tensor hooks.
                    needs_output_grads = False
                    if strategy_name in metric_based_strategies:
                        try:
                            needs_output_grads = bool(getattr(getattr(strategy, "metric", None), "requires_gradients", False))
                        except Exception:
                            needs_output_grads = False

                    # Supernode protection can require gradients too if the *supernode score metric*
                    # is gradient-based.
                    supernode_cfg = getattr(self.config, "supernode_config", {}) or {}
                    supernode_metric = None
                    if isinstance(supernode_cfg, dict):
                        supernode_metric = supernode_cfg.get("score_metric", None)
                    if isinstance(supernode_metric, str) and supernode_metric:
                        try:
                            from alignment.core.registry import get_metric

                            m = get_metric(supernode_metric)
                            if m is not None:
                                needs_output_grads = needs_output_grads or bool(getattr(m, "requires_gradients", False))
                        except Exception:
                            pass

                    did_backward = False

                    if needs_sample_batch:
                        data_iter = iter(self.data_loader)
                        sample_batch, sample_targets = next(data_iter)
                        sample_inputs = sample_batch.to(self.config.device)
                        sample_targets = sample_targets.to(self.config.device)

                    if needs_layer_inputs and self.config.pruning_scope != "cascading":
                        # Capture inputs AND outputs for all layers once (used for global and layer-wise pruning).
                        hooks = []

                        def capture_input_output(name):
                            def hook(module, input, output):
                                # Capture inputs (used by most alignment / MI / RQ metrics).
                                try:
                                    layer_inputs_dict[name] = input[0].detach()
                                except Exception:
                                    layer_inputs_dict[name] = input

                                # Capture outputs (used by activation-based metrics), and optionally
                                # register a gradient hook (used by Taylor-style saliency).
                                out = output
                                if isinstance(out, (tuple, list)):
                                    for item in out:
                                        if torch.is_tensor(item):
                                            out = item
                                            break

                                if torch.is_tensor(out):
                                    if needs_output_grads and out.requires_grad:
                                        def _save_grad(grad, lname=name):
                                            layer_output_grads_dict[lname] = grad.detach()

                                        out.register_hook(_save_grad)
                                    layer_outputs_dict[name] = out.detach()
                                else:
                                    layer_outputs_dict[name] = out

                            return hook

                        # Register hooks
                        for name, module in self.model.named_modules():
                            if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                                hook = module.register_forward_hook(capture_input_output(name))
                                hooks.append(hook)

                        # Forward pass to capture inputs/outputs. If we need output gradients
                        # (e.g., Taylor saliency), run a real forward+backward so hooks can
                        # record dL/d(output) tensors.
                        if needs_output_grads:
                            was_training = self.model.training
                            self.model.eval()
                            self.model.zero_grad(set_to_none=True)
                            try:
                                outputs = self.model(sample_inputs)
                                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                                loss = nn.CrossEntropyLoss()(logits, sample_targets)
                                loss.backward()
                                did_backward = True
                            finally:
                                if was_training:
                                    self.model.train()
                        else:
                            with torch.no_grad():
                                _ = self.model(sample_inputs)

                        # Remove hooks
                        for hook in hooks:
                            hook.remove()

                        logger.debug(f"Captured inputs for {len(layer_inputs_dict)} layers")
                        
                        # Preprocess CNN inputs using unfold for proper RQ computation
                        layer_inputs_dict = self._preprocess_pruning_inputs(layer_inputs_dict)

                    if needs_weight_grads and self.config.pruning_scope != "cascading" and not did_backward:
                        # Gradient-based pruning requires a backward pass to populate .grad tensors.
                        was_training = self.model.training
                        self.model.eval()
                        self.model.zero_grad(set_to_none=True)
                        try:
                            outputs = self.model(sample_inputs)
                            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                            loss = nn.CrossEntropyLoss()(logits, sample_targets)
                            loss.backward()
                            if strategy_name == "fisher" and hasattr(strategy, "accumulate_fisher"):
                                strategy.accumulate_fisher(self.model)
                        finally:
                            if was_training:
                                self.model.train()

                    # Apply pruning
                    if pruning_config.global_pruning and hasattr(strategy, "prune_model"):
                        # Global pruning across all layers
                        if needs_layer_inputs:
                            # Global metric-based pruning needs layer inputs
                            masks = strategy.prune_model(self.model, layer_inputs_dict, amount=amount)
                        else:
                            # Global magnitude pruning
                            masks = strategy.prune_model(self.model, amount=amount)

                        # Calculate overall sparsity from masks
                        total_params = sum(mask.numel() for mask in masks.values())
                        zero_params = sum((mask == 0).sum().item() for mask in masks.values())
                        overall_sparsity = zero_params / total_params if total_params > 0 else 0

                    elif self.config.pruning_scope == "cascading":
                        # Cascading pruning: prune layers sequentially, recomputing any required
                        # per-layer statistics (inputs/outputs/gradients) after each pruning step.
                        direction = getattr(self.config, "cascading_direction", "forward")

                        ordered_layers = []
                        for lname, module in self.model.named_modules():
                            if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                                ordered_layers.append((lname, module))
                        if direction == "backward":
                            ordered_layers = ordered_layers[::-1]

                        logger.info(f"Cascading {direction} pruning of {len(ordered_layers)} layers (strategy={strategy_name})")

                        masks = {}
                        pruning_failed = False

                        for idx, (lname, module) in enumerate(ordered_layers):
                            logger.info(f"[Cascading] Pruning layer {idx+1}/{len(ordered_layers)}: {lname}")

                            layer_inputs = None
                            layer_outputs = None

                            if needs_sample_batch:
                                captured = {}

                                def _capture_io(_module, _input, _output):
                                    # Best-effort: capture the tensor input/output if present.
                                    try:
                                        captured["inputs"] = _input[0].detach()
                                    except Exception:
                                        captured["inputs"] = _input
                                    try:
                                        captured["outputs"] = _output.detach()
                                    except Exception:
                                        captured["outputs"] = _output

                                handle = module.register_forward_hook(_capture_io)
                                was_training = self.model.training
                                self.model.eval()
                                self.model.zero_grad(set_to_none=True)
                                try:
                                    if needs_gradients:
                                        outputs = self.model(sample_inputs)
                                        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                                        loss = nn.CrossEntropyLoss()(logits, sample_targets)
                                        loss.backward()
                                        if strategy_name == "fisher" and hasattr(strategy, "accumulate_fisher"):
                                            strategy.accumulate_fisher(self.model)
                                    else:
                                        with torch.no_grad():
                                            _ = self.model(sample_inputs)
                                except Exception as e:
                                    logger.error(f"[Cascading] Failed to compute forward/gradients for {lname}: {e}")
                                    pruning_failed = True
                                finally:
                                    handle.remove()
                                    if was_training:
                                        self.model.train()

                                if pruning_failed:
                                    break

                                raw_inputs = captured.get("inputs", None)
                                if raw_inputs is not None:
                                    # Preprocess CNN inputs for proper RQ computation (unfold, etc).
                                    layer_inputs = self._preprocess_pruning_inputs({lname: raw_inputs}).get(lname)
                                layer_outputs = captured.get("outputs", None)

                            if needs_layer_inputs and layer_inputs is None:
                                logger.debug(f"[Cascading] No captured inputs for layer {lname}; skipping.")
                                continue

                            try:
                                mask = strategy.prune(
                                    module,
                                    inputs=layer_inputs,
                                    outputs=layer_outputs,
                                    targets=sample_targets,
                                    module_name=lname,
                                    amount=amount,
                                )
                                masks[lname] = mask
                            except Exception as e:
                                logger.error(f"Pruning failed for strategy {strategy_name}: {e}")
                                pruning_failed = True
                                break

                        if pruning_failed:
                            logger.warning(f"Skipping strategy {strategy_name} due to errors")
                            continue

                        # Calculate overall sparsity
                        total_params = sum(mask.numel() for mask in masks.values())
                        zero_params = sum((mask == 0).sum().item() for mask in masks.values())
                        overall_sparsity = zero_params / total_params if total_params > 0 else 0
                    else:
                        dependency_result = None
                        if dependency_aware_enabled:
                            dependency_result = self._run_dependency_aware_pruning(
                                strategy=strategy,
                                strategy_name=strategy_name,
                                selection_mode=selection_mode,
                                amount=amount,
                                layer_inputs=layer_inputs_dict,
                                layer_outputs=layer_outputs_dict,
                                targets=sample_targets,
                            )

                        if dependency_result is not None:
                            layer_stats = dependency_result.get("stats", {})
                            layer_details = layer_stats.get("layers", {})
                            layer_sparsities = {
                                layer_name: stats.get("sparsity", 0.0) for layer_name, stats in layer_details.items()
                            }
                            overall_sparsity = layer_stats.get("overall_sparsity", 0.0)
                        else:
                            # Layer-wise pruning (fallback behavior)
                            layer_sparsities = {}
                            pruning_failed = False
                            for name, module in self.model.named_modules():
                                if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                                    layer_inputs = layer_inputs_dict.get(name) if needs_layer_inputs else None
                                    if needs_layer_inputs and layer_inputs is None:
                                        logger.debug(f"No captured inputs for layer {name} - skipping pruning for this layer")
                                        continue

                                    try:
                                        # Get outputs for this layer (needed for activation-based metrics)
                                        layer_outputs = layer_outputs_dict.get(name)
                                        layer_grads = layer_output_grads_dict.get(name) if needs_output_grads else None

                                        # Optional: protect a supernode core during pruning.
                                        sn_cfg = getattr(self.config, "supernode_config", {}) or {}
                                        sn_enabled = bool(sn_cfg.get("enabled", False)) if isinstance(sn_cfg, dict) else False
                                        sn_score_metric = sn_cfg.get("score_metric") if isinstance(sn_cfg, dict) else None
                                        sn_core_fraction = float(sn_cfg.get("core_fraction", 0.01)) if isinstance(sn_cfg, dict) else 0.01
                                        sn_protect_metrics = sn_cfg.get("protect_metrics") if isinstance(sn_cfg, dict) else None

                                        def _should_protect() -> bool:
                                            if not sn_enabled:
                                                return False
                                            if sn_protect_metrics is None:
                                                return True
                                            if isinstance(sn_protect_metrics, str):
                                                token = sn_protect_metrics.strip().lower()
                                                if token in {"all", "true", "yes", "1"}:
                                                    return True
                                                if token in {"none", "false", "no", "0", ""}:
                                                    return False
                                                sn_list = [x.strip() for x in sn_protect_metrics.split(",") if x.strip()]
                                                return strategy_name in set(sn_list)
                                            try:
                                                return strategy_name in set(sn_protect_metrics)
                                            except Exception:
                                                return False

                                        if _should_protect() and isinstance(sn_score_metric, str) and sn_score_metric:
                                            # Compute pruning scores (neuron/channel-wise), apply hard protection to the
                                            # top core_fraction by sn_score_metric, then prune normally by amount.
                                            raw_scores = strategy.compute_importance_scores(
                                                module,
                                                inputs=layer_inputs,
                                                outputs=layer_outputs,
                                                gradients=layer_grads,
                                                targets=sample_targets,
                                                module_name=name,
                                            )
                                            scores = self._reduce_scores_to_output_neurons(module, raw_scores)
                                            if scores is None:
                                                raise ValueError("Failed to reduce pruning scores to output-neuron scores")

                                            # Compute supernode scores (if different from pruning metric).
                                            if sn_score_metric == strategy_name:
                                                sn_scores = scores.detach().clone()
                                            else:
                                                from alignment.pruning.strategies import AlignmentPruning
                                                metric_kwargs = {}
                                                try:
                                                    metric_kwargs = (getattr(self.config, "metric_configs", {}) or {}).get(sn_score_metric, {}) or {}
                                                except Exception:
                                                    metric_kwargs = {}

                                                sn_strategy = AlignmentPruning(
                                                    metric=sn_score_metric,
                                                    config=PruningConfig(amount=0.0, structured=True, pruning_mode=selection_mode),
                                                    **metric_kwargs,
                                                )
                                                sn_raw = sn_strategy.compute_importance_scores(
                                                    module,
                                                    inputs=layer_inputs,
                                                    outputs=layer_outputs,
                                                    gradients=layer_grads,
                                                    targets=sample_targets,
                                                    module_name=name,
                                                )
                                                sn_scores = self._reduce_scores_to_output_neurons(module, sn_raw)
                                                if sn_scores is None:
                                                    raise ValueError("Failed to reduce supernode scores to output-neuron scores")

                                            n = int(scores.numel())
                                            k = max(1, int(round(sn_core_fraction * n)))
                                            # Protect TOP-k by supernode metric.
                                            _, top_idx = torch.topk(sn_scores, k, largest=True)
                                            core_mask = torch.zeros_like(scores, dtype=torch.bool)
                                            core_mask[top_idx] = True

                                            margin = torch.abs(scores).max().detach().item() + 1.0
                                            if selection_mode == "low":
                                                scores[core_mask] = scores.max() + margin
                                            elif selection_mode == "high":
                                                scores[core_mask] = scores.min() - margin

                                            mask = strategy.create_pruning_mask(scores, amount=amount, structured=True, pruning_mode=selection_mode)
                                            strategy.apply_pruning(module, mask)
                                        else:
                                            # Default path (no supernode protection): let the strategy handle pruning.
                                            strategy.prune(
                                                module,
                                                inputs=layer_inputs,
                                                outputs=layer_outputs,
                                                gradients=layer_grads,
                                                targets=sample_targets,
                                                module_name=name,
                                            )
                                        sparsity = strategy.get_sparsity(module)
                                        layer_sparsities[name] = sparsity
                                    except Exception as e:
                                        # Log error and skip this strategy entirely
                                        logger.error(f"Pruning failed for strategy {strategy_name}: {e}")
                                        pruning_failed = True
                                        break

                            if pruning_failed:
                                logger.warning(f"Skipping strategy {strategy_name} due to errors")
                                continue

                        total_params = 0
                        zero_params = 0
                        for module in self.model.modules():
                            if hasattr(module, "weight"):
                                total_params += module.weight.numel()
                                zero_params += (module.weight == 0).sum().item()

                        overall_sparsity = zero_params / total_params if total_params > 0 else 0

                    # Evaluate pruned model
                    test_loss_before, test_acc_before = self._evaluate()

                    # Capture weight distribution before fine-tuning
                    weight_dist_before = self._get_weight_distribution()

                    # Store before fine-tuning results
                    strategy_results["pruning_amounts"].append(amount)
                    strategy_results["accuracies_before_finetune"].append(test_acc_before)
                    strategy_results["losses_before_finetune"].append(test_loss_before)
                    strategy_results["sparsities"].append(overall_sparsity)
                    strategy_results["weight_distributions_before"].append(weight_dist_before)

                    # Fine-tune if configured
                    test_loss_after = test_loss_before
                    test_acc_after = test_acc_before

                    if self.config.fine_tune_after_pruning:
                        # Setup optimizer for fine-tuning
                        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate * 0.1)  # Lower learning rate

                        # Track fine-tuning progress
                        finetune_losses = []
                        finetune_accs = []

                        for epoch in range(self.config.fine_tune_epochs):
                            train_loss, train_acc = self._train_epoch(optimizer, nn.CrossEntropyLoss())
                            finetune_losses.append(train_loss)
                            finetune_accs.append(train_acc)

                        # Re-evaluate after fine-tuning
                        test_loss_after, test_acc_after = self._evaluate()

                        # Capture weight distribution after fine-tuning
                        weight_dist_after = self._get_weight_distribution()
                    else:
                        weight_dist_after = weight_dist_before

                    # Store after fine-tuning results
                    strategy_results["accuracies_after_finetune"].append(test_acc_after)
                    strategy_results["losses_after_finetune"].append(test_loss_after)
                    strategy_results["weight_distributions_after"].append(weight_dist_after)

                results["strategies"][result_key] = strategy_results

        # Final cleanup: remove any remaining pruning masks
        for name, module in self.model.named_modules():
            if hasattr(module, "weight_mask"):
                delattr(module, "weight_mask")
            if hasattr(module, "_pruning_hook"):
                module._pruning_hook.remove()
                delattr(module, "_pruning_hook")
            if hasattr(module, "_original_weight"):
                delattr(module, "_original_weight")

        # Restore original model
        self.model.load_state_dict(original_state)

        return results

    def _pruning_experiments_multi(self) -> Dict[str, Any]:
        """
        Perform pruning experiments on multiple networks with parallel evaluation.
        
        This is the default multi-network pruning implementation that:
        1. Creates masks for all networks and sparsity levels
        2. Evaluates all configurations in parallel
        3. Aggregates results with mean and std across networks
        
        OPTIMIZED: Pre-computes importance scores ONCE per metric, then creates
        all masks for all selection modes and sparsity levels in parallel.
        """
        # Get selection modes to test
        selection_modes = self.config.pruning_selection_mode
        if not isinstance(selection_modes, list):
            selection_modes = [selection_modes]

        results = {"strategies": {}}

        # Save original states efficiently
        original_states = []
        for model in self.networks:
            state = {name: module.weight.data.clone() for name, module in model.named_modules() if hasattr(module, "weight")}
            original_states.append(state)

        # Get sample inputs for alignment computation (ONCE)
        data_iter = iter(self.data_loader)
        sample_batch, sample_targets = next(data_iter)
        sample_inputs = sample_batch.to(self.config.device)
        sample_targets = sample_targets.to(self.config.device)

        # Pre-compute layer inputs for all networks ONCE
        logger.info("Pre-computing layer inputs for all networks...")
        all_layer_inputs = []
        for model in self.networks:
            layer_inputs_dict = self._capture_layer_inputs(model, sample_inputs)
            all_layer_inputs.append(layer_inputs_dict)

        # Pre-compute importance scores for each metric ONCE per network
        logger.info("Pre-computing importance scores for all metrics...")
        all_importance_scores = {}  # {strategy_name: [{layer_name: scores}]}
        
        # Check if we're doing single-layer pruning
        target_layer = getattr(self.config, 'pruning_target_layer', None)
        if target_layer:
            logger.info(f"Single-layer pruning: only computing scores for '{target_layer}'")
        
        for strategy_name in self.config.pruning_strategies:
            all_importance_scores[strategy_name] = []
            
            for net_idx, model in enumerate(self.networks):
                layer_scores = {}
                for name, module in model.named_modules():
                    if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                        # Skip layers that are not the target (if target is specified)
                        if target_layer is not None and name != target_layer:
                            continue
                            
                        layer_inputs = all_layer_inputs[net_idx].get(name)
                        if layer_inputs is not None:
                            try:
                                scores = self._compute_metric_importance(
                                    module, layer_inputs, strategy_name, sample_targets
                                )
                                layer_scores[name] = scores
                            except Exception as e:
                                logger.warning(f"Error computing {strategy_name} for {name}: {e}")
                                # Fallback to magnitude
                                layer_scores[name] = module.weight.abs().mean(dim=tuple(range(1, module.weight.dim())))
                
                all_importance_scores[strategy_name].append(layer_scores)

        # Now process each strategy with all selection modes and sparsity levels
        # This is optimized because we don't recompute scores
        for strategy_name in self.config.pruning_strategies:
            logger.info(f"Testing pruning strategy: {strategy_name}")

            for selection_mode in selection_modes:
                # Use pre-computed scores for fast mask creation
                batch_results = self._evaluate_pruning_with_precomputed_scores(
                    all_importance_scores[strategy_name],
                    selection_mode,
                    self.config.pruning_amounts,
                    original_states,
                )

                # Store aggregated results
                strategy_key = f"{strategy_name}_{selection_mode}"
                strategy_results = {
                    "pruning_amounts": self.config.pruning_amounts,
                    "sparsities": batch_results["sparsities"].mean(dim=0).tolist(),
                    "accuracies_before_finetune": batch_results["accuracies_before"].mean(dim=0).tolist(),
                    "accuracies_after_finetune": batch_results["accuracies_after"].mean(dim=0).tolist(),
                    "losses_before_finetune": batch_results["losses_before"].mean(dim=0).tolist(),
                    "losses_after_finetune": batch_results["losses_after"].mean(dim=0).tolist(),
                    "improvements": (batch_results["accuracies_after"] - batch_results["accuracies_before"]).mean(dim=0).tolist(),
                }

                # Add standard deviations if multiple networks
                if len(self.networks) > 1:
                    strategy_results["accuracies_before_finetune_std"] = batch_results["accuracies_before"].std(dim=0).tolist()
                    strategy_results["accuracies_after_finetune_std"] = batch_results["accuracies_after"].std(dim=0).tolist()

                results["strategies"][strategy_key] = strategy_results

        # Restore original weights
        for net_idx, model in enumerate(self.networks):
            for name, module in model.named_modules():
                if name in original_states[net_idx]:
                    module.weight.data = original_states[net_idx][name]

        return results
    
    def _compute_metric_importance(
        self, 
        module: nn.Module, 
        layer_inputs: torch.Tensor, 
        metric_name: str,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-neuron importance scores using a specified metric.
        
        Args:
            module: The layer module
            layer_inputs: Input activations to the layer
            metric_name: Name of the metric to use
            targets: Optional class labels for conditional metrics
            
        Returns:
            Per-neuron importance scores
        """
        from alignment.metrics import get_metric
        
        W = module.weight.detach()
        
        # Flatten spatial dimensions for conv layers
        if layer_inputs.dim() > 2:
            X = layer_inputs.view(layer_inputs.size(0), -1)
        else:
            X = layer_inputs
        
        # Flatten weights for conv layers
        if W.dim() > 2:
            W_flat = W.view(W.size(0), -1)
        else:
            W_flat = W
        
        try:
            metric = get_metric(metric_name)
            
            # Different metrics have different requirements
            if hasattr(metric, 'requires_weights') and metric.requires_weights:
                if hasattr(metric, 'requires_outputs') and metric.requires_outputs:
                    # Compute outputs
                    outputs = X @ W_flat.T if X.shape[1] == W_flat.shape[1] else None
                    if outputs is not None:
                        scores = metric.compute(inputs=X, weights=W_flat, outputs=outputs, targets=targets)
                    else:
                        scores = metric.compute(inputs=X, weights=W_flat, targets=targets)
                else:
                    scores = metric.compute(inputs=X, weights=W_flat, targets=targets)
            elif hasattr(metric, 'requires_outputs') and metric.requires_outputs:
                # Compute outputs for output-based metrics
                outputs = X @ W_flat.T if X.shape[1] == W_flat.shape[1] else None
                if outputs is not None:
                    scores = metric.compute(outputs=outputs, targets=targets)
                else:
                    raise ValueError(f"Cannot compute outputs for metric {metric_name}")
            else:
                scores = metric.compute(inputs=X, weights=W_flat, targets=targets)
            
            return scores.abs()
            
        except Exception as e:
            logger.warning(f"Metric {metric_name} computation failed: {e}. Using magnitude fallback.")
            # Fallback to magnitude
            return W.abs().mean(dim=tuple(range(1, W.dim())))
    
    def _evaluate_pruning_with_precomputed_scores(
        self,
        all_importance_scores: List[Dict[str, torch.Tensor]],
        selection_mode: str,
        pruning_amounts: List[float],
        original_states: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate pruning using pre-computed importance scores.
        
        This is much faster than recomputing scores for each configuration.
        
        If config.pruning_target_layer is set, only that layer will be pruned.
        """
        num_networks = len(self.networks)
        num_amounts = len(pruning_amounts)
        
        # Check if we're doing single-layer pruning
        target_layer = getattr(self.config, 'pruning_target_layer', None)
        if target_layer:
            logger.info(f"    Single-layer pruning mode: targeting layer '{target_layer}'")
        
        # Initialize result tensors
        accuracies_before = torch.zeros(num_networks, num_amounts)
        losses_before = torch.zeros(num_networks, num_amounts)
        accuracies_after = torch.zeros(num_networks, num_amounts)
        losses_after = torch.zeros(num_networks, num_amounts)
        sparsities = torch.zeros(num_networks, num_amounts)
        
        # Create all masks upfront using pre-computed scores
        all_masks = []  # [network][amount][layer_name -> mask]
        for net_idx in range(num_networks):
            network_masks = []
            for amount in pruning_amounts:
                layer_masks = {}
                for layer_name, scores in all_importance_scores[net_idx].items():
                    # Skip layers that are not the target (if target is specified)
                    if target_layer is not None and layer_name != target_layer:
                        continue
                    
                    mask = self._create_mask_from_scores(
                        self._get_module_by_name(self.networks[net_idx], layer_name),
                        scores,
                        amount,
                        selection_mode,
                    )
                    layer_masks[layer_name] = mask
                network_masks.append(layer_masks)
            all_masks.append(network_masks)
        
        # Evaluate all pruning amounts
        logger.info(f"    Evaluating {num_amounts} pruning amounts...")
        
        for amount_idx, amount in enumerate(pruning_amounts):
            # Apply masks to all networks for this pruning amount
            for net_idx, model in enumerate(self.networks):
                self._restore_weights_from_dict(model, original_states[net_idx])
                self._apply_tensorized_mask(model, all_masks[net_idx][amount_idx])
                sparsities[net_idx, amount_idx] = self._calculate_model_sparsity(model)
            
            # Evaluate all networks simultaneously
            batch_losses_before, batch_accs_before = self._evaluate_networks_batch()
            accuracies_before[:, amount_idx] = batch_accs_before
            losses_before[:, amount_idx] = batch_losses_before
            
            if amount_idx % 5 == 0 or amount_idx == num_amounts - 1:
                logger.info(
                    f"      {amount*100:.0f}% pruning - Avg Acc: {batch_accs_before.mean():.2f}%"
                )
        
        # Fine-tuning if requested
        if self.config.fine_tune_after_pruning:
            logger.info(f"    Fine-tuning all pruning configurations")
            for amount_idx, amount in enumerate(pruning_amounts):
                for net_idx, model in enumerate(self.networks):
                    self._restore_weights_from_dict(model, original_states[net_idx])
                    self._apply_tensorized_mask(model, all_masks[net_idx][amount_idx])
                
                self._finetune_networks_batch()
                batch_losses_after, batch_accs_after = self._evaluate_networks_batch()
                accuracies_after[:, amount_idx] = batch_accs_after
                losses_after[:, amount_idx] = batch_losses_after
        else:
            accuracies_after = accuracies_before.clone()
            losses_after = losses_before.clone()

        # Restore original weights
        for net_idx, model in enumerate(self.networks):
            self._restore_weights_from_dict(model, original_states[net_idx])
            self._cleanup_pruning_artifacts(model)

        return {
            "accuracies_before": accuracies_before,
            "losses_before": losses_before,
            "accuracies_after": accuracies_after,
            "losses_after": losses_after,
            "sparsities": sparsities,
        }
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """Get a module by its name."""
        for n, m in model.named_modules():
            if n == name:
                return m
        return None
    
    def _create_mask_from_scores(
        self,
        module: nn.Module,
        scores: torch.Tensor,
        amount: float,
        selection_mode: str,
    ) -> torch.Tensor:
        """Create a pruning mask from pre-computed scores."""
        if module is None:
            return None
            
        weights = module.weight
        num_neurons = scores.numel()
        num_to_prune = int(amount * num_neurons)
        
        if num_to_prune == 0:
            return torch.ones_like(weights)
        
        # Create keep mask
        keep_mask = torch.ones(num_neurons, dtype=torch.bool, device=weights.device)
        
        if selection_mode == "random":
            indices_to_prune = torch.randperm(num_neurons, device=weights.device)[:num_to_prune]
            keep_mask[indices_to_prune] = False
        elif selection_mode == "low":
            # Prune neurons with LOW scores (keep high)
            _, sorted_indices = torch.sort(scores)
            keep_mask[sorted_indices[:num_to_prune]] = False
        elif selection_mode == "high":
            # Prune neurons with HIGH scores (keep low)
            _, sorted_indices = torch.sort(scores, descending=True)
            keep_mask[sorted_indices[:num_to_prune]] = False
        
        # Expand mask to weight shape
        if len(weights.shape) == 2:
            mask = keep_mask.unsqueeze(1).expand_as(weights).float()
        else:
            mask = keep_mask.view(-1, 1, 1, 1).expand_as(weights).float()
        
        return mask

    def _tensorized_pruning_batch(self, strategy_name: str, selection_mode: str, pruning_amounts: List[float]) -> Dict[str, torch.Tensor]:
        """
        Process multiple networks and pruning amounts in a fully tensorized manner.

        Returns tensors of shape [num_networks, num_pruning_amounts] for each metric.
        """
        # Check if we should use the optimized version
        if hasattr(self.config, "use_optimized_pruning") and self.config.use_optimized_pruning:
            return self._tensorized_pruning_batch_optimized(strategy_name, selection_mode, pruning_amounts)

        # Original implementation continues below...
        num_networks = len(self.networks)
        num_amounts = len(pruning_amounts)

        # Initialize result tensors
        accuracies_before = torch.zeros(num_networks, num_amounts)
        losses_before = torch.zeros(num_networks, num_amounts)
        accuracies_after = torch.zeros(num_networks, num_amounts)
        losses_after = torch.zeros(num_networks, num_amounts)
        sparsities = torch.zeros(num_networks, num_amounts)

        # Save original states for all networks
        original_states = []
        for model in self.networks:
            original_states.append({name: module.weight.data.clone() for name, module in model.named_modules() if hasattr(module, "weight")})

        # Create masks for all networks and all pruning amounts at once
            all_masks = self._create_tensorized_masks(strategy_name, selection_mode, pruning_amounts)

        # Process all pruning amounts simultaneously
        logger.info(f"    Processing all {len(pruning_amounts)} pruning amounts simultaneously")

        # First, evaluate all pruning configurations before fine-tuning
        for amount_idx, amount in enumerate(pruning_amounts):
            logger.info(f"      Evaluating pruning amount: {amount * 100:.0f}%")

            # Apply masks to all networks for this pruning amount
            for net_idx, model in enumerate(self.networks):
                # Restore original weights first
                self._restore_weights_from_dict(model, original_states[net_idx])
                # Apply new mask
                self._apply_tensorized_mask(model, all_masks[net_idx][amount_idx])

            # Evaluate all networks simultaneously
            batch_losses_before, batch_accs_before = self._evaluate_networks_batch()
            accuracies_before[:, amount_idx] = batch_accs_before
            losses_before[:, amount_idx] = batch_losses_before

            # Calculate sparsities
            for net_idx, model in enumerate(self.networks):
                sparsities[net_idx, amount_idx] = self._calculate_model_sparsity(model)

            # Debug: Log diversity metrics
            acc_std = batch_accs_before.std().item()
            acc_min = batch_accs_before.min().item()
            acc_max = batch_accs_before.max().item()

            logger.debug(
                f"Before fine-tuning - Avg Acc: {batch_accs_before.mean():.2f}% (std: {acc_std:.2f}, range: [{acc_min:.2f}, {acc_max:.2f}]), Avg Sparsity: {sparsities[:, amount_idx].mean():.2%}"
            )

        # Now process fine-tuning if requested
        if self.config.fine_tune_after_pruning:
            logger.info(f"    Fine-tuning all pruning configurations")

            for amount_idx, amount in enumerate(pruning_amounts):
                logger.info(f"      Fine-tuning pruning amount: {amount * 100:.0f}%")

                # Restore and re-apply masks for this pruning amount
                for net_idx, model in enumerate(self.networks):
                    self._restore_weights_from_dict(model, original_states[net_idx])
                    self._apply_tensorized_mask(model, all_masks[net_idx][amount_idx])

                # Fine-tune all networks for this pruning amount
                self._finetune_networks_batch()

                # Evaluate after fine-tuning
                batch_losses_after, batch_accs_after = self._evaluate_networks_batch()
                accuracies_after[:, amount_idx] = batch_accs_after
                losses_after[:, amount_idx] = batch_losses_after

                # Log improvement
                improvement = batch_accs_after.mean() - accuracies_before[:, amount_idx].mean()
                logger.debug(f"After fine-tuning - Avg Acc: {batch_accs_after.mean():.2f}%, Improvement: {improvement:+.2f}%")
        else:
            # No fine-tuning, just copy before results
            accuracies_after = accuracies_before.clone()
            losses_after = losses_before.clone()

        # Restore original weights to all networks
        for net_idx, model in enumerate(self.networks):
            self._restore_weights_from_dict(model, original_states[net_idx])
            self._cleanup_pruning_artifacts(model)

        return {
            "accuracies_before": accuracies_before,
            "losses_before": losses_before,
            "accuracies_after": accuracies_after,
            "losses_after": losses_after,
            "sparsities": sparsities,
        }

    def _restore_weights_from_dict(self, model: nn.Module, weight_dict: Dict[str, torch.Tensor]):
        """Restore weights from a dictionary."""
        for name, module in model.named_modules():
            if name in weight_dict and hasattr(module, "weight"):
                with torch.no_grad():
                    module.weight.data.copy_(weight_dict[name])

    def _create_tensorized_masks_optimized(
        self, strategy_name: str, selection_mode: str, pruning_amounts: List[float]
    ) -> List[List[Dict[str, torch.Tensor]]]:
        """
        Create pruning masks for all networks and all pruning amounts with better efficiency.

        Returns: List[networks] of List[amounts] of Dict[layer_name, mask]
        """
        all_masks = []

        # Save current random state
        rng_state = torch.get_rng_state()

        # Set different random seeds for each network to ensure diversity
        base_seed = getattr(self.config, "seed", 42)

        for net_idx, model in enumerate(self.networks):
            network_masks = []

            # Get all layer weights for this network ONCE
            layer_info = {}
            for name, module in model.named_modules():
                if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                    layer_info[name] = {"weight": module.weight.detach().clone(), "shape": module.weight.shape}

            # Pre-compute importance scores ONCE for magnitude strategy
            layer_importance = {}

            if strategy_name == "magnitude":
                # Compute magnitude importance once
                for name, info in layer_info.items():
                    layer_importance[name] = info["weight"].abs()
            elif strategy_name == "random":
                # For pure random strategy, use uniform random scores
                for name, info in layer_info.items():
                    # Set seed for reproducibility but different for each layer
                    torch.manual_seed(base_seed + net_idx * 1000 + hash(name) % 1000)
                    layer_importance[name] = torch.rand_like(info["weight"])

            # Create masks for all pruning amounts
            for amount_idx, amount in enumerate(pruning_amounts):
                layer_masks = {}

                # For all strategies, use the importance scores with the selection mode
                for name, importance_scores in layer_importance.items():
                    # For random selection mode with magnitude strategy, we need special handling
                    if strategy_name == "magnitude" and selection_mode == "random":
                        # Use random selection but ensure reproducibility
                        unique_seed = base_seed + net_idx * 10000 + hash(name) % 1000 + amount_idx * 100
                        torch.manual_seed(unique_seed)
                        # Pass dummy scores to force random selection
                        dummy_scores = torch.ones_like(importance_scores)
                        mask = self._create_pruning_mask_tensor(dummy_scores, amount, "random")
                    else:
                        # Use the actual importance scores with the selection mode
                        mask = self._create_pruning_mask_tensor(importance_scores, amount, selection_mode)
                    layer_masks[name] = mask

                network_masks.append(layer_masks)

            all_masks.append(network_masks)

        # Restore original random state
        torch.set_rng_state(rng_state)

        return all_masks

    def _create_tensorized_masks(self, strategy_name: str, selection_mode: str, pruning_amounts: List[float]) -> List[List[Dict[str, torch.Tensor]]]:
        """
        Create pruning masks for all networks and all pruning amounts simultaneously.

        Returns: List[networks] of List[amounts] of Dict[layer_name, mask]
        """
        all_masks = []

        # Save current random state
        rng_state = torch.get_rng_state()

        # Set different random seeds for each network to ensure diversity
        base_seed = getattr(self.config, "seed", 42)

        for net_idx, model in enumerate(self.networks):
            network_masks = []

            # Get all layer weights for this network
            layer_weights = {}
            for name, module in model.named_modules():
                if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                    layer_weights[name] = module.weight.detach().clone()

            # Create masks for all pruning amounts for this network
            for amount_idx, amount in enumerate(pruning_amounts):
                layer_masks = {}

                if strategy_name == "magnitude":
                    # Magnitude-based pruning
                    for name, weight in layer_weights.items():
                        if selection_mode == "random":
                            # Random selection for magnitude strategy
                            unique_seed = base_seed + net_idx * 10000 + hash(name) % 1000 + amount_idx * 100
                            torch.manual_seed(unique_seed)
                            dummy_scores = torch.ones_like(weight)
                            mask = self._create_pruning_mask_tensor(dummy_scores, amount, "random")
                        else:
                            # Low/high selection based on magnitude
                            importance_scores = weight.abs()
                            mask = self._create_pruning_mask_tensor(importance_scores, amount, selection_mode)
                        layer_masks[name] = mask

                elif strategy_name == "random":
                    # Pure random pruning strategy - always use random selection
                    for name, weight in layer_weights.items():
                        # Unique seed: base + network + layer + amount
                        unique_seed = base_seed + net_idx * 10000 + hash(name) % 1000 + amount_idx * 100
                        torch.manual_seed(unique_seed)

                        # For random strategy, always use random selection
                        dummy_scores = torch.ones_like(weight)
                        mask = self._create_pruning_mask_tensor(dummy_scores, amount, "random")
                        layer_masks[name] = mask

                network_masks.append(layer_masks)

            all_masks.append(network_masks)

        # Restore original random state
        torch.set_rng_state(rng_state)

        return all_masks

    def _create_pruning_mask_tensor(self, importance_scores: torch.Tensor, amount: float, selection_mode: str) -> torch.Tensor:
        """
        Create a binary mask tensor based on importance scores and selection mode.

        REFACTORED (v0.2.0): Now uses MaskOperations service to eliminate redundancy.
        """
        # Use MaskOperations service instead of duplicate logic
        if importance_scores.ndim == 1:
            # Structured mask (per-neuron/channel)
            mask = MaskOperations.create_structured_mask(importance_scores, amount=amount, mode=selection_mode)
        else:
            # Unstructured mask (per-weight)
            mask = MaskOperations.create_unstructured_mask(importance_scores, amount=amount, mode=selection_mode)

        return mask.float()

    def _apply_tensorized_mask(self, model: nn.Module, layer_masks: Dict[str, torch.Tensor]):
        """Apply pruning masks to a model."""
        for name, module in model.named_modules():
            if name in layer_masks:
                mask = layer_masks[name]

                # Clean up any existing pruning artifacts first
                self._cleanup_single_module_pruning(module)

                # Store original weight before any modifications
                module.register_buffer("_original_weight", module.weight.data.clone())

                # Register mask as buffer
                module.register_buffer("weight_mask", mask)

                # Apply initial mask
                module.weight.data = module._original_weight * mask

                # Create hooks with proper closure capture
                mask_copy = mask.clone()  # Ensure mask persists in closure

                def create_gradient_hook(mask_tensor):
                    def mask_gradient_hook(grad):
                        # Mask gradients to prevent updates to pruned weights
                        return grad * mask_tensor if grad is not None else grad

                    return mask_gradient_hook

                # Register gradient hook
                module._gradient_hook_handle = module.weight.register_hook(create_gradient_hook(mask_copy))

    def _cleanup_single_module_pruning(self, module: nn.Module):
        """Clean up pruning artifacts from a single module."""
        if hasattr(module, "weight_mask"):
            delattr(module, "weight_mask")
        if hasattr(module, "_pruning_hook"):
            module._pruning_hook.remove()
            delattr(module, "_pruning_hook")
        if hasattr(module, "_original_weight"):
            delattr(module, "_original_weight")
        if hasattr(module, "_gradient_hook_handle"):
            module._gradient_hook_handle.remove()
            delattr(module, "_gradient_hook_handle")

    def _evaluate_networks_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate all networks simultaneously and return batch results."""
        num_networks = len(self.networks)

        # Put all networks in eval mode
        for net in self.networks:
            net.eval()

        batch_losses = torch.zeros(num_networks)
        batch_accs = torch.zeros(num_networks)

        criterion = nn.CrossEntropyLoss()

        # Check if we should limit evaluation batches
        eval_batches = getattr(self.config, "eval_batches", None)
        batch_count = 0

        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                # Forward pass through all networks
                outputs = [net(inputs) for net in self.networks]

                # Calculate losses and accuracies
                for i, output in enumerate(outputs):
                    loss = criterion(output, targets)
                    batch_losses[i] += loss.item()

                    _, predicted = output.max(1)
                    batch_accs[i] += predicted.eq(targets).sum().item()

                batch_count += 1

                # Check if we've evaluated enough batches
                if eval_batches is not None and batch_count >= eval_batches:
                    break

        # Average losses and convert accuracies to percentages
        num_batches = min(len(self.data_loader), eval_batches) if eval_batches else len(self.data_loader)
        batch_losses /= num_batches

        # For accuracy, we need to account for the actual number of samples evaluated
        if eval_batches:
            # Approximate based on batch size
            total_samples = num_batches * self.config.batch_size
        else:
            total_samples = len(self.data_loader.dataset)

        batch_accs = batch_accs * 100.0 / total_samples

        # Reset to train mode
        for net in self.networks:
            net.train()

        return batch_losses, batch_accs

    def _finetune_networks_batch(self):
        """Fine-tune all networks simultaneously with proper mask handling."""
        # Setup optimizers for all networks
        optimizers = []
        for net in self.networks:
            # Only optimize non-pruned parameters
            params_to_optimize = []
            for module in net.modules():
                if hasattr(module, "weight") and module.weight.requires_grad:
                    params_to_optimize.append(module.weight)
                if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
                    params_to_optimize.append(module.bias)

            optimizer = torch.optim.Adam(params_to_optimize, lr=getattr(self.config, "fine_tune_learning_rate", self.config.learning_rate * 0.1))
            optimizers.append(optimizer)

        criterion = nn.CrossEntropyLoss()

        # Fine-tuning loop
        for epoch in range(self.config.fine_tune_epochs):
            # Set all networks to training mode
            for net in self.networks:
                net.train()

            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                # Zero gradients for all optimizers
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # Forward pass through all networks
                outputs = [net(inputs) for net in self.networks]
                losses = [criterion(output, targets) for output in outputs]

                # Backward pass for all networks
                for loss, optimizer in zip(losses, optimizers):
                    loss.backward()

                    # Apply masks after gradient computation but before parameter update
                    optimizer.step()

                    # Re-apply masks to ensure pruned weights stay zero
                    self._reapply_masks_after_update(self.networks[optimizers.index(optimizer)])

    def _reapply_masks_after_update(self, model: nn.Module):
        """Re-apply pruning masks after parameter updates to ensure pruned weights stay zero."""
        for module in model.modules():
            if hasattr(module, "weight_mask") and hasattr(module, "_original_weight"):
                # Re-apply the mask to keep pruned weights at zero
                with torch.no_grad():
                    module.weight.data *= module.weight_mask
                    # Also handle bias if it exists and has a mask
                    if hasattr(module, "bias") and module.bias is not None and hasattr(module, "bias_mask"):
                        module.bias.data *= module.bias_mask

    def _calculate_model_sparsity(self, model: nn.Module) -> float:
        """Calculate the overall sparsity of a model."""
        total_params = 0
        zero_params = 0

        for module in model.modules():
            if hasattr(module, "weight"):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def _reduce_scores_to_output_neurons(self, module: nn.Module, scores: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Collapse importance scores to a 1D tensor aligned with the module's output channels.
        """
        if scores is None:
            return None

        if scores.dim() == 1:
            reduced = scores
        else:
            dims = tuple(range(1, scores.dim()))
            reduced = scores.abs()
            if dims:
                reduced = reduced.mean(dim=dims)

        expected = module.weight.shape[0]
        if reduced.numel() != expected:
            logger.warning(
                f"Dependency-aware pruning: score length {reduced.numel()} does not match output dim {expected} "
                f"for layer {module.__class__.__name__}; skipping."
            )
            return None

        return reduced.reshape(expected)

    def _run_dependency_aware_pruning(
        self,
        strategy,
        strategy_name: str,
        selection_mode: str,
        amount: float,
        layer_inputs: Dict[str, torch.Tensor],
        layer_outputs: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply dependency-aware pruning by converting per-layer scores into masks that respect
        downstream dependencies (e.g., Conv blocks, residual connections).
        
        Uses the shared pruning pipeline with configurable distribution options.
        """
        layer_scores: Dict[str, torch.Tensor] = {}
        layer_outputs = layer_outputs or {}

        for name, module in self.model.named_modules():
            if not hasattr(module, "weight") or module.weight.dim() < 2:
                continue

            # All metric-based strategies (RQ, MI, activation, etc.) require inputs
            # Only magnitude-based pruning doesn't need inputs
            inputs = layer_inputs.get(name)
            if inputs is None:
                logger.debug(f"Dependency-aware pruning: missing inputs for {name}, skipping.")
                continue

            # Get outputs for activation-based metrics
            outputs = layer_outputs.get(name)

            try:
                scores = strategy.compute_importance_scores(module, inputs=inputs, outputs=outputs, targets=targets)
            except Exception as exc:
                logger.error(f"Dependency-aware pruning: failed to compute scores for {name}: {exc}")
                continue

            neuron_scores = self._reduce_scores_to_output_neurons(module, scores)
            if neuron_scores is None:
                continue

            layer_scores[name] = neuron_scores

        if not layer_scores:
            logger.warning("Dependency-aware pruning requested but no valid layer scores were computed.")
            return None

        # Use shared pruning pipeline with config-driven options
        pipeline_options = PruningPipelineOptions(
            distribution=getattr(self.config, "pruning_distribution", "uniform"),
            dependency_aware=True,  # Always true for this method
            min_amount=getattr(self.config, "pruning_min_per_layer", 0.0),
            max_amount=getattr(self.config, "pruning_max_per_layer", 0.95),
        )
        
        try:
            result = run_pruning_pipeline(
                model=self.model,
                layer_scores=layer_scores,
                target_sparsity=amount,
                selection_mode=selection_mode,
                options=pipeline_options,
            )
        except ValueError as exc:
            logger.error(f"Dependency-aware pruning failed validation: {exc}")
            return None

        return result

    def _tensorized_alignment_pruning_batch(self, strategy_name: str, selection_mode: str, pruning_amounts: List[float]) -> Dict[str, torch.Tensor]:
        """
        Handle alignment-based pruning with tensorized evaluation and proper state preservation.

        Alignment-based pruning requires layer inputs, so we need to process networks
        individually but still use tensorized evaluation.
        """
        num_networks = len(self.networks)
        num_amounts = len(pruning_amounts)

        # Initialize result tensors
        accuracies_before = torch.zeros(num_networks, num_amounts)
        losses_before = torch.zeros(num_networks, num_amounts)
        accuracies_after = torch.zeros(num_networks, num_amounts)
        losses_after = torch.zeros(num_networks, num_amounts)
        sparsities = torch.zeros(num_networks, num_amounts)

        # Save original states for all networks
        original_states = []
        for model in self.networks:
            original_states.append({name: module.weight.data.clone() for name, module in model.named_modules() if hasattr(module, "weight")})

        # Get sample inputs for alignment computation
        data_iter = iter(self.data_loader)
        sample_batch, _ = next(data_iter)
        sample_inputs = sample_batch.to(self.config.device)

        # Pre-compute layer inputs for all networks ONCE
        all_layer_inputs = []
        for model in self.networks:
            layer_inputs_dict = self._capture_layer_inputs(model, sample_inputs)
            all_layer_inputs.append(layer_inputs_dict)

        # Process all pruning amounts
        logger.info(f"    Processing all {len(pruning_amounts)} alignment pruning amounts")

        # First, evaluate all pruning configurations before fine-tuning
        for amount_idx, amount in enumerate(pruning_amounts):
            logger.info(f"      Evaluating alignment pruning amount: {amount * 100:.0f}%")

            # Apply alignment-based pruning to each network
            for net_idx, model in enumerate(self.networks):
                # Restore original weights first
                self._restore_weights_from_dict(model, original_states[net_idx])

                # Apply alignment-based pruning using pre-computed inputs
                if self.config.alignment_structured_pruning:
                    # Use structured pruning for alignment methods
                    self._apply_alignment_pruning_optimized(model, all_layer_inputs[net_idx], amount, selection_mode)
                else:
                    # Use unstructured pruning (original behavior)
                    self._apply_alignment_pruning(model, all_layer_inputs[net_idx], amount, selection_mode)

                # Calculate sparsity
                sparsities[net_idx, amount_idx] = self._calculate_model_sparsity(model)

            # Evaluate all networks simultaneously
            batch_losses_before, batch_accs_before = self._evaluate_networks_batch()
            accuracies_before[:, amount_idx] = batch_accs_before
            losses_before[:, amount_idx] = batch_losses_before

            logger.debug(
                f"Before fine-tuning - Avg Acc: {batch_accs_before.mean():.2f}%, Avg Sparsity: {sparsities[:, amount_idx].mean():.2%}"
            )

        # Now process fine-tuning if requested
        if self.config.fine_tune_after_pruning:
            logger.info(f"    Fine-tuning all alignment pruning configurations")

            for amount_idx, amount in enumerate(pruning_amounts):
                logger.info(f"      Fine-tuning alignment pruning amount: {amount * 100:.0f}%")

                # Restore and re-apply pruning for this amount
                for net_idx, model in enumerate(self.networks):
                    self._restore_weights_from_dict(model, original_states[net_idx])
                    if self.config.alignment_structured_pruning:
                        # Use structured pruning for alignment methods
                        self._apply_alignment_pruning_optimized(model, all_layer_inputs[net_idx], amount, selection_mode)
                    else:
                        # Use unstructured pruning (original behavior)
                        self._apply_alignment_pruning(model, all_layer_inputs[net_idx], amount, selection_mode)

                # Fine-tune all networks
                self._finetune_networks_batch()

                # Evaluate after fine-tuning
                batch_losses_after, batch_accs_after = self._evaluate_networks_batch()
                accuracies_after[:, amount_idx] = batch_accs_after
                losses_after[:, amount_idx] = batch_losses_after

                # Log improvement
                improvement = batch_accs_after.mean() - accuracies_before[:, amount_idx].mean()
                logger.debug(f"After fine-tuning - Avg Acc: {batch_accs_after.mean():.2f}%, Improvement: {improvement:+.2f}%")
        else:
            # No fine-tuning, just copy before results
            accuracies_after = accuracies_before.clone()
            losses_after = losses_before.clone()

        # Restore original weights to all networks
        for net_idx, model in enumerate(self.networks):
            self._restore_weights_from_dict(model, original_states[net_idx])
            self._cleanup_pruning_artifacts(model)

        return {
            "accuracies_before": accuracies_before,
            "losses_before": losses_before,
            "accuracies_after": accuracies_after,
            "losses_after": losses_after,
            "sparsities": sparsities,
        }

    def _apply_alignment_pruning_optimized(self, model: nn.Module, layer_inputs_dict: Dict[str, torch.Tensor], amount: float, selection_mode: str):
        """Apply alignment-based pruning more efficiently with proper structured pruning."""
        # Process all layers at once
        all_masks = {}

        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                layer_inputs = layer_inputs_dict.get(name)
                if layer_inputs is not None:
                    # Compute alignment-based importance scores (per neuron)
                    neuron_importance = self._compute_neuron_alignment_importance(module, layer_inputs)

                    # Create structured mask based on selection mode
                    mask = self._create_structured_pruning_mask(module, neuron_importance, amount, selection_mode)
                    all_masks[name] = mask

        # Apply all masks at once
        self._apply_tensorized_mask(model, all_masks)

    def _compute_neuron_alignment_importance(self, module: nn.Module, layer_inputs: torch.Tensor) -> torch.Tensor:
        """Compute per-neuron alignment scores for structured pruning."""
        from alignment.metrics.rayleigh.rayleigh_quotient import RayleighQuotient

        try:
            # Get weight matrix
            W = module.weight.detach()

            # Ensure inputs have the right shape
            if layer_inputs.dim() > 2:
                # Flatten spatial dimensions if needed (for conv layers)
                X = layer_inputs.view(layer_inputs.size(0), -1)
            else:
                X = layer_inputs

            # Initialize the RayleighQuotient metric
            rq_metric = RayleighQuotient(relative=True, min_samples=2)

            # Compute RQ scores for each neuron (these represent alignment with input covariance)
            neuron_scores = rq_metric.compute(inputs=X, weights=W)

            # Return per-neuron scores
            return neuron_scores.abs()

        except Exception as e:
            logger.warning(f"Error computing alignment importance: {e}. Falling back to magnitude.")
            # Fallback to magnitude-based importance per neuron
            return module.weight.abs().mean(dim=1)

    def _create_structured_pruning_mask(self, module: nn.Module, neuron_scores: torch.Tensor, amount: float, selection_mode: str) -> torch.Tensor:
        """Create a structured pruning mask that removes entire neurons."""
        weights = module.weight
        num_neurons = neuron_scores.numel()
        num_to_prune = int(amount * num_neurons)

        if num_to_prune == 0:
            return torch.ones_like(weights)

        # Initialize keep mask (True = keep neuron, False = prune neuron)
        keep_mask = torch.ones(num_neurons, dtype=torch.bool, device=weights.device)

        if selection_mode == "random":
            # Random selection of neurons to prune
            indices_to_prune = torch.randperm(num_neurons, device=weights.device)[:num_to_prune]
            keep_mask[indices_to_prune] = False
        elif selection_mode == "low":
            # Prune neurons with LOW alignment scores (keep high alignment)
            _, sorted_indices = torch.sort(neuron_scores)
            indices_to_prune = sorted_indices[:num_to_prune]
            keep_mask[indices_to_prune] = False
        elif selection_mode == "high":
            # Prune neurons with HIGH alignment scores (keep low alignment)
            _, sorted_indices = torch.sort(neuron_scores, descending=True)
            indices_to_prune = sorted_indices[:num_to_prune]
            keep_mask[indices_to_prune] = False

        # Expand mask to all weights in the neuron
        if len(weights.shape) == 2:  # Linear layer
            mask = keep_mask.unsqueeze(1).expand_as(weights).float()
        else:  # Conv layer
            mask = keep_mask.view(-1, 1, 1, 1).expand_as(weights).float()

        return mask

    def _capture_layer_inputs(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Capture inputs to each layer for alignment-based pruning."""
        layer_inputs_dict = {}
        hooks = []

        def create_capture_hook(name):
            def hook(module, input, output):
                layer_inputs_dict[name] = input[0].detach()

            return hook

        # Register hooks
        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                hook = module.register_forward_hook(create_capture_hook(name))
                hooks.append(hook)

        # Forward pass to capture inputs
        with torch.no_grad():
            model(inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Preprocess CNN inputs for proper RQ computation
        return self._preprocess_pruning_inputs(layer_inputs_dict)

    def _apply_alignment_pruning(self, model: nn.Module, layer_inputs_dict: Dict[str, torch.Tensor], amount: float, selection_mode: str):
        """Apply alignment-based pruning to a model with structured pruning."""
        # For alignment-based pruning, we need to compute alignment scores manually
        # to ensure proper handling of different selection modes

        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) >= 2:
                layer_inputs = layer_inputs_dict.get(name)
                if layer_inputs is not None:
                    # Clean up any existing pruning artifacts
                    self._cleanup_single_module_pruning(module)

                    # Compute alignment-based importance scores (per neuron)
                    neuron_importance = self._compute_neuron_alignment_importance(module, layer_inputs)

                    # Create structured mask based on selection mode
                    mask = self._create_structured_pruning_mask(module, neuron_importance, amount, selection_mode)

                    # Apply the mask
                    layer_masks = {name: mask}
                    self._apply_tensorized_mask(model, layer_masks)
                    break  # Only apply to one layer at a time to avoid conflicts

    def _compute_alignment_importance(self, module: nn.Module, layer_inputs: torch.Tensor) -> torch.Tensor:
        """Compute alignment-based importance scores for a layer."""
        from alignment.metrics.rayleigh.rayleigh_quotient import RayleighQuotient

        try:
            # Get weight matrix
            W = module.weight.detach()

            # Ensure inputs have the right shape
            if layer_inputs.dim() > 2:
                # Flatten spatial dimensions if needed (for conv layers)
                X = layer_inputs.view(layer_inputs.size(0), -1)
            else:
                X = layer_inputs

            # Initialize the RayleighQuotient metric
            rq_metric = RayleighQuotient(relative=True, min_samples=2)

            # Compute RQ scores for each neuron (these represent alignment with input covariance)
            rq_scores = rq_metric.compute(inputs=X, weights=W)

            # Convert per-neuron scores to per-weight importance scores
            # Each weight in a neuron gets the same importance as the neuron
            importance_scores = torch.zeros_like(W)
            for i in range(W.size(0)):
                importance_scores[i, :] = rq_scores[i]

            # Use absolute values and add small noise to break ties for diversity
            importance_scores = importance_scores.abs()

            return importance_scores

        except Exception as e:
            logger.warning(f"Error computing alignment importance: {e}. Falling back to magnitude.")
            # Fallback to magnitude-based importance
            return module.weight.abs()

    def _pruning_experiments_tensorized_detailed(self) -> Dict[str, Any]:
        """Tensorized pruning with detailed per-network results."""
        # TODO: Implement per-network (not aggregated) tensorized pruning results.
        # This should return the same keys as `_pruning_experiments_tensorized()`, but with
        # per-network curves preserved for later variance/error-bar plots.
        logger.info("Detailed tensorized pruning not yet implemented, using aggregated results")
        return self._pruning_experiments_tensorized()

    def run(self) -> Dict[str, Any]:
        """Run the general alignment experiment."""
        logger.info("Starting general alignment experiment")

        # Train model
        self.train_results = self._train_model()

        # Final evaluation
        logger.info("=" * 60)
        logger.info("PHASE: Final Evaluation")
        logger.info("=" * 60)
        test_loss, test_acc = self._evaluate()
        
        logger.info("=" * 60)
        logger.info("PHASE: Measuring Alignment")
        logger.info("=" * 60)
        alignment_results = self._measure_alignment()
        
        # Compute pairwise redundancy matrices for visualization
        redundancy_matrices = self._compute_redundancy_matrices()
        
        self.test_results = {
            "final_loss": test_loss, 
            "final_accuracy": test_acc, 
            "alignment": alignment_results,
            "redundancy_matrices": redundancy_matrices,
        }

        # Dropout analysis
        logger.info("=" * 60)
        logger.info("PHASE: Dropout Analysis")
        logger.info("=" * 60)
        self.dropout_results = self._dropout_analysis()

        # Pruning experiments
        logger.info("=" * 60)
        logger.info("PHASE: Pruning Experiments")
        logger.info("=" * 60)
        self.pruning_results = self._pruning_experiments()

        # Eigenfeature analysis (optional)
        logger.info("=" * 60)
        logger.info("PHASE: Eigenfeature Analysis")
        logger.info("=" * 60)
        self.eigenfeature_results = self._run_eigenfeature_analysis()

        # Combine all results
        all_results = {
            "config": self.config.to_dict(),
            "train_results": self.train_results,
            "test_results": self.test_results,
            "dropout_results": self.dropout_results,
            "pruning_results": self.pruning_results,
            "eigenfeature_results": self.eigenfeature_results,
        }

        # Save results
        self.results.update(all_results)
        self.save_results()

        # Generate visualizations
        if self.config.generate_plots:
            self._generate_visualizations()

        logger.info("General alignment experiment completed")

        return all_results

    def _get_weight_distribution(self) -> Dict[str, Dict[str, Any]]:
        """Get weight distribution statistics for each layer."""
        weight_stats = {}

        for name, module in self.model.named_modules():
            if hasattr(module, "weight"):
                weight = module.weight.detach().cpu()

                # Get the actual weight values (not masked values)
                if hasattr(module, "weight_mask"):
                    # For pruned weights, we want to see the non-zero values
                    mask = module.weight_mask.detach().cpu()
                    non_zero_weights = weight[mask != 0]
                else:
                    non_zero_weights = weight.flatten()

                if len(non_zero_weights) > 0:
                    # Sample if tensor is too large for quantile (limit ~16M elements)
                    max_quantile_size = 10_000_000
                    if len(non_zero_weights) > max_quantile_size:
                        indices = torch.randperm(len(non_zero_weights))[:max_quantile_size]
                        sampled_weights = non_zero_weights[indices]
                    else:
                        sampled_weights = non_zero_weights
                    
                    weight_stats[name] = {
                        "mean": float(non_zero_weights.mean()),
                        "std": float(non_zero_weights.std()),
                        "min": float(non_zero_weights.min()),
                        "max": float(non_zero_weights.max()),
                        "percentiles": {
                            "1": float(torch.quantile(sampled_weights, 0.01)),
                            "25": float(torch.quantile(sampled_weights, 0.25)),
                            "50": float(torch.quantile(sampled_weights, 0.50)),
                            "75": float(torch.quantile(sampled_weights, 0.75)),
                            "99": float(torch.quantile(sampled_weights, 0.99)),
                        },
                        "sparsity": float((weight == 0).sum()) / weight.numel() if weight.numel() > 0 else 0,
                    }

        return weight_stats

    def _generate_visualizations(self):
        """Generate comprehensive visualizations using the unified visualization module."""
        output_dir = Path(getattr(self.config, "plots_dir", Path(self.config.log_dir) / "plots"))
        output_dir.mkdir(parents=True, exist_ok=True)

        from alignment.analysis.visualization import UnifiedVisualizer, generate_experiment_visualizations

        # Use the centralized visualization function for standard plots
        # Calculate total_params from model for secondary x-axis in pruning plots
        total_params = None
        if hasattr(self, 'model') and self.model is not None:
            try:
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            except:
                pass
        
        all_results = {
            "train_results": self.train_results,
            "test_results": self.test_results,
            "dropout_results": self.dropout_results,
            "pruning_results": self.pruning_results,
            "eigenfeature_results": self.eigenfeature_results,
            "model_info": {"total_params": total_params} if total_params is not None else {},
        }
        
        try:
            generate_experiment_visualizations(
                results=all_results,
                output_dir=output_dir,
                config=self.config,
                dpi=self.config.plot_dpi,
            )
        except Exception as e:
            logger.warning(f"Centralized visualization failed, using fallback: {e}")
            # Fallback to inline visualization if centralized fails
            self._generate_visualizations_fallback(output_dir)
        
        logger.info(f"Saved visualizations to {output_dir}")
    
    def _generate_visualizations_fallback(self, output_dir: Path):
        """Fallback visualization method if centralized visualization fails.
        
        Uses consistent folder structure matching LLM experiments:
        - training/ - Training curves, alignment evolution
        - pruning/  - Pruning comparison plots
        - histograms/ - Score distribution histograms
        - scatter/  - Metric scatter plots
        - redundancy/ - Redundancy heatmaps
        """
        from alignment.analysis.visualization import UnifiedVisualizer
        
        visualizer = UnifiedVisualizer()
        
        # Create subfolders for organized output (matching LLM experiment format)
        training_dir = output_dir / "training"
        pruning_dir = output_dir / "pruning"
        histogram_dir = output_dir / "histograms"
        scatter_dir = output_dir / "scatter"
        redundancy_dir = output_dir / "redundancy"
        
        for d in [training_dir, pruning_dir, histogram_dir, scatter_dir, redundancy_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Optional fine-grained visualization control via config.visualization_options
        viz_opts: Dict[str, Any] = getattr(self.config, "visualization_options", {}) or {}
        show_training = viz_opts.get("training_curves", True)
        show_alignment = viz_opts.get("alignment_curves", True)
        show_dropout = viz_opts.get("dropout_plots", True)
        show_eigen = viz_opts.get("eigen_plots", True)
        show_pruning = viz_opts.get("pruning_plots", True)

        # Training curves (saved to training/ subfolder)
        train_losses = self.train_results.get("train_losses", [])
        val_losses = self.train_results.get("val_losses", [])
        train_accs = self.train_results.get("train_accs", [])
        val_accs = self.train_results.get("val_accs", [])
        epochs = list(range(1, len(train_losses) + 1))

        if epochs and show_training:
            loss_series = {}
            if train_losses:
                loss_series["Train Loss"] = train_losses
            if val_losses and len(val_losses) == len(epochs):
                loss_series["Val Loss"] = val_losses
            if loss_series:
                fig = visualizer.plot_metric_evolution(
                    epochs,
                    loss_series,
                    title="Training Loss",
                    xlabel="Epoch",
                    ylabel="Loss",
                    legend_title="Split",
                    show_confidence=False,
                    save_path=training_dir / "training_loss.png",
                )
                plt.close(fig)

            acc_series = {}
            if train_accs:
                acc_series["Train Acc"] = train_accs
            if val_accs and len(val_accs) == len(epochs):
                acc_series["Val Acc"] = val_accs
            if acc_series:
                fig = visualizer.plot_metric_evolution(
                    epochs,
                    acc_series,
                    title="Training Accuracy",
                    xlabel="Epoch",
                    ylabel="Accuracy (%)",
                    legend_title="Split",
                    show_confidence=False,
                    save_path=training_dir / "training_accuracy.png",
                )
                plt.close(fig)

        # Alignment evolution (saved to training/ subfolder)
        alignment_history = self.train_results.get("alignment", {})
        if show_alignment and alignment_history:
            for method, history in alignment_history.items():
                summarized = []
                for snapshot in history:
                    aggregated_scores = []
                    for layer_scores in snapshot.values():
                        aggregated_scores.extend(layer_scores)
                    if aggregated_scores:
                        summarized.append(float(np.mean(aggregated_scores)))
                if summarized:
                    steps = list(range(1, len(summarized) + 1))
                    # Use lowercase underscore naming like LLM
                    method_safe = method.lower().replace(" ", "_")
                    fig = visualizer.plot_metric_evolution(
                        steps,
                        {method: summarized},
                        title=f"{method} Alignment Evolution",
                        xlabel="Measurement Index",
                        ylabel="Average Score",
                        legend_title="Metric",
                        show_confidence=False,
                        save_path=training_dir / f"alignment_{method_safe}.png",
                    )
                    plt.close(fig)

        # Dropout analysis (saved to training/ subfolder)
        dropout_rates = self.dropout_results.get("dropout_rates", [])
        if dropout_rates and show_dropout:
            accuracy_curves = {}
            loss_curves = {}

            if "accuracies" in self.dropout_results:
                for strategy, values in self.dropout_results["accuracies"].items():
                    if values:
                        accuracy_curves[strategy] = values
            else:
                for strategy in ["low", "high", "random"]:
                    key = f"accuracies_{strategy}"
                    if key in self.dropout_results:
                        accuracy_curves[strategy] = self.dropout_results[key]

            if "losses" in self.dropout_results:
                for strategy, values in self.dropout_results["losses"].items():
                    if values:
                        loss_curves[strategy] = values
            else:
                for strategy in ["low", "high", "random"]:
                    key = f"losses_{strategy}"
                    if key in self.dropout_results:
                        loss_curves[strategy] = self.dropout_results[key]

            dropout_steps = [rate * 100.0 for rate in dropout_rates]

            if accuracy_curves:
                fig = visualizer.plot_metric_evolution(
                    dropout_steps,
                    accuracy_curves,
                    title="Dropout Accuracy vs Rate",
                    xlabel="Dropout (%)",
                    ylabel="Accuracy (%)",
                    legend_title="Strategy",
                    show_confidence=False,
                    save_path=training_dir / "dropout_accuracy.png",
                )
                plt.close(fig)

            if loss_curves:
                fig = visualizer.plot_metric_evolution(
                    dropout_steps,
                    loss_curves,
                    title="Dropout Loss vs Rate",
                    xlabel="Dropout (%)",
                    ylabel="Loss",
                    legend_title="Strategy",
                    show_confidence=False,
                    save_path=training_dir / "dropout_loss.png",
                )
                plt.close(fig)

        # Eigenfeature analysis visualizations (saved to training/ subfolder)
        if self.eigenfeature_results and show_eigen:
            eigen_heatmap_data = {}
            for layer_name, info in self.eigenfeature_results.items():
                eigenvalues = info.get("top_eigenvalues", [])
                if eigenvalues:
                    eigen_heatmap_data[layer_name] = {f"eig{i+1}": val for i, val in enumerate(eigenvalues)}

            if eigen_heatmap_data:
                fig = visualizer.plot_heatmap(
                    data=eigen_heatmap_data,
                    title="Top Eigenvalues per Layer",
                    xlabel="Eigenvalue Index",
                    ylabel="Layer",
                    save_path=training_dir / "eigenvalues_heatmap.png",
                )
                plt.close(fig)

        # Pruning experiments (saved to pruning/ subfolder) - now enhanced with before/after comparisons
        if self.pruning_results and "strategies" in self.pruning_results and show_pruning:
            # Group results by algorithm (for multi-selection mode comparison)
            algorithm_results = {}

            for strategy_key, strategy_results in self.pruning_results["strategies"].items():
                if not strategy_results.get("pruning_amounts"):
                    continue

                # Extract algorithm name and selection mode from key (use lowercase like LLM)
                if "_" in strategy_key and strategy_key.split("_")[-1] in ["low", "high", "random"]:
                    # Format: "algorithm_selectionmode"
                    parts = strategy_key.rsplit("_", 1)
                    algorithm = parts[0].lower().replace(" ", "_")
                    selection_mode = parts[1]
                else:
                    # Single selection mode
                    algorithm = strategy_key.lower().replace(" ", "_")
                    selection_mode = self.config.pruning_selection_mode
                    if isinstance(selection_mode, list):
                        selection_mode = selection_mode[0]

                # Initialize algorithm group if needed
                if algorithm not in algorithm_results:
                    algorithm_results[algorithm] = {
                        "sparsities": strategy_results["sparsities"],
                        "before": {},
                        "after": {},
                        "before_losses": {},
                        "after_losses": {},
                    }

                # Store accuracies by selection mode
                algorithm_results[algorithm]["before"][selection_mode] = strategy_results["accuracies_before_finetune"]
                algorithm_results[algorithm]["after"][selection_mode] = strategy_results["accuracies_after_finetune"]

                # Store losses by selection mode if available
                if "losses_before_finetune" in strategy_results:
                    algorithm_results[algorithm]["before_losses"][selection_mode] = strategy_results["losses_before_finetune"]
                if "losses_after_finetune" in strategy_results:
                    algorithm_results[algorithm]["after_losses"][selection_mode] = strategy_results["losses_after_finetune"]

                # Store standard deviations if available
                if "accuracies_before_finetune_std" in strategy_results:
                    if "before_std" not in algorithm_results[algorithm]:
                        algorithm_results[algorithm]["before_std"] = {}
                        algorithm_results[algorithm]["after_std"] = {}
                    algorithm_results[algorithm]["before_std"][selection_mode] = strategy_results["accuracies_before_finetune_std"]
                    algorithm_results[algorithm]["after_std"][selection_mode] = strategy_results["accuracies_after_finetune_std"]
            
            # Compute total model parameters for secondary x-axis
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # Generate plots for each algorithm using UnifiedVisualizer (use pruning_dir)
            for algorithm, results in algorithm_results.items():
                # Create before/after comparison plots (saved to pruning/ subfolder)
                figs = visualizer.plot_pruning_before_after(
                    sparsities=results["sparsities"],
                    before_accuracies=results["before"],
                    after_accuracies=results["after"],
                    before_std=results.get("before_std"),
                    after_std=results.get("after_std"),
                    algorithm=algorithm.replace("_", " ").title(),
                    save_dir=pruning_dir,  # Use pruning subfolder
                    dpi=self.config.plot_dpi,
                    total_params=total_params,
                )
                for fig in figs:
                    plt.close(fig)

                # Create loss plots if available
                if results.get("before_losses") and results.get("after_losses"):
                    loss_figs = visualizer.plot_pruning_loss_before_after(
                        sparsities=results["sparsities"],
                        before_losses=results["before_losses"],
                        after_losses=results["after_losses"],
                        before_std=results.get("before_losses_std"),
                        after_std=results.get("after_losses_std"),
                        algorithm=algorithm.replace("_", " ").title(),
                        save_dir=pruning_dir,
                        dpi=self.config.plot_dpi,
                        total_params=total_params,
                    )
                    for fig in loss_figs:
                        plt.close(fig)

                # Create improvement plots for each selection mode
                for selection_mode in results["before"]:
                    if selection_mode in results["after"]:
                        suffix = f"_{selection_mode}" if len(results["before"]) > 1 else ""
                        fig = visualizer.plot_pruning_improvement(
                            sparsities=results["sparsities"],
                            before_accuracies=results["before"][selection_mode],
                            after_accuracies=results["after"][selection_mode],
                            algorithm=algorithm.replace("_", " ").title(),
                            selection_mode=selection_mode,
                            save_path=pruning_dir / f"pruning_{algorithm}_improvement{suffix}.png",
                            dpi=self.config.plot_dpi,
                        )
                        plt.close(fig)

            # Generate comparison plots
            try:
                # Prepare data for comparison plot (use lowercase underscore keys)
                comparison_data = {}
                for strategy_key, strategy_results in self.pruning_results.get("strategies", {}).items():
                    normalized_key = strategy_key.lower().replace(" ", "_")
                    comparison_data[normalized_key] = {
                        "sparsities": strategy_results.get("sparsities", strategy_results.get("pruning_amounts", [])),
                        "accuracies_before_finetune": strategy_results.get("accuracies_before_finetune", []),
                        "accuracies_after_finetune": strategy_results.get("accuracies_after_finetune", []),
                        "accuracies_std": strategy_results.get("accuracies_after_finetune_std"),
                    }
                
                if comparison_data:
                    # Comparison plot (save to pruning subfolder)
                    fig = visualizer.plot_pruning_comparison(
                        results=comparison_data,
                        metric="accuracy",
                        title=f"Pruning Strategy Comparison - {self.config.model_name}",
                        save_path=pruning_dir / "pruning_comparison.png",
                        total_params=total_params,
                    )
                    plt.close(fig)
                    
                    # Summary grid plot (save to pruning subfolder)
                    fig = visualizer.plot_pruning_summary_grid(
                        results=comparison_data,
                        save_path=pruning_dir / "pruning_summary_grid.png",
                    )
                    plt.close(fig)
                    
                    logger.info("Generated professional pruning comparison plots in pruning/ subfolder")
            except Exception as e:
                logger.warning(f"Could not generate comparison plots: {e}")

        logger.info(f"Saved visualizations to {output_dir}")

    def _cleanup_pruning_artifacts(self, model: nn.Module):
        """Clean up pruning artifacts from a model."""
        for name, module in model.named_modules():
            if hasattr(module, "weight_mask"):
                delattr(module, "weight_mask")
            if hasattr(module, "_pruning_hook"):
                module._pruning_hook.remove()
                delattr(module, "_pruning_hook")
            if hasattr(module, "_original_weight"):
                delattr(module, "_original_weight")
            if hasattr(module, "_gradient_hook_handle"):
                module._gradient_hook_handle.remove()
                delattr(module, "_gradient_hook_handle")

    def _create_pruning_strategy(self, strategy_name: str, pruning_config: PruningConfig):
        """Create a pruning strategy instance.
        
        Args:
            strategy_name: Name of the strategy. Can be:
                - A registered strategy ("magnitude", "random", "alignment")
                - A metric name ("rayleigh_quotient", "activation_l2_norm", etc.)
            pruning_config: Configuration for the pruning strategy
            
        Returns:
            Initialized pruning strategy or None if creation fails
        """
        from alignment.pruning import get_pruning_strategy
        from alignment.pruning.strategies import GlobalMagnitudePruning, GlobalAlignmentPruning

        try:
            # Handle global pruning variants
            if pruning_config.global_pruning:
                if strategy_name == "magnitude":
                    return GlobalMagnitudePruning(config=pruning_config)
            elif strategy_name == "alignment":
                alignment_metric = getattr(self.config, "pruning_alignment_metric", "rayleigh_quotient")
                return GlobalAlignmentPruning(metric=alignment_metric, config=pruning_config)
            
            # Use the registry function which handles both registered strategies and metrics
            return get_pruning_strategy(strategy_name, config=pruning_config)
            
        except Exception as e:
            logger.error(f"Error creating strategy {strategy_name}: {e}")
            return None

    def _pruning_experiments_single_network(self, model: nn.Module, wrapped_model: ModelWrapper, network_id: int) -> Dict[str, Any]:
        """Perform pruning experiments on a single specific network (fallback for compatibility)."""
        logger.info(f"Using single network pruning for network {network_id} (fallback mode)")

        # TODO: Implement a real single-network pruning run (strategy loop + finetune + eval),
        # matching the tensorized outputs structure, so callers don't get empty results.
        results = {"strategies": {}, "final_model_performance": {}, "network_id": network_id}

        # For now, return empty results - the tensorized version should handle everything
        return results

    def _evaluate_single_model(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate a specific model (fallback for compatibility)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        model.train()  # Reset to train mode

        avg_loss = total_loss / len(self.data_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _finetune_single_model(self, model: nn.Module) -> Tuple[float, float]:
        """Fine-tune a specific model (fallback for compatibility)."""
        # Setup optimizer for fine-tuning
        optimizer = torch.optim.Adam(model.parameters(), lr=getattr(self.config, "fine_tune_learning_rate", self.config.learning_rate * 0.1))
        criterion = nn.CrossEntropyLoss()

        # Fine-tuning loop
        for epoch in range(self.config.fine_tune_epochs):
            model.train()
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate after fine-tuning
        return self._evaluate_single_model(model)

    def _aggregate_dropout_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate dropout results from multiple networks (fallback for compatibility)."""
        # Import moved to top

        if not results:
            return {
                "dropout_rates": [],
                "accuracies": {"low": [], "high": [], "random": []},
                "losses": {"low": [], "high": [], "random": []},
                "alignment_values": {},
            }

        # Structure: same as single network but with averaged values
        aggregated = {
            "dropout_rates": results[0].get("dropout_rates", []),
            "accuracies": {"low": [], "high": [], "random": []},
            "losses": {"low": [], "high": [], "random": []},
            "alignment_values": {},
        }

        # Average accuracies and losses
        for strategy in ["low", "high", "random"]:
            for i in range(len(results[0].get("dropout_rates", []))):
                acc_values = [r["accuracies"][strategy][i] for r in results if "accuracies" in r and strategy in r["accuracies"]]
                loss_values = [r["losses"][strategy][i] for r in results if "losses" in r and strategy in r["losses"]]

                if acc_values:
                    aggregated["accuracies"][strategy].append(np.mean(acc_values))
                    aggregated["losses"][strategy].append(np.mean(loss_values))

        return aggregated

    def _apply_dropout_and_evaluate(
        self, dropout_rate: float, strategy: str, alignment_values: Dict[str, Dict[str, List[float]]]
    ) -> Tuple[float, float]:
        """Apply targeted dropout (low/high/random) based on alignment scores and evaluate."""
        if dropout_rate <= 0 or not alignment_values:
            return self._evaluate()

        metric_name = next((m for m, layers in alignment_values.items() if layers), None)
        if metric_name is None:
            logger.warning("No alignment values available for targeted dropout; running baseline evaluation.")
            return self._evaluate()

        saved_weights: Dict[str, torch.Tensor] = {}
        saved_biases: Dict[str, torch.Tensor] = {}
        modules = dict(self.model.named_modules())
        layer_scores = alignment_values[metric_name]

        try:
            for layer_name, scores in layer_scores.items():
                if layer_name not in modules or not scores:
                    continue

                module = modules[layer_name]
                if not hasattr(module, "weight"):
                    continue

                weight = module.weight
                num_units = weight.shape[0]
                if num_units == 0:
                    continue

                scores_tensor = torch.as_tensor(scores, device=weight.device, dtype=weight.dtype)
                if scores_tensor.numel() < num_units:
                    # Pad scores if needed
                    pad = num_units - scores_tensor.numel()
                    scores_tensor = torch.nn.functional.pad(scores_tensor, (0, pad), value=scores_tensor.mean().item())
                elif scores_tensor.numel() > num_units:
                    scores_tensor = scores_tensor[:num_units]

                count = max(1, int(round(dropout_rate * num_units)))
                if count >= num_units:
                    count = num_units - 1
                if count <= 0:
                    continue

                if strategy == "high":
                    _, indices = torch.topk(scores_tensor, count, largest=True)
                elif strategy == "low":
                    _, indices = torch.topk(scores_tensor, count, largest=False)
                else:  # random
                    perm = torch.randperm(num_units, device=weight.device)
                    indices = perm[:count]

                saved_weights[layer_name] = weight.detach().clone()
                if hasattr(module, "bias") and module.bias is not None:
                    saved_biases[layer_name] = module.bias.detach().clone()

                with torch.no_grad():
                    weight[indices] = 0
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias[indices] = 0

            loss, acc = self._evaluate()
        finally:
            for layer_name, tensor in saved_weights.items():
                if layer_name in modules and hasattr(modules[layer_name], "weight"):
                    with torch.no_grad():
                        modules[layer_name].weight.copy_(tensor)
            for layer_name, tensor in saved_biases.items():
                if layer_name in modules and hasattr(modules[layer_name], "bias") and modules[layer_name].bias is not None:
                    with torch.no_grad():
                        modules[layer_name].bias.copy_(tensor)

        return loss, acc

    def _pruning_experiments_multi_sequential(self) -> Dict[str, Any]:
        """Perform pruning experiments on multiple networks sequentially (original slow method)."""
        all_results = []

        # Run pruning experiments for each network independently
        for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
            logger.info(f"Pruning experiments for network {i+1}/{self.config.num_networks}")

            # Run experiments on this specific network
            result = self._pruning_experiments_single_network(net, wrapped_net, i)
            all_results.append(result)

        # Aggregate results
        if self.config.aggregate_metrics:
            return self._aggregate_pruning_results(all_results)
        else:
            return {"networks": all_results}

    def _dropout_analysis_multi_sequential(self) -> Dict[str, Any]:
        """Perform dropout analysis on multiple networks sequentially (original slow method)."""
        all_results = []

        # Run dropout analysis for each network
        for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
            logger.info(f"Dropout analysis for network {i+1}/{self.config.num_networks}")

            # Temporarily set as current model
            old_model = self.model
            old_wrapped = self.wrapped_model
            self.model = net
            self.wrapped_model = wrapped_net

            # Run analysis
            result = self._dropout_analysis_single()
            all_results.append(result)

        # Reset
        self.model = None
        self.wrapped_model = None

        # Aggregate results
        if self.config.aggregate_metrics:
            return self._aggregate_dropout_results(all_results)
        else:
            return {"networks": all_results}

    def _aggregate_pruning_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate pruning results from multiple networks."""
        # Import moved to top

        # Get structure from first result
        aggregated = {"strategies": {}, "final_model_performance": {}}

        # Aggregate each strategy
        if not results or not results[0].get("strategies"):
            return aggregated

        first_result = results[0]
        for strategy_key in first_result["strategies"]:
            strategy_data = first_result["strategies"][strategy_key]

            # Initialize aggregated strategy data
            agg_strategy = {
                "pruning_amounts": strategy_data.get("pruning_amounts", []),
                "accuracies_before_finetune": [],
                "losses_before_finetune": [],
                "accuracies_after_finetune": [],
                "losses_after_finetune": [],
                "sparsities": [],
            }

            # Average metrics across networks
            num_amounts = len(strategy_data.get("pruning_amounts", []))
            for i in range(num_amounts):
                # Collect values from all networks
                acc_before = [r["strategies"][strategy_key]["accuracies_before_finetune"][i] for r in results if strategy_key in r["strategies"]]
                loss_before = [r["strategies"][strategy_key]["losses_before_finetune"][i] for r in results if strategy_key in r["strategies"]]
                acc_after = [r["strategies"][strategy_key]["accuracies_after_finetune"][i] for r in results if strategy_key in r["strategies"]]
                loss_after = [r["strategies"][strategy_key]["losses_after_finetune"][i] for r in results if strategy_key in r["strategies"]]
                sparsity = [r["strategies"][strategy_key]["sparsities"][i] for r in results if strategy_key in r["strategies"]]

                # Average
                if acc_before:
                    agg_strategy["accuracies_before_finetune"].append(np.mean(acc_before))
                    agg_strategy["losses_before_finetune"].append(np.mean(loss_before))
                    agg_strategy["accuracies_after_finetune"].append(np.mean(acc_after))
                    agg_strategy["losses_after_finetune"].append(np.mean(loss_after))
                    agg_strategy["sparsities"].append(np.mean(sparsity))

            aggregated["strategies"][strategy_key] = agg_strategy

        return aggregated

    def _tensorized_pruning_batch_ultra_fast(self, strategy_name: str, selection_mode: str, pruning_amounts: List[float]) -> Dict[str, torch.Tensor]:
        """
        Ultra-fast version that processes all networks and pruning amounts truly in parallel.

        This method creates temporary copies of networks to avoid sequential processing.
        Trade-off: Uses more memory but is significantly faster.

        Returns tensors of shape [num_networks, num_pruning_amounts] for each metric.
        """
        num_networks = len(self.networks)
        num_amounts = len(pruning_amounts)

        logger.info(f"    Ultra-fast mode: Processing {num_networks} networks × {num_amounts} pruning amounts in parallel")

        # Initialize result tensors
        accuracies_before = torch.zeros(num_networks, num_amounts)
        losses_before = torch.zeros(num_networks, num_amounts)
        accuracies_after = torch.zeros(num_networks, num_amounts)
        losses_after = torch.zeros(num_networks, num_amounts)
        sparsities = torch.zeros(num_networks, num_amounts)

        # Create masks for all networks and all pruning amounts at once
        all_masks = self._create_tensorized_masks(strategy_name, selection_mode, pruning_amounts)

        # Create a batch of pruned network copies
        pruned_networks = []
        for net_idx in range(num_networks):
            network_copies = []
            for amount_idx in range(num_amounts):
                # Create a deep copy of the network
                import copy

                net_copy = copy.deepcopy(self.networks[net_idx])

                # Apply masks to the copy
                self._apply_tensorized_mask(net_copy, all_masks[net_idx][amount_idx])

                network_copies.append(net_copy)
            pruned_networks.append(network_copies)

        # Evaluate all network-pruning combinations in a single batch
        logger.info("    Evaluating all pruning configurations simultaneously...")

        # Batch evaluation before fine-tuning
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()

            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                # Process all networks and pruning amounts
                for net_idx in range(num_networks):
                    for amount_idx in range(num_amounts):
                        net = pruned_networks[net_idx][amount_idx]
                        net.eval()

                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        losses_before[net_idx, amount_idx] += loss.item()

                        _, predicted = outputs.max(1)
                        accuracies_before[net_idx, amount_idx] += predicted.eq(targets).sum().item()

        # Normalize results
        losses_before /= len(self.data_loader)
        accuracies_before = accuracies_before * 100.0 / len(self.data_loader.dataset)

        # Calculate sparsities
        for net_idx in range(num_networks):
            for amount_idx in range(num_amounts):
                sparsities[net_idx, amount_idx] = self._calculate_model_sparsity(pruned_networks[net_idx][amount_idx])

        logger.info(f"    Overall before fine-tuning - Avg Acc: {accuracies_before.mean():.2f}%, Avg Sparsity: {sparsities.mean():.2%}")

        # Fine-tuning phase if requested
        if self.config.fine_tune_after_pruning:
            logger.info(f"    Fine-tuning all {num_networks * num_amounts} configurations in parallel...")

            # Create optimizers for all network copies
            all_optimizers = []
            for net_idx in range(num_networks):
                for amount_idx in range(num_amounts):
                    net = pruned_networks[net_idx][amount_idx]
                    optimizer = torch.optim.Adam(
                        net.parameters(), lr=getattr(self.config, "fine_tune_learning_rate", self.config.learning_rate * 0.1)
                    )
                    all_optimizers.append(optimizer)

            # Fine-tune all networks simultaneously
            for epoch in range(self.config.fine_tune_epochs):
                for inputs, targets in self.data_loader:
                    inputs = inputs.to(self.config.device)
                    targets = targets.to(self.config.device)

                    # Train all network copies
                    opt_idx = 0
                    for net_idx in range(num_networks):
                        for amount_idx in range(num_amounts):
                            net = pruned_networks[net_idx][amount_idx]
                            optimizer = all_optimizers[opt_idx]
                            opt_idx += 1

                            net.train()
                            optimizer.zero_grad()

                            outputs = net(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()

                            # Re-apply masks after update
                            self._reapply_masks_after_update(net)

            # Evaluate after fine-tuning
            logger.info("    Evaluating all fine-tuned configurations...")
            with torch.no_grad():
                for inputs, targets in self.data_loader:
                    inputs = inputs.to(self.config.device)
                    targets = targets.to(self.config.device)

                    for net_idx in range(num_networks):
                        for amount_idx in range(num_amounts):
                            net = pruned_networks[net_idx][amount_idx]
                            net.eval()

                            outputs = net(inputs)
                            loss = criterion(outputs, targets)
                            losses_after[net_idx, amount_idx] += loss.item()

                            _, predicted = outputs.max(1)
                            accuracies_after[net_idx, amount_idx] += predicted.eq(targets).sum().item()

            # Normalize results
            losses_after /= len(self.data_loader)
            accuracies_after = accuracies_after * 100.0 / len(self.data_loader.dataset)

            logger.info(
                f"    Overall after fine-tuning - Avg Acc: {accuracies_after.mean():.2f}%, Avg Improvement: {(accuracies_after - accuracies_before).mean():+.2f}%"
            )
        else:
            # No fine-tuning
            accuracies_after = accuracies_before.clone()
            losses_after = losses_before.clone()

        # Clean up - delete temporary network copies to free memory
        del pruned_networks
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "accuracies_before": accuracies_before,
            "losses_before": losses_before,
            "accuracies_after": accuracies_after,
            "losses_after": losses_after,
            "sparsities": sparsities,
        }

    def _tensorized_pruning_batch_optimized(self, strategy_name: str, selection_mode: str, pruning_amounts: List[float]) -> Dict[str, torch.Tensor]:
        """
        Optimized version of tensorized pruning that reduces redundant operations.

        Key optimizations:
        1. Batch mask application across networks
        2. Vectorized sparsity calculation
        3. Reduced model state copying
        4. Parallel evaluation when possible
        """
        num_networks = len(self.networks)
        num_amounts = len(pruning_amounts)

        logger.info(f"    [Optimized] Processing {len(pruning_amounts)} pruning amounts for {num_networks} networks")

        # Initialize result tensors
        accuracies_before = torch.zeros(num_networks, num_amounts)
        losses_before = torch.zeros(num_networks, num_amounts)
        accuracies_after = torch.zeros(num_networks, num_amounts)
        losses_after = torch.zeros(num_networks, num_amounts)
        sparsities = torch.zeros(num_networks, num_amounts)

        # Save original weights more efficiently - only save non-zero tensors
        original_weights = []
        for model in self.networks:
            model_weights = {}
            for name, module in model.named_modules():
                if hasattr(module, "weight"):
                    # Use in-place operations where possible
                    model_weights[name] = module.weight.data
            original_weights.append(model_weights)

        # Pre-create all masks at once for better memory locality
        logger.info("    Creating all masks...")
        all_masks = self._create_tensorized_masks(strategy_name, selection_mode, pruning_amounts)

        # Batch evaluation - process multiple networks at once when possible
        logger.info("    Evaluating pruned networks...")

        # For each pruning amount, apply masks to all networks at once
        for amount_idx, amount in enumerate(pruning_amounts):
            # Apply masks to all networks in parallel
            for net_idx, model in enumerate(self.networks):
                # Direct weight assignment (faster than dict copy)
                for name, module in model.named_modules():
                    if name in original_weights[net_idx]:
                        module.weight.data = original_weights[net_idx][name].clone()

                # Apply mask
                self._apply_tensorized_mask_fast(model, all_masks[net_idx][amount_idx])

                # Calculate sparsity inline
                total_params = 0
                zero_params = 0
                for module in model.modules():
                    if hasattr(module, "weight"):
                        weight = module.weight
                        total_params += weight.numel()
                        zero_params += (weight == 0).sum().item()
                sparsities[net_idx, amount_idx] = zero_params / total_params if total_params > 0 else 0.0

            # Batch evaluate all networks
            batch_losses_before, batch_accs_before = self._evaluate_networks_batch_optimized()
            accuracies_before[:, amount_idx] = batch_accs_before
            losses_before[:, amount_idx] = batch_losses_before

        logger.info(f"    Evaluation complete. Avg accuracy: {accuracies_before.mean():.2f}%")

        # Fine-tuning phase
        if self.config.fine_tune_after_pruning:
            logger.info("    Fine-tuning pruned networks...")

            # Pre-create all optimizers to avoid recreation
            base_lr = getattr(self.config, "fine_tune_learning_rate", self.config.learning_rate * 0.1)

            for amount_idx, amount in enumerate(pruning_amounts):
                # Restore and apply masks
                for net_idx, model in enumerate(self.networks):
                    # Fast weight restoration
                    for name, module in model.named_modules():
                        if name in original_weights[net_idx]:
                            module.weight.data = original_weights[net_idx][name].clone()
                    self._apply_tensorized_mask_fast(model, all_masks[net_idx][amount_idx])

                # Fine-tune with optimized batch processing
                self._finetune_networks_batch_optimized(epochs=self.config.fine_tune_epochs, lr=base_lr)

                # Evaluate
                batch_losses_after, batch_accs_after = self._evaluate_networks_batch_optimized()
                accuracies_after[:, amount_idx] = batch_accs_after
                losses_after[:, amount_idx] = batch_losses_after
        else:
            accuracies_after = accuracies_before.clone()
            losses_after = losses_before.clone()

        # Final cleanup - restore original weights
        for net_idx, model in enumerate(self.networks):
            for name, module in model.named_modules():
                if name in original_weights[net_idx]:
                    module.weight.data = original_weights[net_idx][name]
            self._cleanup_pruning_artifacts(model)

        return {
            "accuracies_before": accuracies_before,
            "losses_before": losses_before,
            "accuracies_after": accuracies_after,
            "losses_after": losses_after,
            "sparsities": sparsities,
        }

    def _apply_tensorized_mask_fast(self, model: nn.Module, layer_masks: Dict[str, torch.Tensor]):
        """Fast version of mask application without unnecessary copies."""
        for name, module in model.named_modules():
            if name in layer_masks:
                mask = layer_masks[name]
                # Apply mask directly without storing original
                module.weight.data.mul_(mask)
                # Simple forward hook for maintaining mask
                if not hasattr(module, "_mask_hook"):

                    def make_hook(m):
                        def hook(mod, inp, out):
                            return out

                        return hook

                    module._mask_hook = module.register_forward_hook(make_hook(mask))

    def _evaluate_networks_batch_optimized(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized batch evaluation with better memory usage."""
        num_networks = len(self.networks)
        total_loss = torch.zeros(num_networks, device=self.config.device)
        total_correct = torch.zeros(num_networks, device=self.config.device)
        total_samples = 0
        num_batches = 0

        # Set eval mode for all
        for net in self.networks:
            net.eval()

        criterion = nn.CrossEntropyLoss(reduction="none")
        eval_batches = self.config.eval_batches  # Limit evaluation batches (None = all)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                if eval_batches is not None and batch_idx >= eval_batches:
                    break
                    
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                batch_size = targets.size(0)

                # Process all networks in one go if memory allows
                for i, net in enumerate(self.networks):
                    outputs = net(inputs)
                    losses = criterion(outputs, targets)
                    total_loss[i] += losses.sum()

                    preds = outputs.argmax(dim=1)
                    total_correct[i] += preds.eq(targets).sum()

                total_samples += batch_size
                num_batches += 1

        # Reset to train mode
        for net in self.networks:
            net.train()

        # Convert to CPU for return
        avg_losses = (total_loss / max(num_batches, 1)).cpu()
        avg_accs = (total_correct * 100.0 / max(total_samples, 1)).cpu()

        return avg_losses, avg_accs

    def _finetune_networks_batch_optimized(self, epochs: int, lr: float):
        """Optimized fine-tuning with shared computation where possible."""
        # Create optimizers
        optimizers = []
        for net in self.networks:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            optimizers.append(optimizer)

        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                # Process each network
                for net, opt in zip(self.networks, optimizers):
                    net.train()
                    opt.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    opt.step()

    def _evaluate_networks_batch_ultra_parallel(self, network_configs: List[Tuple[nn.Module, Dict[str, torch.Tensor]]]) -> torch.Tensor:
        """
        Ultra-parallel evaluation that processes all networks and configurations in a single pass.

        Args:
            network_configs: List of (network, mask_dict) tuples where mask_dict contains
                            masks to apply for each layer

        Returns:
            Tensor of shape [num_configs] containing accuracy for each configuration
        """
        num_configs = len(network_configs)

        # Pre-allocate GPU tensors for all results
        total_correct = torch.zeros(num_configs, device=self.config.device)
        total_samples = 0
        
        eval_batches = self.config.eval_batches  # Limit evaluation batches (None = all)

        # Set all networks to eval mode
        for net, _ in network_configs:
            net.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                if eval_batches is not None and batch_idx >= eval_batches:
                    break
                    
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                batch_size = targets.size(0)

                # Stack all network outputs into a single tensor for parallel processing
                # Shape: [num_configs, batch_size, num_classes]
                all_outputs = []

                for net, masks in network_configs:
                    # Apply masks if provided
                    if masks:
                        self._apply_masks_fast(net, masks)

                    # Forward pass
                    outputs = net(inputs)
                    all_outputs.append(outputs)

                # Stack outputs for parallel processing
                stacked_outputs = torch.stack(all_outputs, dim=0)

                # Get predictions for all configs at once
                # Shape: [num_configs, batch_size]
                all_preds = stacked_outputs.argmax(dim=2)

                # Expand targets for comparison
                # Shape: [num_configs, batch_size]
                expanded_targets = targets.unsqueeze(0).expand(num_configs, -1)

                # Compute correct predictions for all configs
                # Shape: [num_configs]
                correct = all_preds.eq(expanded_targets).sum(dim=1)
                total_correct += correct

                total_samples += batch_size

        # Convert to accuracy percentages
        accuracies = (total_correct * 100.0 / max(total_samples, 1)).cpu()

        # Reset networks to train mode
        for net, _ in network_configs:
            net.train()

        return accuracies

    def _evaluate_all_pruning_configs_parallel(
        self, all_masks: List[List[Dict[str, torch.Tensor]]], original_states: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Evaluate all pruning configurations (networks × sparsity levels) in parallel.

        Args:
            all_masks: Masks for each network and pruning amount
                      Shape: [num_networks][num_amounts][layer_name -> mask]
            original_states: Original weights for each network

        Returns:
            Tensor of shape [num_networks, num_amounts] with accuracies
        """
        num_networks = len(self.networks)
        num_amounts = len(all_masks[0])

        # Create all network configurations
        network_configs = []

        for net_idx in range(num_networks):
            for amount_idx in range(num_amounts):
                # Create a temporary copy with applied masks
                net_copy = copy.deepcopy(self.networks[net_idx])

                # Restore original weights
                for name, module in net_copy.named_modules():
                    if name in original_states[net_idx]:
                        module.weight.data = original_states[net_idx][name].clone()

                # Prepare mask dict
                masks = all_masks[net_idx][amount_idx]

                network_configs.append((net_copy, masks))

        # Evaluate all configurations in parallel
        all_accuracies = self._evaluate_networks_batch_ultra_parallel(network_configs)

        # Reshape results
        accuracies = all_accuracies.view(num_networks, num_amounts)

        # Clean up copies
        del network_configs
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

        return accuracies

    def _apply_masks_fast(self, model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Fast mask application without hooks or permanent modifications."""
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in masks and hasattr(module, "weight"):
                    module.weight.data *= masks[name]

    def _apply_tensorized_mask_fast(self, model: nn.Module, masks: Dict[str, torch.Tensor]):
        """Optimized version of mask application."""
        self._apply_masks_fast(model, masks)

    def _pruning_experiments_tensorized_v2(self) -> Dict[str, Any]:
        """
        Version 2: Truly parallel pruning experiments with maximum efficiency.

        Key improvements:
        1. Single forward pass per batch for ALL configurations
        2. Tensor operations for mask application
        3. Minimal memory copying
        4. Parallel sparsity calculation
        """
        # Imports moved to top

        results = {"strategies": {}}

        # Get selection modes to test
        selection_modes = self.config.pruning_selection_mode
        if not isinstance(selection_modes, list):
            selection_modes = [selection_modes]

        # Save original states efficiently
        original_states = []
        for model in self.networks:
            state = {name: module.weight.data.clone() for name, module in model.named_modules() if hasattr(module, "weight")}
            original_states.append(state)

        # Process each strategy
        for strategy_name in self.config.pruning_strategies:
            logger.info(f"Testing pruning strategy: {strategy_name}")

            for selection_mode in selection_modes:
                # Use the ultra-parallel evaluation
                if strategy_name == "alignment" and hasattr(self.config, "use_ultra_parallel_eval") and self.config.use_ultra_parallel_eval:
                    batch_results = self._tensorized_pruning_ultra_parallel(strategy_name, selection_mode, self.config.pruning_amounts)
                else:
                    # Existing implementation
                    batch_results = self._tensorized_pruning_batch(strategy_name, selection_mode, self.config.pruning_amounts)

                # Store results
                strategy_key = f"{strategy_name}_{selection_mode}"
                strategy_results = {
                    "sparsities": batch_results["sparsities"].mean(dim=0).tolist(),
                    "accuracies_before_finetune": batch_results["accuracies_before"].mean(dim=0).tolist(),
                    "accuracies_after_finetune": batch_results["accuracies_after"].mean(dim=0).tolist(),
                    "losses_before_finetune": batch_results["losses_before"].mean(dim=0).tolist(),
                    "losses_after_finetune": batch_results["losses_after"].mean(dim=0).tolist(),
                    "improvements": (batch_results["accuracies_after"] - batch_results["accuracies_before"]).mean(dim=0).tolist(),
                }

                # Add standard deviations if multiple networks
                if self.config.num_networks > 1:
                    strategy_results["accuracies_before_finetune_std"] = batch_results["accuracies_before"].std(dim=0).tolist()
                    strategy_results["accuracies_after_finetune_std"] = batch_results["accuracies_after"].std(dim=0).tolist()

                results["strategies"][strategy_key] = strategy_results

        # Restore original weights
        for net_idx, model in enumerate(self.networks):
            for name, module in model.named_modules():
                if name in original_states[net_idx]:
                    module.weight.data = original_states[net_idx][name]

        return results

    def _tensorized_pruning_ultra_parallel(self, strategy_name: str, selection_mode: str, pruning_amounts: List[float]) -> Dict[str, torch.Tensor]:
        """
        Ultra-parallel pruning that evaluates ALL configurations in minimal passes.

        This method:
        1. Creates all masks upfront
        2. Evaluates all network×sparsity combinations in parallel
        3. Optionally fine-tunes in parallel (if memory allows)
        """
        num_networks = len(self.networks)
        num_amounts = len(pruning_amounts)
        total_configs = num_networks * num_amounts

        logger.info(f"    [Ultra-Parallel] Processing {total_configs} configurations in parallel")
        logger.info(f"    Networks: {num_networks}, Sparsity levels: {num_amounts}")

        # Save original states
        original_states = []
        for model in self.networks:
            state = {name: module.weight.data.clone() for name, module in model.named_modules() if hasattr(module, "weight")}
            original_states.append(state)

        # Create all masks using alignment-based scoring
        logger.info("    Creating masks for all configurations...")
        all_masks = self._create_alignment_masks_batch(selection_mode, pruning_amounts)

        # Calculate sparsities
        sparsities = torch.zeros(num_networks, num_amounts)
        for net_idx in range(num_networks):
            for amount_idx, amount in enumerate(pruning_amounts):
                # TODO: Compute actual sparsity from the masks (not the requested pruning amount),
                # especially if mask construction has ties/constraints that affect the achieved rate.
                sparsities[net_idx, amount_idx] = amount

        # TRULY PARALLEL EVALUATION - all configs at once!
        logger.info("    Starting TRULY PARALLEL evaluation of all configurations...")
        start_time = time.time()

        accuracies_before, losses_before = self._evaluate_all_configs_truly_parallel(self.networks, all_masks, original_states)

        eval_time = time.time() - start_time
        logger.info(f"    Parallel evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"    Average accuracy before pruning: {accuracies_before.mean():.2f}%")

        # Fine-tuning phase
        if self.config.fine_tune_after_pruning:
            # TODO: Implement fine-tuning for ultra-parallel mode (e.g., batched finetune or per-config micro-finetune),
            # or explicitly disable this mode when fine_tune_after_pruning=True.
            logger.info("    Fine-tuning is not yet implemented for ultra-parallel mode")
            # For now, just copy the before results
            accuracies_after = accuracies_before.clone()
            losses_after = losses_before.clone()
        else:
            accuracies_after = accuracies_before.clone()
            losses_after = losses_before.clone()

        return {
            "accuracies_before": accuracies_before,
            "losses_before": losses_before,
            "accuracies_after": accuracies_after,
            "losses_after": losses_after,
            "sparsities": sparsities,
        }

    def _create_alignment_masks_batch(self, selection_mode: str, pruning_amounts: List[float]) -> List[List[Dict[str, torch.Tensor]]]:
        """
        Create alignment-based masks for all networks and pruning amounts in batch.

        Returns:
            List[network][amount][layer_name -> mask tensor]
        """
        num_networks = len(self.networks)
        num_amounts = len(pruning_amounts)

        # Get sample inputs for alignment computation
        sample_inputs, _ = next(iter(self.data_loader))
        sample_inputs = sample_inputs.to(self.config.device)

        # Pre-compute alignment scores for all networks
        all_alignment_scores = []

        for net_idx, model in enumerate(self.networks):
            # Capture layer inputs
            layer_inputs = self._capture_layer_inputs(model, sample_inputs)

            # Compute alignment scores for each layer
            layer_scores = {}
            for layer_name, module in model.named_modules():
                if hasattr(module, "weight") and layer_name in layer_inputs:
                    # Compute alignment scores
                    scores = self._compute_neuron_alignment_importance(module, layer_inputs[layer_name])
                    layer_scores[layer_name] = scores

            all_alignment_scores.append(layer_scores)

        # Create masks for all configurations
        all_masks = []

        for net_idx in range(num_networks):
            network_masks = []

            for amount in pruning_amounts:
                layer_masks = {}

                for layer_name, scores in all_alignment_scores[net_idx].items():
                    # Create mask based on selection mode
                    num_neurons = scores.shape[0]
                    num_to_prune = int(amount * num_neurons)

                    if num_to_prune == 0:
                        # No pruning
                        mask = torch.ones_like(scores)
                    elif selection_mode == "low":
                        # Prune neurons with lowest alignment
                        _, indices = scores.sort()
                        mask = torch.ones_like(scores)
                        mask[indices[:num_to_prune]] = 0
                    elif selection_mode == "high":
                        # Prune neurons with highest alignment
                        _, indices = scores.sort(descending=True)
                        mask = torch.ones_like(scores)
                        mask[indices[:num_to_prune]] = 0
                    elif selection_mode == "random":
                        # Random pruning
                        mask = torch.ones_like(scores)
                        perm = torch.randperm(num_neurons)
                        mask[perm[:num_to_prune]] = 0

                    layer_masks[layer_name] = mask

                network_masks.append(layer_masks)

            all_masks.append(network_masks)

        return all_masks

    def _evaluate_all_configs_truly_parallel(
        self, networks: List[nn.Module], all_masks: List[List[Dict[str, torch.Tensor]]], original_states: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Truly parallel evaluation of ALL configurations in a single forward pass per batch.

        Returns:
            accuracies: Tensor of shape [num_networks, num_pruning_amounts]
            losses: Tensor of shape [num_networks, num_pruning_amounts]
        """
        num_networks = len(networks)
        num_amounts = len(all_masks[0])
        total_configs = num_networks * num_amounts

        logger.info(f"    Evaluating {total_configs} configurations in TRUE parallel...")

        # Pre-allocate result tensors on GPU
        all_correct = torch.zeros(total_configs, device=self.config.device)
        all_loss = torch.zeros(total_configs, device=self.config.device)
        total_samples = 0

        # Prepare all network states upfront
        config_states = []
        config_idx = 0

        for net_idx in range(num_networks):
            for amount_idx in range(num_amounts):
                # Save the configuration mapping
                config_states.append(
                    {
                        "net_idx": net_idx,
                        "amount_idx": amount_idx,
                        "original_state": original_states[net_idx],
                        "masks": all_masks[net_idx][amount_idx],
                    }
                )
                config_idx += 1

        # Set all networks to eval mode
        for net in networks:
            net.eval()

        # Process batches
        criterion = nn.CrossEntropyLoss(reduction="none")
        eval_batches = getattr(self.config, "eval_batches", None)
        batch_count = 0

        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                batch_size = targets.size(0)

                # Collect outputs from all configurations
                all_outputs = []

                for config_idx, config in enumerate(config_states):
                    net = networks[config["net_idx"]]

                    # Apply configuration (weights and masks)
                    for name, module in net.named_modules():
                        if name in config["original_state"]:
                            module.weight.data = config["original_state"][name].clone()
                        if name in config["masks"] and hasattr(module, "weight"):
                            mask = config["masks"][name]
                            # Handle structured pruning - mask is per neuron
                            if mask.dim() == 1 and module.weight.dim() == 2:
                                # Expand mask to match weight dimensions
                                # For Linear layers: weight is [out_features, in_features]
                                # Mask is [out_features], so expand along dim 1
                                mask = mask.unsqueeze(1).expand_as(module.weight)
                            module.weight.data *= mask

                    # Forward pass
                    outputs = net(inputs)
                    all_outputs.append(outputs)

                # Stack all outputs for parallel processing
                # Shape: [total_configs, batch_size, num_classes]
                stacked_outputs = torch.stack(all_outputs, dim=0)

                # Compute losses for all configs at once
                # Expand targets to match
                expanded_targets = targets.unsqueeze(0).expand(total_configs, -1)
                losses = criterion(stacked_outputs.view(-1, stacked_outputs.size(-1)), expanded_targets.reshape(-1))
                losses = losses.view(total_configs, batch_size).sum(dim=1)
                all_loss += losses

                # Compute predictions
                predictions = stacked_outputs.argmax(dim=2)  # [total_configs, batch_size]
                correct = predictions.eq(expanded_targets).sum(dim=1)  # [total_configs]
                all_correct += correct.float()

                total_samples += batch_size
                batch_count += 1

                if eval_batches is not None and batch_count >= eval_batches:
                    break

        # Average results
        num_batches = batch_count
        all_loss /= num_batches
        all_accuracy = all_correct * 100.0 / total_samples

        # Reshape to [num_networks, num_amounts]
        accuracies = all_accuracy.view(num_networks, num_amounts).cpu()
        losses = all_loss.view(num_networks, num_amounts).cpu()

        # Reset networks to train mode
        for net in networks:
            net.train()

        # Restore original weights
        for net_idx, net in enumerate(networks):
            for name, module in net.named_modules():
                if name in original_states[net_idx]:
                    module.weight.data = original_states[net_idx][name]

        return accuracies, losses
