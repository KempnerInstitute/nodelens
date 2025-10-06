"""
Experiment-specific trainer that extends BaseTrainer.

This module provides training functionality specifically designed for
alignment experiments, with support for multi-network training.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentTrainingConfig(TrainingConfig):
    """Extended training configuration for experiments."""
    num_networks: int = 1  # Support for multi-network training
    tensorized: bool = True  # Use tensorized training when num_networks <= 8
    save_all_networks: bool = False  # Save all networks or just first
    metric_aggregation: str = "mean"  # How to aggregate metrics across networks


class ExperimentTrainer(BaseTrainer):
    """
    Trainer for alignment experiments with multi-network support.

    This trainer extends BaseTrainer with:
    - Support for training multiple networks simultaneously
    - Tensorized training for efficiency
    - Experiment-specific metrics and logging
    """

    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        config: Optional[ExperimentTrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize experiment trainer.

        Args:
            model: Single model or list of models for multi-network training
            config: Training configuration
            loss_fn: Loss function
            callbacks: List of callback functions
        """
        # Handle multi-network case
        if isinstance(model, list):
            self.models = model
            self.num_networks = len(model)
            # Use first model for base class
            super().__init__(model[0], config, loss_fn, callbacks)
        else:
            self.models = [model]
            self.num_networks = 1
            super().__init__(model, config, loss_fn, callbacks)

        # Override config if needed
        if config and hasattr(config, 'num_networks'):
            assert config.num_networks == self.num_networks, \
                f"Config num_networks ({config.num_networks}) != actual ({self.num_networks})"

        # Setup optimizers for all models
        if self.num_networks > 1:
            self.optimizers = [self._create_optimizer_for_model(m) for m in self.models]
            self.schedulers = [self._create_scheduler_for_optimizer(opt) for opt in self.optimizers]
        else:
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler] if self.scheduler else [None]

        # Move all models to device
        for model in self.models:
            model.to(self.device)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model(s).

        Returns:
            Training history with support for multiple networks
        """
        logger.info(f"Starting training for {self.config.epochs} epochs with {self.num_networks} network(s)")

        # Initialize per-network history if multiple networks
        if self.num_networks > 1:
            self.history['per_network'] = {
                i: {
                    'train_loss': [],
                    'train_metrics': [],
                    'val_loss': [],
                    'val_metrics': []
                }
                for i in range(self.num_networks)
            }

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training phase
            if self.num_networks > 1 and self.config.tensorized and self.num_networks <= 8:
                # Use tensorized training for efficiency
                train_losses, train_metrics = self._train_epoch_tensorized(
                    train_loader, metric_fn
                )
            else:
                # Train each network separately
                train_losses, train_metrics = self._train_epoch_multi(
                    train_loader, metric_fn
                )

            # Aggregate metrics
            avg_train_loss = self._aggregate_metric(train_losses)
            avg_train_metrics = self._aggregate_metrics_dict(train_metrics)

            # Validation phase
            val_losses, val_metrics = [0.0] * self.num_networks, [{} for _ in range(self.num_networks)]
            if val_loader and (epoch + 1) % self.config.eval_interval == 0:
                if self.num_networks > 1 and self.config.tensorized and self.num_networks <= 8:
                    val_losses, val_metrics = self._validate_tensorized(
                        val_loader, metric_fn
                    )
                else:
                    val_losses, val_metrics = self._validate_multi(
                        val_loader, metric_fn
                    )

                # Early stopping check on average validation loss
                avg_val_loss = self._aggregate_metric(val_losses)
                if self._should_stop_early(avg_val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Update schedulers
            for i, scheduler in enumerate(self.schedulers):
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_losses[i] if val_losses[i] > 0 else train_losses[i])
                    else:
                        scheduler.step()

            # Record history
            epoch_time = time.time() - epoch_start
            avg_val_loss = self._aggregate_metric(val_losses)
            avg_val_metrics = self._aggregate_metrics_dict(val_metrics)

            self.history['train_loss'].append(avg_train_loss)
            self.history['train_metrics'].append(avg_train_metrics)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_metrics'].append(avg_val_metrics)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(
                self.optimizers[0].param_groups[0]['lr']
            )

            # Record per-network history
            if self.num_networks > 1:
                for i in range(self.num_networks):
                    self.history['per_network'][i]['train_loss'].append(train_losses[i])
                    self.history['per_network'][i]['train_metrics'].append(train_metrics[i])
                    self.history['per_network'][i]['val_loss'].append(val_losses[i])
                    self.history['per_network'][i]['val_metrics'].append(val_metrics[i])

            # Log progress
            self._log_epoch_progress(
                epoch, avg_train_loss, avg_train_metrics,
                avg_val_loss, avg_val_metrics, epoch_time
            )

            # Save checkpoint
            if self.config.checkpoint_dir and (epoch + 1) % self.config.eval_interval == 0:
                self._save_checkpoint_multi(epoch)

            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch)

        return self.history

    def _train_epoch_multi(
        self,
        train_loader: torch.utils.data.DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Train multiple networks for one epoch."""
        all_losses = []
        all_metrics = []

        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            total_loss = 0.0
            metrics_list = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                if self.config.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_val
                    )

                optimizer.step()

                total_loss += loss.item()
                if metric_fn:
                    metrics = metric_fn(outputs, targets)
                    metrics_list.append(metrics)

            avg_loss = total_loss / len(train_loader)
            avg_metrics = self._average_metrics(metrics_list) if metrics_list else {}

            all_losses.append(avg_loss)
            all_metrics.append(avg_metrics)

        return all_losses, all_metrics

    def _train_epoch_tensorized(
        self,
        train_loader: torch.utils.data.DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Train multiple networks using tensorized operations."""
        # Set all models to train mode
        for model in self.models:
            model.train()

        total_losses = [0.0] * self.num_networks
        all_metrics = [[] for _ in range(self.num_networks)]

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Process each network separately (simpler and more robust)
            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                if self.config.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_val
                    )

                optimizer.step()

                total_losses[i] += loss.item()
                if metric_fn:
                    metrics = metric_fn(outputs, targets)
                    all_metrics[i].append(metrics)

        # Average losses and metrics
        avg_losses = [total / len(train_loader) for total in total_losses]
        avg_metrics = [
            self._average_metrics(metrics) if metrics else {}
            for metrics in all_metrics
        ]

        return avg_losses, avg_metrics

    def _validate_multi(
        self,
        val_loader: torch.utils.data.DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Validate multiple networks."""
        all_losses = []
        all_metrics = []

        for model in self.models:
            model.eval()
            total_loss = 0.0
            metrics_list = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = model(inputs)
                    loss = self.loss_fn(outputs, targets)

                    total_loss += loss.item()
                    if metric_fn:
                        metrics = metric_fn(outputs, targets)
                        metrics_list.append(metrics)

            avg_loss = total_loss / len(val_loader)
            avg_metrics = self._average_metrics(metrics_list) if metrics_list else {}

            all_losses.append(avg_loss)
            all_metrics.append(avg_metrics)

        return all_losses, all_metrics

    def _validate_tensorized(
        self,
        val_loader: torch.utils.data.DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Validate multiple networks using tensorized operations."""
        for model in self.models:
            model.eval()

        total_losses = [0.0] * self.num_networks
        all_metrics = [[] for _ in range(self.num_networks)]

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                inputs.shape[0]

                # Process each network
                for i, model in enumerate(self.models):
                    outputs = model(inputs)
                    loss = self.loss_fn(outputs, targets)

                    total_losses[i] += loss.item()
                    if metric_fn:
                        metrics = metric_fn(outputs, targets)
                        all_metrics[i].append(metrics)

        avg_losses = [total / len(val_loader) for total in total_losses]
        avg_metrics = [
            self._average_metrics(metrics) if metrics else {}
            for metrics in all_metrics
        ]

        return avg_losses, avg_metrics

    def _aggregate_metric(self, values: List[float]) -> float:
        """Aggregate a metric across networks."""
        if self.config.metric_aggregation == "mean":
            return sum(values) / len(values)
        elif self.config.metric_aggregation == "min":
            return min(values)
        elif self.config.metric_aggregation == "max":
            return max(values)
        else:
            raise ValueError(f"Unknown aggregation: {self.config.metric_aggregation}")

    def _aggregate_metrics_dict(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate dictionary of metrics across networks."""
        if not metrics_list or not metrics_list[0]:
            return {}

        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            aggregated[key] = self._aggregate_metric(values)

        return aggregated

    def _create_optimizer_for_model(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for a specific model."""
        optimizer_kwargs = self.config.optimizer_kwargs or {}

        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler_for_optimizer(
        self,
        optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create scheduler for a specific optimizer."""
        if not self.config.scheduler:
            return None

        scheduler_kwargs = self.config.scheduler_kwargs or {}

        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                **scheduler_kwargs
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_kwargs
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                **scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _save_checkpoint_multi(self, epoch: int):
        """Save checkpoint for multiple networks."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_all_networks and self.num_networks > 1:
            # Save each network separately
            for i, (model, optimizer, scheduler) in enumerate(
                zip(self.models, self.optimizers, self.schedulers)
            ):
                checkpoint = {
                    'epoch': epoch,
                    'network_id': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'history': self.history['per_network'][i],
                    'config': self.config,
                }

                checkpoint_path = checkpoint_dir / f"checkpoint_network_{i}_epoch_{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
        else:
            # Save only first network
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.models[0].state_dict(),
                'optimizer_state_dict': self.optimizers[0].state_dict(),
                'scheduler_state_dict': self.schedulers[0].state_dict() if self.schedulers[0] else None,
                'history': self.history,
                'config': self.config,
                'best_val_metric': self.best_val_metric,
                'num_networks': self.num_networks
            }

            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

        logger.debug(f"Saved checkpoint(s) for epoch {epoch}")
