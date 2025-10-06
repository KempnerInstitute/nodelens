"""
Base training utilities for neural networks.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    scheduler: Optional[str] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None
    device: str = "cuda"
    log_interval: int = 100
    eval_interval: int = 1
    checkpoint_dir: Optional[Path] = None
    early_stopping_patience: Optional[int] = None
    gradient_clip_val: Optional[float] = None


class BaseTrainer:
    """
    Base trainer class for neural network training.

    This class provides standard training functionality that can be
    extended for specific training strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            loss_fn: Loss function (default: CrossEntropyLoss)
            callbacks: List of callback functions
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.callbacks = callbacks or []

        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0

        # History
        self.history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': [],
            'epoch_times': [],
            'learning_rates': []
        }

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            metric_fn: Optional metric function

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training phase
            train_loss, train_metrics = self._train_epoch(
                train_loader, metric_fn
            )

            # Validation phase
            val_loss, val_metrics = 0.0, {}
            if val_loader and (epoch + 1) % self.config.eval_interval == 0:
                val_loss, val_metrics = self._validate(
                    val_loader, metric_fn
                )

                # Early stopping check
                if self._should_stop_early(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Log progress
            self._log_epoch_progress(
                epoch, train_loss, train_metrics, val_loss, val_metrics, epoch_time
            )

            # Save checkpoint
            if self.config.checkpoint_dir and (epoch + 1) % self.config.eval_interval == 0:
                self._save_checkpoint(epoch)

            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch)

        return self.history

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_metrics = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            if metric_fn:
                metrics = metric_fn(outputs, targets)
                all_metrics.append(metrics)

            # Logging
            if batch_idx % self.config.log_interval == 0:
                logger.debug(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

            self.global_step += 1

        avg_loss = total_loss / len(train_loader)
        avg_metrics = self._average_metrics(all_metrics) if all_metrics else {}

        return avg_loss, avg_metrics

    def _validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                if metric_fn:
                    metrics = metric_fn(outputs, targets)
                    all_metrics.append(metrics)

        avg_loss = total_loss / len(val_loader)
        avg_metrics = self._average_metrics(all_metrics) if all_metrics else {}

        return avg_loss, avg_metrics

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_kwargs = self.config.optimizer_kwargs or {}

        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        if not self.config.scheduler:
            return None

        scheduler_kwargs = self.config.scheduler_kwargs or {}

        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                **scheduler_kwargs
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_kwargs
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                **scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if not self.config.early_stopping_patience:
            return False

        if val_loss < self.best_val_metric:
            self.best_val_metric = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def _log_epoch_progress(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: float,
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log training progress for an epoch."""
        log_msg = f"Epoch {epoch + 1}/{self.config.epochs} - "
        log_msg += f"Train Loss: {train_loss:.4f}"

        for key, value in train_metrics.items():
            log_msg += f", Train {key}: {value:.4f}"

        if val_loss > 0:
            log_msg += f", Val Loss: {val_loss:.4f}"
            for key, value in val_metrics.items():
                log_msg += f", Val {key}: {value:.4f}"

        log_msg += f", Time: {epoch_time:.2f}s"
        log_msg += f", LR: {self.optimizer.param_groups[0]['lr']:.6f}"

        logger.info(log_msg)

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config,
            'best_val_metric': self.best_val_metric
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
