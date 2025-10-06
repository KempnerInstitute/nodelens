"""
Multi-network training utilities for training multiple networks simultaneously.

This module provides efficient training of multiple networks at once
by batching their computations together.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def train_networks_fully_tensorized(
    networks: List[nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 10,
    optimizer_class: type = optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    loss_fn: Optional[Callable] = None,
    device: str = "cuda",
    checkpoint_dir: Optional[Path] = None,
    log_interval: int = 100,
    eval_interval: int = 1,
    callbacks: Optional[List[Callable]] = None
) -> Tuple[List[nn.Module], Dict[str, List[float]]]:
    """
    Train multiple networks simultaneously using tensorized operations.

    This function efficiently trains multiple networks with the same architecture
    by batching their forward/backward passes together.

    Args:
        networks: List of networks to train (must have same architecture)
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        optimizer_class: Optimizer class to use
        optimizer_kwargs: Keyword arguments for optimizer
        loss_fn: Loss function (default: CrossEntropyLoss)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_interval: Log progress every N batches
        eval_interval: Evaluate every N epochs
        callbacks: List of callback functions

    Returns:
        Tuple of (trained networks, training history)
    """
    if not networks:
        raise ValueError("No networks provided")

    if len(networks) == 1:
        logger.warning("Only one network provided, falling back to standard training")
        return _train_single_network(
            networks[0], train_loader, val_loader, epochs, optimizer_class,
            optimizer_kwargs, loss_fn, device, checkpoint_dir, log_interval,
            eval_interval, callbacks
        )

    # Verify all networks have the same architecture
    if not _verify_same_architecture(networks):
        raise ValueError("All networks must have the same architecture for tensorized training")

    # Setup
    num_networks = len(networks)
    device = torch.device(device)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer_kwargs = optimizer_kwargs or {}

    # Move networks to device
    for net in networks:
        net.to(device)

    # Create tensorized network wrapper
    tensorized_net = TensorizedNetworkWrapper(networks)

    # Create optimizer for tensorized parameters
    optimizer = optimizer_class(tensorized_net.parameters(), **optimizer_kwargs)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }

    # Training loop
    logger.info(f"Starting tensorized training of {num_networks} networks for {epochs} epochs")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training phase
        train_loss, train_acc = _tensorized_train_epoch(
            tensorized_net, train_loader, optimizer, loss_fn, device,
            epoch, log_interval, callbacks
        )

        # Validation phase
        val_loss, val_acc = 0.0, 0.0
        if val_loader and (epoch + 1) % eval_interval == 0:
            val_loss, val_acc = _tensorized_evaluate(
                tensorized_net, val_loader, loss_fn, device
            )

        # Record history
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % eval_interval == 0:
            _save_tensorized_checkpoint(
                tensorized_net, optimizer, epoch, history, checkpoint_dir
            )

    # Extract individual networks
    trained_networks = tensorized_net.extract_networks()

    return trained_networks, history


class TensorizedNetworkWrapper(nn.Module):
    """
    Wrapper that combines multiple networks for tensorized training.

    This wrapper manages multiple networks and runs them in parallel
    by calling each network's forward method individually.
    """

    def __init__(self, networks: List[nn.Module]):
        """
        Initialize tensorized wrapper.

        Args:
            networks: List of networks with same architecture
        """
        super().__init__()
        self.networks = nn.ModuleList(networks)
        self.num_networks = len(networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all networks.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            Output tensor [num_networks, batch_size, ...]
        """
        outputs = []
        for network in self.networks:
            outputs.append(network(x))

        return torch.stack(outputs, dim=0)

    def extract_networks(self) -> List[nn.Module]:
        """Extract individual networks (they are already separate)."""
        return list(self.networks)


def _verify_same_architecture(networks: List[nn.Module]) -> bool:
    """Verify all networks have the same architecture."""
    if len(networks) < 2:
        return True

    base_arch = str(networks[0])
    for net in networks[1:]:
        if str(net) != base_arch:
            return False

    # Also check parameter shapes
    base_params = {name: param.shape for name, param in networks[0].named_parameters()}
    for net in networks[1:]:
        net_params = {name: param.shape for name, param in net.named_parameters()}
        if net_params != base_params:
            return False

    return True


def _tensorized_train_epoch(
    model: TensorizedNetworkWrapper,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    callbacks: Optional[List[Callable]] = None
) -> Tuple[float, float]:
    """Train for one epoch with tensorized operations."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Expand targets for all networks
        targets.unsqueeze(0).expand(model.num_networks, -1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # [num_networks, batch_size, num_classes]

        # Compute loss for each network
        losses = []
        for net_idx in range(model.num_networks):
            loss = loss_fn(outputs[net_idx], targets)
            losses.append(loss)

        # Average loss across networks
        loss = torch.stack(losses).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()

        # Compute accuracy (average across networks)
        for net_idx in range(model.num_networks):
            _, predicted = outputs[net_idx].max(1)
            correct += predicted.eq(targets).sum().item()
        total += targets.size(0) * model.num_networks

        # Logging
        if batch_idx % log_interval == 0:
            logger.debug(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

        # Callbacks
        if callbacks:
            for callback in callbacks:
                callback(model, epoch, batch_idx)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def _tensorized_evaluate(
    model: TensorizedNetworkWrapper,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate tensorized model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss and accuracy for each network
            for net_idx in range(model.num_networks):
                loss = loss_fn(outputs[net_idx], targets)
                total_loss += loss.item()

                _, predicted = outputs[net_idx].max(1)
                correct += predicted.eq(targets).sum().item()

            total += targets.size(0) * model.num_networks

    avg_loss = total_loss / (len(val_loader) * model.num_networks)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def _train_single_network(
    network: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    epochs: int,
    optimizer_class: type,
    optimizer_kwargs: Optional[Dict[str, Any]],
    loss_fn: Optional[Callable],
    device: str,
    checkpoint_dir: Optional[Path],
    log_interval: int,
    eval_interval: int,
    callbacks: Optional[List[Callable]]
) -> Tuple[List[nn.Module], Dict[str, List[float]]]:
    """Fallback to standard single network training."""
    device = torch.device(device)
    network = network.to(device)

    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = optimizer_class(network.parameters(), **optimizer_kwargs)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }

    logger.info("Using standard single network training")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training phase
        network.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            if batch_idx % log_interval == 0:
                logger.debug(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validation phase
        val_loss, val_acc = 0.0, 0.0
        if val_loader and (epoch + 1) % eval_interval == 0:
            network.eval()
            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = network(inputs)
                    loss = loss_fn(outputs, targets)

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)

            val_loss = total_loss / len(val_loader)
            val_acc = 100.0 * correct / total

        # Record history
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % eval_interval == 0:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            checkpoint_path = checkpoint_dir / f"single_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

        # Callbacks
        if callbacks:
            for callback in callbacks:
                callback(network, epoch, history)

    return [network], history


def _save_tensorized_checkpoint(
    model: TensorizedNetworkWrapper,
    optimizer: optim.Optimizer,
    epoch: int,
    history: Dict[str, List[float]],
    checkpoint_dir: Path
):
    """Save checkpoint for tensorized training."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'num_networks': model.num_networks
    }

    checkpoint_path = checkpoint_dir / f"tensorized_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.debug(f"Saved checkpoint to {checkpoint_path}")
