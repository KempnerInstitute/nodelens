"""
Training functions for neural network models.

This module provides functions for training and testing neural networks,
with standardized interfaces and reporting.
"""

import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from alignment.metrics import AlignmentMetric, get_metric
from alignment.datasets import load_dataset
from alignment.models import AlignmentNetwork
from alignment.datasets import DataSet
from alignment.utils.core import setup_logging, timed

# Setup module logger
logger = logging.getLogger(__name__)


def train_network(
    network: AlignmentNetwork,
    dataset: DataSet,
    epochs: int = 10,
    learning_rate: float = 0.001,
    optimizer_name: str = "Adam",
    batch_size: Optional[int] = None,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Dict[str, List[float]]:
    """
    Train a neural network.
    
    Args:
        network: The neural network to train
        dataset: Dataset to train on
        epochs: Number of epochs to train
        learning_rate: Learning rate
        optimizer_name: Name of optimizer ('Adam', 'SGD', etc.)
        batch_size: Batch size (uses dataset default if None)
        weight_decay: Weight decay for regularization
        device: Device to train on
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Dictionary with training metrics (loss, accuracy)
    """
    if device is None:
        device = next(network.parameters()).device
    
    # Set network to training mode
    network.train()
    
    # Get data loader (use dataset's batch size if not specified)
    if batch_size is None:
        batch_size = dataset.batch_size
    
    train_loader = DataLoader(
        dataset.train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(
        network.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        # Iterate over batches
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time
        
        # Save history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Log progress
        logger.info(f'Epoch {epoch+1}/{epochs} - '
                   f'Loss: {epoch_loss:.4f}, '
                   f'Accuracy: {epoch_acc:.2f}%, '
                   f'Time: {epoch_time:.2f}s')
        
        # Progress callback
        if progress_callback:
            progress_callback(epoch, epoch_loss, epoch_acc)
    
    return history


def test_network(
    network: AlignmentNetwork,
    dataset: DataSet,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Test a neural network.
    
    Args:
        network: The neural network to test
        dataset: Dataset to test on
        batch_size: Batch size (uses dataset default if None)
        device: Device to test on
    
    Returns:
        Dictionary with test metrics (loss, accuracy)
    """
    if device is None:
        device = next(network.parameters()).device
    
    # Set network to evaluation mode
    network.eval()
    
    # Get data loader (use dataset's batch size if not specified)
    if batch_size is None:
        batch_size = dataset.batch_size
    
    test_loader = DataLoader(
        dataset.test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # No gradient computation during testing
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    # Log results
    logger.info(f'Test - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return {
        'test_loss': avg_loss,
        'test_acc': accuracy
    }


@timed
def train_and_test(
    network: AlignmentNetwork,
    dataset: DataSet,
    epochs: int = 10,
    learning_rate: float = 0.001,
    optimizer_name: str = "Adam",
    batch_size: Optional[int] = None,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Dict[str, Dict[str, Union[List[float], float]]]:
    """
    Train and test a neural network.
    
    Args:
        network: The neural network to train
        dataset: Dataset to train on
        epochs: Number of epochs to train
        learning_rate: Learning rate
        optimizer_name: Name of optimizer ('Adam', 'SGD', etc.)
        batch_size: Batch size (uses dataset default if None)
        weight_decay: Weight decay for regularization
        device: Device to train on
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Dictionary with training and test metrics
    """
    # Train network
    train_history = train_network(
        network=network,
        dataset=dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        batch_size=batch_size,
        weight_decay=weight_decay,
        device=device,
        progress_callback=progress_callback
    )
    
    # Test network
    test_metrics = test_network(
        network=network,
        dataset=dataset,
        batch_size=batch_size,
        device=device
    )
    
    # Combine results
    return {
        'train': train_history,
        'test': test_metrics
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    optimizer: Optional[optim.Optimizer] = None,
    num_epochs: int = 10,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_freq: int = 1,
    return_history: bool = False
) -> Dict[str, Any]:
    """
    Train a neural network model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        optimizer: Optimizer to use for training (if None, creates Adam optimizer)
        num_epochs: Number of epochs to train
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Frequency to save checkpoints (in epochs)
        return_history: Whether to return training history
        
    Returns:
        Dictionary containing training metrics and results
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize metric tracking
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # If model returns (outputs, hidden), take outputs
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar - simplified
            pbar.set_postfix({
                'loss': f"{total_loss / total:.4f}",
                'acc': f"{100. * correct / total:.1f}%"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / total
        epoch_accuracy = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['learning_rate'].append(current_lr)
        
        # Evaluate on validation set if provided
        if val_loader is not None:
            val_results = evaluate_model(
                model=model,
                data_loader=val_loader,
                device=device
            )
            
            history['val_loss'].append(val_results['loss'])
            history['val_accuracy'].append(val_results['accuracy'])
            
            # Simplified logging - only log final epoch or every 5 epochs
            if epoch == num_epochs - 1 or (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.2f}%, "
                          f"Val Loss={val_results['loss']:.4f}, Val Acc={val_results['accuracy']:.2f}%")
        else:
            # Simplified logging - only log final epoch or every 5 epochs
            if epoch == num_epochs - 1 or (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.2f}%")
        
        # Save checkpoint if directory is provided
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_freq == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_accuracy': epoch_accuracy,
                'val_loss': history['val_loss'][-1] if val_loader is not None else None,
                'val_accuracy': history['val_accuracy'][-1] if val_loader is not None else None
            }, checkpoint_file)
    
    # Return training history if requested
    if return_history:
        return history
    
    # Return final model and optimizer for compatibility
    return {
        'model': model,
        'optimizer': optimizer,
        'final_loss': history['train_loss'][-1],
        'final_accuracy': history['train_accuracy'][-1],
        'val_loss': history['val_loss'][-1] if val_loader is not None else None,
        'val_accuracy': history['val_accuracy'][-1] if val_loader is not None else None
    }


def evaluate_model(
    model: nn.Module,
    dataset_config: Any,
    device: Optional[torch.device] = None,
    loader_name: str = 'test_loader',
    extra_config: Optional[Any] = None,
    with_alignment: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a neural network model.
    
    Args:
        model: Model to evaluate
        dataset_config: Configuration for the dataset
        device: Device to evaluate on
        loader_name: Name of the loader to use ('test_loader', 'val_loader', etc.)
        extra_config: Additional configuration
        with_alignment: Whether to measure alignment metrics
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load dataset
    batch_size = dataset_config.batch_size if hasattr(dataset_config, 'batch_size') else 128
    dataset = load_dataset(dataset_config, batch_size=batch_size)
    
    # Get the specified data loader
    loader = getattr(dataset, loader_name, None)
    if loader is None:
        raise ValueError(f"Data loader '{loader_name}' not found in dataset")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Measure alignment if requested
    alignment_metrics = {}
    metric = None
    
    if with_alignment or (extra_config and hasattr(extra_config, 'measure_alignment')):
        metric_name = 'RQ'  # Default
        if extra_config and hasattr(extra_config, 'alignment') and hasattr(extra_config.alignment, 'metric'):
            metric_name = extra_config.alignment.metric
        
        metric = get_metric(metric_name)
    
    # Collect alignment values if measuring
    alignment_values = []
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with store_hidden if measuring alignment
            store_hidden = metric is not None
            if store_hidden:
                model.forward(inputs, store_hidden=True)
                outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            # Handle tuple output (predictions, hidden_activations)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Measure alignment if requested
            if store_hidden:
                # For AlignmentNetwork models
                if hasattr(model, 'measure_alignment'):
                    batch_alignment = model.measure_alignment(inputs, precomputed=True, method=metric_name)
                    alignment_values.append(batch_alignment)
    
    # Calculate final metrics
    metrics = {
        'loss': total_loss / total,
        'accuracy': 100. * correct / total
    }
    
    # Process alignment metrics if measured
    if alignment_values:
        # Average alignment values across batches for each layer
        avg_alignment = []
        for layer_idx in range(len(alignment_values[0])):
            layer_values = [batch[layer_idx] for batch in alignment_values]
            avg_alignment.append(sum(layer_values) / len(layer_values))
        
        metrics['alignment'] = avg_alignment
    
    # Log results
    logger.info(f"Evaluation: Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%")
    if 'alignment' in metrics:
        alignment_str = ', '.join([f"{val:.4f}" for val in metrics['alignment']])
        logger.info(f"Alignment: [{alignment_str}]")
    
    return metrics


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to the checkpoint file
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load the checkpoint to
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint to CPU first to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device if specified
    if device is not None:
        model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return checkpoint metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metrics': checkpoint.get('metrics', {})
    }
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {metadata['epoch']})")
    
    return metadata


# Continue with the existing train and evaluate functions for backward compatibility
@torch.no_grad()
def evaluate(nets, dataset, **parameters):
    """
    Evaluate networks on a dataset.
    
    Args:
        nets: Neural network(s) to evaluate
        dataset: Dataset to evaluate on
        **parameters: Additional parameters
            - train_set: Whether to use training set (default: False)
            - alignment: Whether to measure alignment (default: False)
            - methods: List of alignment methods to measure (default: ["RQ"])
            - measure_expected: Whether to compute expected distributions (default: True)
            - bins: Number of bins for histograms (default: 50)
            
    Returns:
        Dictionary of evaluation results
    """
    nets = nets if isinstance(nets, list) else [nets]
    num_nets = len(nets)
    device = dataset.device
    
    # Get parameters with defaults
    train_set = parameters.get("train_set", False)
    do_alignment = parameters.get("alignment", False)
    alignment_methods = parameters.get("methods", ["RQ"])
    measure_expected = parameters.get("measure_expected", True)
    bins = parameters.get("bins", 50)
    
    # Get dataloader
    dataloader = dataset.train_loader if train_set else dataset.test_loader
    
    # Initialize results
    results = {}
    loss_vec = torch.zeros(num_nets)
    acc_vec = torch.zeros(num_nets)
    
    # Track totals for average calculation
    total_correct = torch.zeros(num_nets, device=device)
    total_loss = torch.zeros(num_nets, device=device)
    total_samples = 0
    
    # Evaluate on dataloader
    for batch in dataloader:
        images, labels = dataset.unwrap_batch(batch, device=device)
        
        # Forward pass and calculate metrics for each network
        for idx, net in enumerate(nets):
            out = net(images)
            
            # Loss (sum for now, will divide by total_samples later)
            loss_val = dataset.measure_loss(out, labels, reduction="sum")
            total_loss[idx] += loss_val.detach()
            
            # Accuracy
            pred = out.argmax(dim=1)
            total_correct[idx] += (pred == labels).sum()
            
        total_samples += labels.size(0)
        
    # Calculate average loss and accuracy
    if total_samples > 0:
        for idx in range(num_nets):
            loss_vec[idx] = total_loss[idx].cpu().item() / float(total_samples)
            acc_vec[idx] = (total_correct[idx].cpu().item() / total_samples) * 100.0
            
    # Store basic metrics
    results["loss"] = loss_vec
    results["accuracy"] = acc_vec
    
    # Measure alignment if requested
    if do_alignment:
        # Get a batch of data for alignment measurement
        images, labels = next(iter(dataloader))
        images, labels = dataset.unwrap_batch((images, labels), device=device)
        
        # Measure alignment metrics for each network
        align_data = []
        dist_data = []
        exp_data = []
        
        for net in nets:
            # Forward pass to store activations
            net.forward(images, store_hidden=True)
            
            # Measure alignment metrics
            metrics = AlignmentMetrics.measure_methods(net, images, methods=alignment_methods, precomputed=False)
            align_data.append(metrics)
            
            # Compute distributions of alignment values
            layer_dists = []
            for layer_dict in metrics:
                m_d = {}
                for m, val_tensor in layer_dict.items():
                    val_cpu = val_tensor.detach().cpu()
                    c, e = torch.histogram(val_cpu, bins=bins, density=True)
                    m_d[m] = (c, e)
                layer_dists.append(m_d)
            dist_data.append(layer_dists)
            
            # Compute expected distributions if requested
            if measure_expected:
                net_inps = net.get_layer_inputs(images, precomputed=False)
                layer_exp_list = []
                for inp in net_inps:
                    if inp.ndim == 4:
                        inp = inp.flatten(start_dim=1)
                    wvals, _ = AlignmentMetrics.compute_eigenvalues(inp)
                    method_exp = {}
                    for m in alignment_methods:
                        ccounts, cedges = AlignmentMetrics.measure_expected_distribution(m, wvals, bins=bins)
                        method_exp[m] = (ccounts, cedges)
                    layer_exp_list.append(method_exp)
                exp_data.append(layer_exp_list)
                
        # Store alignment results
        results["alignment"] = [{"epoch": "test", "batch": "all", "data": align_data}]
        results["alignment_distribution"] = [{"epoch": "test", "batch": "all", "data": dist_data}]
        
        if measure_expected:
            results["expected_distribution"] = [{"epoch": "test", "batch": "all", "data": exp_data}]
    
    return results 