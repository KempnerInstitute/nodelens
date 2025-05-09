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
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from alignment.metrics import AlignmentMetric, get_metric
from alignment.datasets import load_dataset
from alignment.models import AlignmentNetwork
from alignment.datasets import DataSet
from alignment.utils.core import setup_logging, timed
from alignment.utils.evaluation import evaluate_networks, evaluate_on_loader
from alignment.utils.model_utils import _normalize_device, _ensure_model_on_device

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


def train_networks_fully_tensorized(
    networks: List[nn.Module],
    dataset,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    device=None,
    show_progress: bool = True,
    optimizer_class=torch.optim.Adam,
    weight_decay: float = 0.0,
    **optimizer_kwargs
) -> Dict:
    """
    Train multiple networks using a fully tensorized approach for maximum efficiency.
    
    This method trains all networks in a single forward/backward pass by using DP/DDP-like 
    data parallelism techniques. It's much faster than even the previous tensorized method.
    
    Args:
        networks: List of networks to train (must have identical architecture)
        dataset: Dataset object for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: Device to train on
        show_progress: Whether to show progress bars
        optimizer_class: The optimizer class to use.
        weight_decay: Weight decay for the optimizer.
        **optimizer_kwargs: Additional arguments for the optimizer.

    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _normalize_device(device)

    for net in networks:
        _ensure_model_on_device(net, device)
    
    # Track training history for plotting
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Number of networks
    num_networks = len(networks)
    
    # Create a replicated batch processor
    class NetworkEnsemble(nn.Module):
        def __init__(self, networks_list):
            super().__init__()
            self.networks = nn.ModuleList(networks_list)
            self.num_networks = len(networks_list)
            
        def forward(self, x):
            batch_size = x.size(0)
            outputs = []
            for network_item in self.networks:
                outputs.append(network_item(x))
            
            return torch.stack(outputs)
    
    ensemble = NetworkEnsemble(networks).to(device)
    
    # Create optimizer for the ensemble
    optimizer = optimizer_class(
        ensemble.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
        **optimizer_kwargs
    )
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs (Fully Tensorized)", position=0) if show_progress else range(num_epochs)
    
    for epoch in epoch_pbar:
        ensemble.train()
        
        train_loss_sum = 0.0
        train_correct_sum = torch.zeros(num_networks, device=device)
        train_total_samples = 0
        
        train_loader_iter = tqdm(dataset.train_loader, desc="Training batches", position=1, leave=False) if show_progress else dataset.train_loader
        
        for inputs, targets in train_loader_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            train_total_samples += batch_size
            
            optimizer.zero_grad()
            
            # outputs shape: [num_networks, batch_size, num_classes]
            outputs_ensemble = ensemble(inputs)
            
            current_batch_loss = 0
            for net_idx in range(num_networks):
                net_output = outputs_ensemble[net_idx]
                net_loss = F.cross_entropy(net_output, targets, reduction='mean')
                current_batch_loss += net_loss
                
                _, predicted = net_output.max(1)
                train_correct_sum[net_idx] += predicted.eq(targets).sum().item()
            
            avg_batch_loss = current_batch_loss / num_networks
            train_loss_sum += avg_batch_loss.item() * batch_size
            
            avg_batch_loss.backward()
            optimizer.step()
            
            if show_progress:
                current_avg_acc = train_correct_sum.sum().item() / (train_total_samples * num_networks) * 100.0
                train_loader_iter.set_postfix({'loss': f"{avg_batch_loss.item():.4f}", 'acc': f"{current_avg_acc:.2f}%"})
        
        avg_epoch_train_loss = train_loss_sum / train_total_samples
        avg_epoch_train_acc = 100.0 * train_correct_sum.sum().item() / (train_total_samples * num_networks)
        
        # Evaluation phase (evaluate individual networks in the ensemble)
        ensemble.eval()
        
        # We need to evaluate networks individually for accurate test metrics per network
        # then average them, as evaluate_networks expects a list of individual networks.
        current_epoch_test_losses = []
        current_epoch_test_accs = []
        with torch.no_grad():
            for i in range(num_networks):
                individual_network = ensemble.networks[i]
                individual_network.eval()
                test_metrics_single_net = evaluate_on_loader(individual_network, dataset.test_loader, device, show_progress=False)
                current_epoch_test_losses.append(test_metrics_single_net['loss'])
                current_epoch_test_accs.append(test_metrics_single_net['accuracy'])

        avg_epoch_test_loss = np.mean(current_epoch_test_losses)
        avg_epoch_test_acc = np.mean(current_epoch_test_accs)
        
        training_history['train_loss'].append(avg_epoch_train_loss)
        training_history['train_acc'].append(avg_epoch_train_acc)
        training_history['test_loss'].append(avg_epoch_test_loss)
        training_history['test_acc'].append(avg_epoch_test_acc)
        
        if show_progress:
            epoch_pbar.set_postfix({
                'train_loss': f"{avg_epoch_train_loss:.4f}",
                'train_acc': f"{avg_epoch_train_acc:.2f}%",
                'test_loss': f"{avg_epoch_test_loss:.4f}",
                'test_acc': f"{avg_epoch_test_acc:.2f}%"
            })
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} (Fully Tensorized): "
                   f"Train Loss={avg_epoch_train_loss:.4f}, Train Acc={avg_epoch_train_acc:.2f}%, "
                   f"Test Loss={avg_epoch_test_loss:.4f}, Test Acc={avg_epoch_test_acc:.2f}%")
    
    return training_history


def train_networks_sequential(
    networks: List[nn.Module],
    dataset,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    device=None,
    show_progress: bool = True,
    optimizer_class=torch.optim.Adam,
    weight_decay: float = 0.0,
    **optimizer_kwargs
) -> Dict[str, List[float]]:
    """
    Train multiple networks on the given dataset sequentially.
    This is also used for training a single network.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _normalize_device(device)

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    # Accumulators for averaging metrics over all networks at each epoch
    epoch_avg_train_losses = []
    epoch_avg_train_accs = []
    epoch_avg_test_losses = []
    epoch_avg_test_accs = []

    for epoch in range(num_epochs):
        current_epoch_train_losses = []
        current_epoch_train_accs = []
        current_epoch_test_losses = []
        current_epoch_test_accs = []

        net_iter_desc = f"Epoch {epoch+1}/{num_epochs} (Sequential)"
        net_iter = tqdm(networks, desc=net_iter_desc, leave=False) if show_progress and len(networks) > 1 else networks

        for net in net_iter:
            _ensure_model_on_device(net, device)
            optimizer = optimizer_class(
                net.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_kwargs
            )
            net.train()
            epoch_train_loss_single_net = 0.0
            epoch_train_correct_single_net = 0
            epoch_train_total_single_net = 0

            batch_iter_desc = "Training Batches"
            if len(networks) == 1 and show_progress:
                 batch_iter_desc = f"Epoch {epoch+1}/{num_epochs} Training"
            
            train_loader_iter = tqdm(dataset.train_loader, desc=batch_iter_desc, leave=False) if show_progress else dataset.train_loader

            for inputs, targets in train_loader_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss_single_net += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                epoch_train_correct_single_net += predicted.eq(targets).sum().item()
                epoch_train_total_single_net += targets.size(0)

            # Avg loss/acc for this network for this epoch
            avg_loss_this_net_epoch = epoch_train_loss_single_net / epoch_train_total_single_net
            avg_acc_this_net_epoch = 100.0 * epoch_train_correct_single_net / epoch_train_total_single_net
            current_epoch_train_losses.append(avg_loss_this_net_epoch)
            current_epoch_train_accs.append(avg_acc_this_net_epoch)

            # Evaluate this network on test set
            test_metrics_single_net = evaluate_on_loader(net, dataset.test_loader, device, show_progress=False)
            current_epoch_test_losses.append(test_metrics_single_net['loss'])
            current_epoch_test_accs.append(test_metrics_single_net['accuracy'])

        # Average metrics over all networks for the current epoch
        history['train_loss'].append(np.mean(current_epoch_train_losses))
        history['train_acc'].append(np.mean(current_epoch_train_accs))
        history['test_loss'].append(np.mean(current_epoch_test_losses))
        history['test_acc'].append(np.mean(current_epoch_test_accs))

        if show_progress:
            print(f"Epoch {epoch+1}/{num_epochs} (Sequential Avg) - "
                  f"train_loss: {history['train_loss'][-1]:.4f}, train_acc: {history['train_acc'][-1]:.2f}%, "
                  f"test_loss: {history['test_loss'][-1]:.4f}, test_acc: {history['test_acc'][-1]:.2f}%")
            
    return history


def train_networks(
    networks: List[nn.Module],
    dataset,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    device=None,
    show_progress: bool = True,
    optimizer_class=torch.optim.Adam,
    weight_decay: float = 0.0,
    training_method: str = "auto",
    **optimizer_kwargs
) -> Dict[str, List[float]]:
    """
    Train multiple networks on the given dataset using the specified or auto-selected method.
    
    Args:
        networks: List of networks to train
        dataset: Dataset object with train_loader and test_loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: Device to train on
        show_progress: Whether to show progress bar
        optimizer_class: Optimizer class to use (e.g., torch.optim.Adam)
        weight_decay: Weight decay for optimizer
        training_method: Method for training ('auto', 'sequential', 'fully_tensorized')
        **optimizer_kwargs: Additional arguments to pass to optimizer
        
    Returns:
        Dictionary with training history
    """
    if not networks:
        logger.warning("No networks provided to train_networks")
        return {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _normalize_device(device)
    
    # Check if networks have the same architecture for optimized training
    # (simple check based on class type, might need more robust for complex cases)
    same_architecture = True
    if len(networks) > 1:
        base_net_type = type(networks[0].base_model if hasattr(networks[0], 'base_model') else networks[0])
        for net in networks[1:]:
            current_net_type = type(net.base_model if hasattr(net, 'base_model') else net)
            if current_net_type != base_net_type:
                same_architecture = False
                break
    
    selected_method = training_method
    if training_method == "auto":
        if len(networks) == 1:
            selected_method = "sequential"
        elif same_architecture:
            selected_method = "fully_tensorized"
        else:
            selected_method = "sequential" # Fallback for different architectures
        logger.info(f"Auto-selected training method: {selected_method}")

    common_args = {
        "networks": networks,
        "dataset": dataset,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "device": device,
        "show_progress": show_progress,
        "optimizer_class": optimizer_class,
        "weight_decay": weight_decay,
        **optimizer_kwargs
    }

    if selected_method == "fully_tensorized":
        if not same_architecture and len(networks) > 1:
            logger.warning("Fully_tensorized training requires all networks to have the same architecture. "
                           "Falling back to sequential training.")
            return train_networks_sequential(**common_args)
        try:
            logger.info(f"Using fully_tensorized training for {len(networks)} networks.")
            return train_networks_fully_tensorized(**common_args)
        except Exception as e:
            logger.error(f"Fully_tensorized training failed: {str(e)}. Falling back to sequential.", exc_info=True)
            return train_networks_sequential(**common_args)
    elif selected_method == "sequential":
        logger.info(f"Using sequential training for {len(networks)} networks.")
        return train_networks_sequential(**common_args)
    else:
        logger.warning(f"Unknown training_method: {selected_method}. Defaulting to sequential training.")
        return train_networks_sequential(**common_args)

# Ensure the older single-network train_network and test_network functions are distinct
# if they are still used elsewhere. For now, the dispatcher `train_networks` is the main entry point
# for AlignmentExperiment.

# The original `train_network` (singular) can remain if it serves a purpose for single network training
# outside the multi-network replicate scenario handled by `AlignmentExperiment`.
# Same for `test_network` and `train_and_test`.
# `train_model` and `evaluate_model` also seem like general utility functions.
# `evaluate` is a more complex evaluation function, also seems distinct. 