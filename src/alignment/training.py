"""
Training and evaluation utilities for neural networks.

This module provides utilities for training and evaluating neural networks,
including methods for measuring alignment during training.
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

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    dataset_config: Any,
    training_config: Any,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    extra_config: Optional[Any] = None,
    callbacks: Optional[List[Callable]] = None
) -> Dict[str, Any]:
    """
    Train a neural network model.
    
    Args:
        model: Model to train
        dataset_config: Configuration for the dataset
        training_config: Configuration for training (epochs, optimizer, etc.)
        device: Device to train on
        checkpoint_path: Path to save checkpoints
        extra_config: Additional configuration
        callbacks: List of callback functions to call during training
        
    Returns:
        Dictionary containing training metrics and results
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load dataset
    batch_size = dataset_config.batch_size if hasattr(dataset_config, 'batch_size') else 128
    dataset = load_dataset(dataset_config, batch_size=batch_size)
    train_loader = dataset.train_loader
    
    # Set up optimizer
    optimizer_name = training_config.optimizer
    optimizer_cls = getattr(optim, optimizer_name)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Set up learning rate scheduler if configured
    scheduler = None
    if hasattr(training_config, 'scheduler') and training_config.scheduler:
        scheduler_cls = getattr(optim.lr_scheduler, training_config.scheduler.name)
        scheduler = scheduler_cls(optimizer, **training_config.scheduler.params)
    
    # Initialize metric tracking
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'learning_rate': []
    }
    
    # Set up W&B if enabled
    wandb_enabled = False
    if extra_config and hasattr(extra_config, 'use_wandb') and extra_config.use_wandb:
        try:
            import wandb
            if not wandb.run:
                wandb.init(
                    project=getattr(extra_config, 'wandb_project', 'alignment'),
                    config={
                        'dataset': vars(dataset_config) if hasattr(dataset_config, 'to_dict') else vars(dataset_config),
                        'training': vars(training_config) if hasattr(training_config, 'to_dict') else vars(training_config),
                        'extra': vars(extra_config) if hasattr(extra_config, 'to_dict') else vars(extra_config)
                    }
                )
            wandb_enabled = True
        except ImportError:
            logger.warning("wandb not installed, but use_wandb=True. Disabling W&B logging.")
    
    # Training loop
    for epoch in range(training_config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.epochs}")
        
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / total
        epoch_accuracy = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        metrics['train_loss'].append(epoch_loss)
        metrics['train_accuracy'].append(epoch_accuracy)
        metrics['learning_rate'].append(current_lr)
        
        # Log to W&B if enabled
        if wandb_enabled:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_accuracy': epoch_accuracy,
                'learning_rate': current_lr
            })
        
        # Step scheduler if configured
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint if path is provided
        if checkpoint_path is not None and hasattr(training_config, 'checkpoint_frequency') and (epoch + 1) % training_config.checkpoint_frequency == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'metrics': metrics
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
        
        # Call callbacks if provided
        if callbacks is not None:
            for callback in callbacks:
                callback(epoch=epoch, model=model, metrics=metrics)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{training_config.epochs}: "
                   f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
                   f"LR: {current_lr:.6f}")
    
    # Evaluate on validation set if available
    if hasattr(dataset, 'val_loader') and dataset.val_loader is not None:
        val_metrics = evaluate_model(
            model,
            dataset_config,
            device=device,
            loader_name='val_loader',
            extra_config=extra_config
        )
        metrics.update({
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy']
        })
    
    return metrics


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