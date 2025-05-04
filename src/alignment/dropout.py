"""
Dropout implementation and analysis utilities.

This module provides functionality for various types of dropout in neural networks,
including progressive dropout and eigenvector-based dropout, which are used
to analyze alignment properties of networks.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alignment.metrics import AlignmentMetric

logger = logging.getLogger(__name__)


def progressive_dropout(
    model: nn.Module,
    dataset_config: Any,
    dropout_fraction: float = 0.1,
    metric: Optional[AlignmentMetric] = None,
    batch_size: int = 128,
    num_batches: int = 10,
    device: Optional[torch.device] = None,
    dropout_mode: str = "scaled"
) -> Tuple[float, List[float]]:
    """
    Apply progressive dropout to a network and measure accuracy and alignment.
    
    This function performs dropout on each layer of the network in sequence,
    starting with random node dropout and measuring the impact on accuracy.
    
    Args:
        model: Neural network to analyze
        dataset_config: Configuration for the dataset
        dropout_fraction: Fraction of nodes to drop out
        metric: Alignment metric to use for measuring alignment
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate
        device: Device to run the computation on
        dropout_mode: Mode for dropout application ('scaled' or 'unscaled')
        
    Returns:
        Tuple of (accuracy, list of alignment values for each layer)
    """
    if not hasattr(model, "alignment_layers"):
        raise ValueError("Model must have alignment_layers attribute for progressive dropout")
    
    if device is None:
        device = next(model.parameters()).device
        
    # Import dataset loader here to avoid circular imports
    from alignment.datasets import load_dataset
    
    # Prepare dataset
    dataset = load_dataset(dataset_config, batch_size=batch_size)
    test_loader = dataset.test_loader
    
    # Get number of classes from dataset
    num_classes = dataset.num_classes
    
    # Set model to evaluation mode
    model.eval()
    
    # Calculate number of nodes to dropout per layer
    n_layers = len(model.alignment_layers)
    dropout_indices = []
    
    for layer_idx, layer in enumerate(model.alignment_layers):
        # Get output dimension of the layer
        output_dim = layer.weight.size(0) if hasattr(layer, "weight") else 0
        
        if output_dim == 0:
            logger.warning(f"Layer {layer_idx} has no weight attribute, skipping")
            dropout_indices.append(torch.tensor([], device=device, dtype=torch.long))
            continue
            
        # Calculate number of nodes to dropout
        n_dropout = int(output_dim * dropout_fraction)
        
        # Generate random indices to dropout
        indices = torch.randperm(output_dim, device=device)[:n_dropout]
        dropout_indices.append(indices)
        
    # Validation loop
    correct = 0
    total = 0
    alignment_values = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply dropout to all layers and get activations
        with torch.no_grad():
            outputs, hidden = model.forward_targeted_dropout(
                inputs, dropout_indices, list(range(n_layers)), dropout_mode=dropout_mode
            )
            
            # Calculate alignment if metric is provided
            if metric is not None:
                batch_alignment = metric.measure(hidden, targets, num_classes)
                if len(alignment_values) == 0:
                    alignment_values = batch_alignment
                else:
                    # Average with previous batches
                    for i, val in enumerate(batch_alignment):
                        alignment_values[i] = (alignment_values[i] * batch_idx + val) / (batch_idx + 1)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"Progressive dropout with fraction {dropout_fraction:.4f}: Accuracy = {accuracy:.4f}")
    
    return accuracy, alignment_values


def eigenvector_dropout(
    model: nn.Module,
    dataset_config: Any,
    dropout_fraction: float = 0.1,
    metric: Optional[AlignmentMetric] = None,
    batch_size: int = 128,
    num_batches: int = 10,
    device: Optional[torch.device] = None,
    dropout_mode: str = "scaled"
) -> Tuple[float, List[float]]:
    """
    Apply eigenvector-based dropout to a network and measure accuracy and alignment.
    
    This function performs dropout based on the principal eigenvectors of each layer,
    targeting the nodes with the highest activation along the principal component.
    
    Args:
        model: Neural network to analyze
        dataset_config: Configuration for the dataset
        dropout_fraction: Fraction of nodes to drop out
        metric: Alignment metric to use for measuring alignment
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate
        device: Device to run the computation on
        dropout_mode: Mode for dropout application ('scaled' or 'unscaled')
        
    Returns:
        Tuple of (accuracy, list of alignment values for each layer)
    """
    if not hasattr(model, "alignment_layers"):
        raise ValueError("Model must have alignment_layers attribute for eigenvector dropout")
    
    if device is None:
        device = next(model.parameters()).device
        
    # Import dataset loader here to avoid circular imports
    from alignment.datasets import load_dataset
    
    # Prepare dataset
    dataset = load_dataset(dataset_config, batch_size=batch_size)
    test_loader = dataset.test_loader
    
    # Get number of classes from dataset
    num_classes = dataset.num_classes
    
    # Set model to evaluation mode
    model.eval()
    
    # Get activations for computing eigenvectors
    activation_stats = _collect_activations(model, test_loader, num_batches, device)
    
    # Calculate eigenvectors for each layer
    n_layers = len(model.alignment_layers)
    dropout_indices = []
    
    for layer_idx in range(n_layers):
        activations = activation_stats[layer_idx]["activations"]
        
        if activations.size(1) == 0:
            logger.warning(f"Layer {layer_idx} has no activations, skipping")
            dropout_indices.append(torch.tensor([], device=device, dtype=torch.long))
            continue
        
        # Calculate covariance matrix
        activations = activations - activations.mean(0, keepdim=True)
        cov = torch.mm(activations.T, activations) / activations.size(0)
        
        # Get principal eigenvector
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        principal_eigenvector = eigenvectors[:, -1]  # Last column is the principal eigenvector
        
        # Calculate importance scores based on eigenvector components
        importance = torch.abs(principal_eigenvector)
        
        # Sort by importance and select top nodes
        output_dim = importance.size(0)
        n_dropout = int(output_dim * dropout_fraction)
        
        # Get indices of highest importance nodes
        _, indices = torch.topk(importance, n_dropout)
        dropout_indices.append(indices)
    
    # Validation loop with eigenvector dropout
    correct = 0
    total = 0
    alignment_values = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply dropout to all layers and get activations
        with torch.no_grad():
            outputs, hidden = model.forward_targeted_dropout(
                inputs, dropout_indices, list(range(n_layers)), dropout_mode=dropout_mode
            )
            
            # Calculate alignment if metric is provided
            if metric is not None:
                batch_alignment = metric.measure(hidden, targets, num_classes)
                if len(alignment_values) == 0:
                    alignment_values = batch_alignment
                else:
                    # Average with previous batches
                    for i, val in enumerate(batch_alignment):
                        alignment_values[i] = (alignment_values[i] * batch_idx + val) / (batch_idx + 1)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"Eigenvector dropout with fraction {dropout_fraction:.4f}: Accuracy = {accuracy:.4f}")
    
    return accuracy, alignment_values


def _collect_activations(
    model: nn.Module, 
    data_loader: DataLoader, 
    num_batches: int,
    device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    """
    Collect activations from all layers of the model.
    
    Args:
        model: Neural network
        data_loader: DataLoader for input data
        num_batches: Number of batches to process
        device: Device to run computation on
        
    Returns:
        List of dictionaries with activations for each layer
    """
    n_layers = len(model.alignment_layers)
    activation_stats = [{"activations": None} for _ in range(n_layers)]
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            # Forward pass to get activations
            _, hidden = model(inputs)
            
            # Store activations
            for layer_idx, layer_output in enumerate(hidden):
                # Reshape to 2D: (batch_size, -1)
                flat_output = layer_output.view(layer_output.size(0), -1)
                
                if activation_stats[layer_idx]["activations"] is None:
                    activation_stats[layer_idx]["activations"] = flat_output.cpu()
                else:
                    activation_stats[layer_idx]["activations"] = torch.cat(
                        [activation_stats[layer_idx]["activations"], flat_output.cpu()], dim=0
                    )
    
    # Move activations back to device for computation
    for layer_idx in range(n_layers):
        if activation_stats[layer_idx]["activations"] is not None:
            activation_stats[layer_idx]["activations"] = activation_stats[layer_idx]["activations"].to(device)
    
    return activation_stats 