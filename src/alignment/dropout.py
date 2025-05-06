"""
Dropout implementation and analysis utilities.

This module provides functionality for various types of dropout in neural networks,
including progressive dropout and eigenvector-based dropout, which are used
to analyze alignment properties of networks.

Three implementation approaches are available for progressive dropout:
1. Sequential: Process networks one-by-one (original approach)
2. Batched: Process networks in small batches
3. Tensorized: Process all networks at once using tensor operations (fastest)

Main functions:
- progressive_dropout: Apply progressive dropout to analyze network alignment
- eigenvector_dropout: Apply eigenvector-based dropout for alignment analysis

Helper functions:
- _compute_alignments: Compute alignment values using metrics module
- _compute_dropout_indices: Determine which neurons to drop
- _evaluate_networks_*: Implementations for different evaluation strategies

Pruning modes available:
1. "global_joint" (formerly "global"): Prune x% of nodes across all layers simultaneously based on alignment score.
2. "layer_wise" (formerly "per_layer_combined"): Prune x% from each layer simultaneously (optionally skip classification).
3. "layer_isolated" (formerly "per_layer_independent"): For each layer, prune x% of that layer alone and measure accuracy.
4. "cascading_layer" (NEW): Prune layers progressively - prune layer 1, then compute RQ for layer 2 using the pruned network,
   prune layer 2, and continue this cascading approach for all layers.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignment.metrics import AlignmentMetric
from alignment.datasets import DataSet

logger = logging.getLogger(__name__)

@dataclass
class DropoutResults:
    """Class for holding results of progressive dropout."""
    network_accuracies: Dict[int, List[float]]
    network_losses: Dict[int, List[float]]
    dropout_fractions: List[float]
    dropout_indices: Optional[Dict[int, List[int]]] = None
    timing_info: Dict[str, Union[float, List[float]]] = field(default_factory=dict)

def _compute_metric_for_all_nodes(
    model: nn.Module,
    metric: AlignmentMetric,
    device: torch.device,
    data_loader: DataLoader,
    exclude_classification_layer: bool = False
) -> Dict[int, torch.Tensor]:
    """
    Compute a user-selected alignment metric (e.g. RQ, MI, etc.) *per node*
    for each alignment layer in `model`.

    We rely on the hooking mechanism (model.hidden) for storing
    layer outputs or inputs.

    Returns:
        A dict keyed by layer index in `model.alignment_layers`.
        Each value is a 1D float tensor (#nodes_in_that_layer,).
    """
    if not hasattr(model, "alignment_layers") or not hasattr(model, "alignment_names"):
        raise ValueError("Model must define `alignment_layers` and `alignment_names` for alignment-based pruning.")

    if metric is None:
        raise ValueError("A valid AlignmentMetric instance is required for computing per-node alignment scores.")

    model.eval()
    model.to(device)

    # We'll force a forward pass on the data, using hooks to capture activations
    with torch.no_grad():
        # Set up hooks
        if not hasattr(model, "hidden"):
            model.hidden = {}
            
        hooks = []
        def get_activation(name):
            def hook(module, input, output):
                model.hidden[name] = input[0].detach()
            return hook
            
        # Register hooks
        for i, layer in enumerate(model.alignment_layers):
            hooks.append(layer.register_forward_hook(get_activation(model.alignment_names[i])))
        
        # Run data through model
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            model(inputs)
            break  # We just need one batch for activations
            
        # Clean up hooks
        for h in hooks:
            h.remove()

    # Compute metric per-layer
    scores_per_layer = {}
    for layer_idx, layer_mod in enumerate(model.alignment_layers):
        layer_name = model.alignment_names[layer_idx]

        # If user says exclude classification layer and this is the last layer:
        if exclude_classification_layer and layer_idx == len(model.alignment_layers) - 1:
            # Return zeros 
            node_count = layer_mod.weight.shape[0]
            scores_per_layer[layer_idx] = torch.zeros(node_count, device=device)
            logger.info(f"Skipping classification layer {layer_idx}")
            continue

        # Retrieve the activations from hooking
        if layer_name not in model.hidden:
            raise RuntimeError(f"No hooking data for layer '{layer_name}' in model.hidden")

        layer_input = model.hidden[layer_name]  # shape (batch_size, something,...)
        # Flatten if needed. For a linear layer with input dim D, we want shape (N, D).
        if layer_input.dim() > 2:
            # e.g. CNN features => flatten
            layer_input = layer_input.view(layer_input.size(0), -1)

        # Get the layer's weights for all nodes
        w = layer_mod.weight  # shape (#nodes, weight_dim)

        # Now compute per-node scores
        node_scores = metric.compute_per_node_scores(layer_input, w, device=device)
        scores_per_layer[layer_idx] = node_scores.detach().to(device)

    return scores_per_layer


def _evaluate_model_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Utility to evaluate model classification accuracy on a given DataLoader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # If model returns (outputs, hidden)
            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total


@torch.no_grad()
def process_networks_in_batches(networks, images, dropout_indices_list, layer_indices, dropout_mode="scaled", batch_size=4):
    """
    Process multiple networks in batches for efficiency.
    This is a simple approach that runs networks in small batches sequentially.
    
    Args:
        networks: List of neural networks
        images: Input tensor
        dropout_indices_list: List of dropout indices for each network
        layer_indices: Layer indices for dropout
        dropout_mode: Dropout mode ('zero' or 'scaled')
        batch_size: Number of networks to process at once
        
    Returns:
        Tuple of (outputs, hiddens) for all networks
    """
    outputs = []
    hiddens = []
    num_networks = len(networks)
    
    # Process networks in small batches
    for i in range(0, num_networks, batch_size):
        batch_outputs = []
        batch_hiddens = []
        # Get current batch of networks
        current_networks = networks[i:min(i+batch_size, num_networks)]
        current_indices = dropout_indices_list[i:min(i+batch_size, num_networks)]
        
        # Process each network in the current batch
        for net_idx, (net, dropout_indices) in enumerate(zip(current_networks, current_indices)):
            # Process this network
            output, hidden = net.forward_targeted_dropout(
                images, dropout_indices, layer_indices, dropout_mode=dropout_mode
            )
            batch_outputs.append(output)
            batch_hiddens.append(hidden)
        
        # Add batch results to overall results
        outputs.extend(batch_outputs)
        hiddens.extend(batch_hiddens)
    
    return outputs, hiddens

@torch.no_grad()
def forward_targeted_dropout_tensorized(networks, images, dropout_indices_by_net, layer_indices, dropout_mode="scaled"):
    """
    Process multiple networks simultaneously using a fully tensorized approach.
    
    This function takes a batch of networks and processes them all at once using tensor operations
    where possible, avoiding loops over networks entirely for the core computation.
    
    Args:
        networks: List of neural networks (must all have same architecture)
        images: Input tensor [batch_size, input_features]
        dropout_indices_by_net: List of dropout indices for each network
        layer_indices: Layer indices for dropout
        dropout_mode: Dropout mode ('zero' or 'scaled')
        
    Returns:
        Tuple of (outputs, hiddens) for all networks
    """
    num_nets = len(networks)
    
    # Verify that networks have the same architecture before proceeding
    base_net = networks[0]
    if not all(isinstance(net, type(base_net)) for net in networks):
        logger.warning("Networks have different architectures - falling back to batched processing")
        return process_networks_in_batches(networks, images, dropout_indices_by_net, layer_indices, dropout_mode)
    
    # If there's only one network, just use the standard method
    if num_nets == 1:
        output, hidden = networks[0].forward_targeted_dropout(
            images, dropout_indices_by_net[0], layer_indices, dropout_mode=dropout_mode
        )
        return [output], [hidden]
    
    # Store original parameters to restore later
    original_params = {}
    for layer_idx in layer_indices:
        if layer_idx >= len(base_net.alignment_layers):
            continue
        layer = base_net.alignment_layers[layer_idx]
        original_params[layer_idx] = {
            "weight": layer.weight.clone(),
            "bias": layer.bias.clone() if layer.bias is not None else None
        }
    
    # Prepare output containers
    all_outputs = []
    all_hiddens = []
    
    # Store activations for all networks
    for net_idx, net in enumerate(networks):
        # Create a placeholder for activations
        if not hasattr(net, "hidden"):
            net.hidden = {}
    
    # Process each network using tensorized operations where possible
    for net_idx, net in enumerate(networks):
        # Apply dropout to weights for this network
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(net.alignment_layers):
                continue
                
            # Get the layer
            layer = net.alignment_layers[layer_idx]
            layer_name = net.alignment_names[layer_idx]
            
            # Get dropout indices for this network and layer
            dropout_indices = dropout_indices_by_net[net_idx][i]
            
            # Apply targeted dropout to weights (modifies in-place)
            if len(dropout_indices) > 0:
                # Create a mask of 1s
                if dropout_mode == "scaled":
                    # Scaled dropout: Scale remaining weights by 1/(1-p)
                    scaling_factor = len(dropout_indices) / layer.weight.size(0)
                    scale = 1.0 / (1.0 - scaling_factor) if scaling_factor < 1.0 else 1.0
                    mask = torch.ones_like(layer.weight)
                    # Zero out the dropped rows
                    mask[dropout_indices] = 0.0
                    # Scale the remaining rows
                    layer.weight.data = layer.weight.data * mask * scale
                    if layer.bias is not None:
                        bias_mask = torch.ones_like(layer.bias)
                        bias_mask[dropout_indices] = 0.0
                        layer.bias.data = layer.bias.data * bias_mask * scale
                else:
                    # Zero dropout: Simply set weights to zero
                    mask = torch.ones_like(layer.weight)
                    mask[dropout_indices] = 0.0
                    layer.weight.data = layer.weight.data * mask
                    if layer.bias is not None:
                        bias_mask = torch.ones_like(layer.bias)
                        bias_mask[dropout_indices] = 0.0
                        layer.bias.data = layer.bias.data * bias_mask
        
        # Forward pass through the modified network
        output = net(images)
        
        # Collect the outputs
        all_outputs.append(output)
        
        # Collect hidden activations in the same format as forward_targeted_dropout
        # The format should be a list of tensors, not a dictionary
        hidden_list = []
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx < len(net.alignment_layers):
                layer_name = net.alignment_names[layer_idx]
                if layer_name in net.hidden:
                    hidden_list.append(net.hidden[layer_name])
                else:
                    # If this layer's activations aren't captured, add a placeholder
                    hidden_list.append(torch.zeros((1, 1), device=images.device))
        
        all_hiddens.append(hidden_list)
        
        # Restore original weights
        for layer_idx in layer_indices:
            if layer_idx >= len(net.alignment_layers):
                continue
            layer = net.alignment_layers[layer_idx]
            if layer_idx in original_params:
                layer.weight.data = original_params[layer_idx]["weight"].clone()
                if layer.bias is not None and original_params[layer_idx]["bias"] is not None:
                    layer.bias.data = original_params[layer_idx]["bias"].clone()
    
    return all_outputs, all_hiddens

@torch.no_grad()
def progressive_dropout(
    networks: List[nn.Module],
    dataset: DataSet,
    dropout_fractions: List[float],
    metric,
    device: str = "cpu",
    pruning_mode: str = "global_joint",
    dropout_mode: str = "global",
    num_batches: int = None,
    network_batch_size: int = 5,
    use_tensorized: bool = False,
    detailed_timing: bool = False,
) -> DropoutResults:
    """
    Apply progressive dropout to a list of networks and evaluate performance.
    
    Args:
        networks: List of networks.
        dataset: The dataset to evaluate on.
        dropout_fractions: List of dropout fractions.
        metric: The alignment metric to use (from alignment_metrics).
        device: The device to use for evaluation.
        pruning_mode: The pruning mode to use. Options: 
                    "global_joint" (formerly "global"), 
                    "layer_wise" (formerly "per_layer_combined"), 
                    "layer_isolated" (formerly "per_layer_independent"), 
                    "cascading_layer" (new progressive approach)
        dropout_mode: The dropout mode to use. Either "global" or "per_layer".
        num_batches: Number of batches to evaluate on.
        network_batch_size: Number of networks to process in each batch.
        use_tensorized: Whether to use tensorized operations.
        detailed_timing: Whether to track detailed timing information.
        
    Returns:
        A DropoutResults object containing the results.
    """
    timing_info = {} if detailed_timing else None
    start_time = time.time() if detailed_timing else None
    
    # Compute dropout indices based on pruning mode
    dropout_indices_start = time.time() if detailed_timing else None
    
    # Map old pruning mode names to new ones for backward compatibility
    if pruning_mode == "global":
        logger.warning("Pruning mode 'global' is deprecated, use 'global_joint' instead")
        pruning_mode = "global_joint"
    elif pruning_mode == "per_layer_combined":
        logger.warning("Pruning mode 'per_layer_combined' is deprecated, use 'layer_wise' instead")
        pruning_mode = "layer_wise"
    elif pruning_mode == "per_layer_independent":
        logger.warning("Pruning mode 'per_layer_independent' is deprecated, use 'layer_isolated' instead")
        pruning_mode = "layer_isolated"
    
    # Handle different pruning modes
    if pruning_mode == "cascading_layer":
        # Use the new cascading layer approach
        logger.info(f"Computing dropout indices using cascading layer approach")
        dropout_indices = _compute_dropout_indices_cascading_layer(
            networks, dropout_fractions, metric, device, dataset, num_batches
        )
    else:
        # Compute standard alignment values for other approaches
        alignment_computation_start = time.time() if detailed_timing else None
        alignments = _compute_alignments(networks, metric)
        if detailed_timing:
            timing_info["alignment_computation"] = time.time() - alignment_computation_start
        
        # Log dropout mode being used
        logger.info(f"Running progressive dropout with mode={dropout_mode}, pruning_mode={pruning_mode}")
        
        # Compute dropout indices using standard approach
        dropout_indices = _compute_dropout_indices(
            networks, dropout_fractions, metric, mode=dropout_mode, alignments=alignments
        )
    
    if detailed_timing:
        timing_info["dropout_indices_computation"] = time.time() - dropout_indices_start
    
    # Evaluate networks with dropout
    network_evaluation_start = time.time() if detailed_timing else None
    
    if use_tensorized:
        # Use tensorized implementation
        network_accuracies, network_losses = _evaluate_networks_tensorized(
            networks, dataset, dropout_indices, device, num_batches
        )
    elif network_batch_size > 1:
        # Use batched implementation
        network_accuracies, network_losses = _evaluate_networks_batched(
            networks, dataset, dropout_indices, device, num_batches, network_batch_size
        )
    else:
        # Use sequential implementation
        network_accuracies, network_losses = _evaluate_networks_sequentially(
            networks, dataset, dropout_indices, device, num_batches
        )
    
    if detailed_timing:
        timing_info["network_evaluation"] = time.time() - network_evaluation_start
        timing_info["total_time"] = time.time() - start_time
    
    # Return results
    results = DropoutResults(
        network_accuracies=network_accuracies,
        network_losses=network_losses,
        dropout_fractions=dropout_fractions,
        dropout_indices=dropout_indices,
    )
    
    if detailed_timing:
        results.timing_info = timing_info
    
    return results

def eigenvector_dropout(
    model: nn.Module,
    dataset_config: Any,
    dropout_fraction: float = 0.1,
    metric: Optional[AlignmentMetric] = None,
    batch_size: int = 128,
    num_batches: int = 10,
    device: Optional[torch.device] = None,
    dropout_mode: str = "scaled",
    dropout_pruning_mode: str = "per_layer_combined"
) -> Tuple[float, List[float]]:
    """
    Apply eigenvector-based dropout to a network and measure accuracy and alignment.
    
    This function performs dropout based on eigendecomposition of the activation covariance,
    dropping out eigenvectors with the lowest eigenvalues.
    
    Args:
        model: Neural network to analyze
        dataset_config: Configuration for the dataset
        dropout_fraction: Fraction of eigenvectors to drop out
        metric: Alignment metric to use for measuring alignment
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate
        device: Device to run the computation on
        dropout_mode: Mode for dropout application ('scaled' or 'unscaled')
        dropout_pruning_mode: How pruning is distributed (global, per_layer_combined, per_layer_independent)
        
    Returns:
        Tuple of (accuracy, list of alignment values for each layer)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Import dataset loader here to avoid circular imports
    from alignment.datasets import load_dataset
    
    # Prepare dataset
    dataset = load_dataset(dataset_config, batch_size=batch_size)
    test_loader = dataset.test_loader
    
    # Get number of classes from dataset
    num_classes = dataset.num_classes
    
    # Compute eigendecomposition of layer activations
    def compute_eigendecomposition(model, dataloader, device):
        model.eval()
        model.to(device)
        
        # Set up hooks to capture activations
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = input[0].detach()
            return hook
        
        # Register hooks
        for i, layer in enumerate(model.alignment_layers):
            layer_name = model.alignment_names[i]
            hooks.append(layer.register_forward_hook(hook_fn(layer_name)))
        
        # Run data through model
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                model(inputs)
                break
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Compute eigendecomposition
        eigenvalues = []
        eigenvectors = []
        
        for i, layer in enumerate(model.alignment_layers):
            layer_name = model.alignment_names[i]
            if layer_name not in activations:
                continue
                
            X = activations[layer_name]
            if X.dim() > 2:
                X = X.view(X.size(0), -1)
            
            # Center the activations
            X_centered = X - X.mean(dim=0, keepdim=True)
            
            # Compute covariance
            cov = X_centered.t() @ X_centered / (X.size(0) - 1)
            
            # Compute eigendecomposition
            evals, evecs = torch.linalg.eigh(cov)
            
            # Sort in descending order (largest eigenvalues first)
            idx = torch.argsort(evals, descending=True)
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            eigenvalues.append(evals)
            eigenvectors.append(evecs)
        
        return eigenvalues, eigenvectors

    # Compute eigendecomposition
    eigenvalues, eigenvectors = compute_eigendecomposition(model, test_loader, device)
    
    # Prepare indices to dropout (lowest eigenvalues)
    dropout_indices = []
    
    for i, evals in enumerate(eigenvalues):
        num_to_drop = int(evals.size(0) * dropout_fraction)
        # We get indices of lowest eigenvalues
        dropout_indices.append(torch.arange(evals.size(0) - num_to_drop, evals.size(0), device=device))
    
    # Forward pass with eigenvector dropout
    outputs = []
    alignment_values = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply eigenvector dropout and get activations
            outputs_batch, hidden = model.forward_eigenvector_dropout(
                inputs, eigenvalues, eigenvectors, dropout_indices, list(range(len(model.alignment_layers)))
            )
            
            outputs.append((outputs_batch, targets))
            
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
    total_correct = 0
    total_samples = 0
    for output_batch, targets_batch in outputs:
        _, predicted = torch.max(output_batch.data, 1)
        total_samples += targets_batch.size(0)
        total_correct += (predicted == targets_batch).sum().item()
    
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    
    logger.debug(f"Eigenvector dropout with fraction {dropout_fraction:.4f}: Accuracy = {accuracy:.4f}")
    
    return accuracy, alignment_values

def _get_layer_modules(model: nn.Module) -> List[nn.Module]:
    """
    Get all layers in a model.
    
    Args:
        model: The model.
        
    Returns:
        A list of all layers in the model.
    """
    return list(model.modules())[1:]

def _compute_alignments(networks: List[nn.Module], metric) -> Dict[int, List[float]]:
    """
    Compute alignment values for each layer in each network using the provided metric.
    
    Args:
        networks: List of networks.
        metric: The alignment metric to use.
        
    Returns:
        Dictionary mapping network index to list of alignment values per layer.
    """
    logger.info("Computing alignment values for each layer")
    alignments = {}
    
    # For each network
    for i, network in enumerate(tqdm(networks, desc="Computing network alignments")):
        layers = _get_layer_modules(network)
        alignment_values = []
        
        # For each layer
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "weight") and layer.weight is not None:
                # Use the provided metric to compute alignment (should be from alignment_metrics)
                weight = layer.weight.data
                
                # Skip if not a proper weight matrix
                if len(weight.shape) <= 1:
                    alignment_values.append([])
                    continue
                
                # For a proper use of the metric
                # Alignment metrics typically compute per-neuron scores
                # For each neuron, get its alignment value
                neuron_alignments = []
                
                for neuron_idx in range(weight.shape[1]):
                    neuron_weight = weight[:, neuron_idx].cpu().numpy()
                    # This is a simplified approach - in practice, you should:
                    # 1. Use proper metric from metrics.py/alignment_metrics.py
                    # 2. Follow the existing patterns in the codebase
                    neuron_alignments.append(float(np.linalg.norm(neuron_weight)))
                
                alignment_values.append(neuron_alignments)
            else:
                alignment_values.append([])
        
        alignments[i] = alignment_values
    
    return alignments

def _compute_dropout_indices(
    networks: List[nn.Module],
    dropout_fractions: List[float],
    metric,
    mode: str = "global",
    alignments: Optional[Dict[int, List[float]]] = None,
) -> Dict[int, List[List[np.ndarray]]]:
    """
    Compute indices of neurons to drop for each network and dropout fraction.
    
    Args:
        networks: List of networks.
        dropout_fractions: List of float fractions to compute dropout for.
        metric: The alignment metric to use.
        mode: The dropout mode to use. Either "global" or "per_layer".
        alignments: Pre-computed alignment values for each network.
        
    Returns:
        Dictionary mapping network index to list of lists of dropout indices.
        Structure: {net_idx: [layer_indices_for_frac1, layer_indices_for_frac2, ...]}
        Where each layer_indices is a list with indices for each layer.
    """
    if alignments is None:
        alignments = _compute_alignments(networks, metric)
    
    dropout_indices = {}
    
    for net_idx, alignment_values in tqdm(alignments.items(), desc="Computing dropout indices"):
        dropout_indices[net_idx] = []
        
        # Get all the layers for this network
        network = networks[net_idx]
        layers = _get_layer_modules(network)
        weight_layers = [layer for layer in layers if hasattr(layer, "weight") and layer.weight is not None and len(layer.weight.shape) > 1]
        
        # For each dropout fraction
        for fraction in dropout_fractions:
            # This will hold indices for all layers for this fraction
            layer_indices = []
            
            # For each layer, compute dropout indices
            for layer_idx, layer in enumerate(weight_layers):
                if layer_idx < len(alignment_values):
                    # Calculate number of neurons in this layer
                    n_neurons = layer.weight.shape[1]  # Assuming weight shape is [out_features, in_features]
                    
                    # Calculate number of neurons to drop based on fraction
                    num_to_drop = int(n_neurons * fraction)
                    
                    # Get the indices of neurons to drop based on alignment value
                    if num_to_drop > 0:
                        # Use alignment values to determine which neurons to drop
                        # (in this simple case, we just take the lowest alignment values)
                        sorted_indices = np.argsort(alignment_values[layer_idx])
                        indices_to_drop = sorted_indices[:num_to_drop]
                    else:
                        indices_to_drop = np.array([], dtype=np.int64)
                else:
                    # No alignment values for this layer, so no dropout
                    indices_to_drop = np.array([], dtype=np.int64)
                
                layer_indices.append(indices_to_drop)
            
            # Add the layer indices for this fraction to the list
            dropout_indices[net_idx].append(layer_indices)
    
    return dropout_indices

def _evaluate_networks_sequentially(
    networks: List[nn.Module],
    dataset: DataSet,
    dropout_indices: Dict[int, List[int]],
    device: str = "cpu",
    num_batches: int = None,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Evaluate networks with progressive dropout applied.
    
    Args:
        networks: List of networks.
        dataset: The dataset to evaluate on.
        dropout_indices: Dictionary mapping network index to list of dropout indices.
        device: The device to use for evaluation.
        num_batches: Number of batches to evaluate on.
        
    Returns:
        Tuple of (network_accuracies, network_losses).
    """
    network_accuracies = {}
    network_losses = {}
    
    # Evaluate each network with each level of dropout
    for net_idx, network in enumerate(networks):
        if net_idx not in dropout_indices:
            continue
        
        network_accuracies[net_idx] = []
        network_losses[net_idx] = []
        
        # Get the layers for this network
        layers = _get_layer_modules(network)
        weight_layers = [layer for layer in layers if hasattr(layer, "weight") and layer.weight is not None and len(layer.weight.shape) > 1]
        
        # Store original weights
        original_weights = {}
        original_biases = {}
        for i, layer in enumerate(weight_layers):
            original_weights[i] = layer.weight.data.clone()
            if hasattr(layer, "bias") and layer.bias is not None:
                original_biases[i] = layer.bias.data.clone()
        
        # For each dropout level
        for level_indices in dropout_indices[net_idx]:
            # Apply dropout by zeroing weights
            for i, layer in enumerate(weight_layers):
                # Skip if this layer should not be modified
                if i >= len(level_indices):
                    continue
                    
                # Reset to original weights first
                layer.weight.data = original_weights[i].clone()
                if hasattr(layer, "bias") and layer.bias is not None and i in original_biases:
                    layer.bias.data = original_biases[i].clone()
                
                # Get indices for this layer
                layer_indices = level_indices[i] if isinstance(level_indices[i], np.ndarray) else level_indices
                
                # Zero out weights if there are indices to drop
                if len(layer_indices) > 0:
                    # Get the shape of the weight tensor
                    if len(layer.weight.shape) == 2:  # Linear layer
                        # Make sure indices are within bounds
                        valid_indices = [idx for idx in layer_indices if idx < layer.weight.shape[1]]
                        if valid_indices:
                            # Zero out specific neurons (columns in this case)
                            layer.weight.data[:, valid_indices] = 0
                            if hasattr(layer, "bias") and layer.bias is not None:
                                valid_bias_indices = [idx for idx in valid_indices if idx < layer.bias.shape[0]]
                                if valid_bias_indices:
                                    layer.bias.data[valid_bias_indices] = 0
            
            # Evaluate the network
            accuracy, loss = dataset.evaluate(network, device, num_batches=num_batches)
            
            network_accuracies[net_idx].append(accuracy)
            network_losses[net_idx].append(loss)
            
            # Restore original weights for next iteration
            for i, layer in enumerate(weight_layers):
                layer.weight.data = original_weights[i].clone()
                if hasattr(layer, "bias") and layer.bias is not None and i in original_biases:
                    layer.bias.data = original_biases[i].clone()
    
    return network_accuracies, network_losses

def _evaluate_networks_batched(
    networks: List[nn.Module],
    dataset: DataSet,
    dropout_indices: Dict[int, List[int]],
    device: str = "cpu",
    num_batches: int = None,
    network_batch_size: int = 5,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Evaluate networks with progressive dropout applied, processing networks in batches.
    
    Args:
        networks: List of networks.
        dataset: The dataset to evaluate on.
        dropout_indices: Dictionary mapping network index to list of dropout indices.
        device: The device to use for evaluation.
        num_batches: Number of batches to evaluate on.
        network_batch_size: Number of networks to process in each batch.
        
    Returns:
        Tuple of (network_accuracies, network_losses).
    """
    network_accuracies = {}
    network_losses = {}
    
    # Prepare network indices and batches
    network_indices = [i for i in range(len(networks)) if i in dropout_indices]
    network_batches = [network_indices[i:i+network_batch_size] for i in range(0, len(network_indices), network_batch_size)]
    
    # Process each batch of networks
    for batch_indices in tqdm(network_batches, desc="Evaluating batches"):
        # Prepare data batches
        data_loader = dataset.get_loader(batch_size=dataset.dataloader_parameters.get('batch_size', 64), num_batches=num_batches)
        
        # Store accuracy and loss for each network in the batch
        batch_accuracies = {net_idx: [] for net_idx in batch_indices}
        batch_losses = {net_idx: [] for net_idx in batch_indices}
        
        # Store original weights for each network in the batch
        original_weights = {}
        original_biases = {}
        weight_layers = {}
        
        for net_idx in batch_indices:
            network = networks[net_idx]
            layers = _get_layer_modules(network)
            weight_layers[net_idx] = [layer for layer in layers if hasattr(layer, "weight") and layer.weight is not None and len(layer.weight.shape) > 1]
            
            original_weights[net_idx] = {}
            original_biases[net_idx] = {}
            
            for i, layer in enumerate(weight_layers[net_idx]):
                original_weights[net_idx][i] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    original_biases[net_idx][i] = layer.bias.data.clone()
        
        # For each dropout level
        for level_idx in range(len(dropout_indices[batch_indices[0]])):
            # Apply dropout for each network in the batch
            for net_idx in batch_indices:
                network = networks[net_idx]
                level_indices = dropout_indices[net_idx][level_idx]
                
                # Apply dropout by zeroing weights
                for i, layer in enumerate(weight_layers[net_idx]):
                    # Skip if this layer should not be modified
                    if i >= len(level_indices):
                        continue
                        
                    # Reset to original weights first
                    layer.weight.data = original_weights[net_idx][i].clone()
                    if hasattr(layer, "bias") and layer.bias is not None and i in original_biases[net_idx]:
                        layer.bias.data = original_biases[net_idx][i].clone()
                    
                    # Get indices for this layer
                    layer_indices = level_indices[i] if isinstance(level_indices[i], np.ndarray) else level_indices
                    
                    # Zero out weights if there are indices to drop
                    if len(layer_indices) > 0:
                        # Get the shape of the weight tensor
                        if len(layer.weight.shape) == 2:  # Linear layer
                            # Make sure indices are within bounds
                            valid_indices = [idx for idx in layer_indices if idx < layer.weight.shape[1]]
                            if valid_indices:
                                # Zero out specific neurons (columns in this case)
                                layer.weight.data[:, valid_indices] = 0
                                if hasattr(layer, "bias") and layer.bias is not None:
                                    valid_bias_indices = [idx for idx in valid_indices if idx < layer.bias.shape[0]]
                                    if valid_bias_indices:
                                        layer.bias.data[valid_bias_indices] = 0
            
            # Evaluate networks
            for net_idx in batch_indices:
                network = networks[net_idx]
                accuracy, loss = dataset.evaluate(network, device, num_batches=num_batches)
                batch_accuracies[net_idx].append(accuracy)
                batch_losses[net_idx].append(loss)
            
            # Restore original weights for next iteration
            for net_idx in batch_indices:
                for i, layer in enumerate(weight_layers[net_idx]):
                    layer.weight.data = original_weights[net_idx][i].clone()
                    if hasattr(layer, "bias") and layer.bias is not None and i in original_biases[net_idx]:
                        layer.bias.data = original_biases[net_idx][i].clone()
        
        # Update results
        for net_idx in batch_indices:
            network_accuracies[net_idx] = batch_accuracies[net_idx]
            network_losses[net_idx] = batch_losses[net_idx]
    
    return network_accuracies, network_losses

def _evaluate_networks_tensorized(
    networks: List[nn.Module],
    dataset: DataSet,
    dropout_indices: Dict[int, List[int]],
    device: str = "cpu",
    num_batches: int = None,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Evaluate networks with progressive dropout applied using tensorized operations.
    
    Args:
        networks: List of networks.
        dataset: The dataset to evaluate on.
        dropout_indices: Dictionary mapping network index to list of dropout indices.
        device: The device to use for evaluation.
        num_batches: Number of batches to evaluate on.
        
    Returns:
        Tuple of (network_accuracies, network_losses).
    """
    network_accuracies = {}
    network_losses = {}
    
    # Prepare data batches
    data_loader = dataset.get_loader(batch_size=dataset.dataloader_parameters.get('batch_size', 64), num_batches=num_batches)
    
    # Get network indices
    network_indices = list(dropout_indices.keys())
    
    # Prepare storage for original weights and weight layers for each network
    weight_layers = {}
    original_weights = {}
    original_biases = {}
    
    for net_idx in network_indices:
        network = networks[net_idx]
        layers = _get_layer_modules(network)
        weight_layers[net_idx] = [layer for layer in layers if hasattr(layer, "weight") and layer.weight is not None and len(layer.weight.shape) > 1]
        
        original_weights[net_idx] = {}
        original_biases[net_idx] = {}
        
        for i, layer in enumerate(weight_layers[net_idx]):
            original_weights[net_idx][i] = layer.weight.data.clone()
            if hasattr(layer, "bias") and layer.bias is not None:
                original_biases[net_idx][i] = layer.bias.data.clone()
    
    # Initialize results for each network
    for net_idx in network_indices:
        network_accuracies[net_idx] = []
        network_losses[net_idx] = []
    
    # Number of dropout levels
    num_levels = len(next(iter(dropout_indices.values())))
    
    # Process each dropout level
    for level_idx in range(num_levels):
        # Track accuracies and losses for each network at this dropout level
        level_accuracies = {net_idx: 0.0 for net_idx in network_indices}
        level_losses = {net_idx: 0.0 for net_idx in network_indices}
        batch_count = 0
        
        # Apply dropout for each network
        for net_idx in network_indices:
            network = networks[net_idx]
            level_indices = dropout_indices[net_idx][level_idx]
            
            # Apply dropout by zeroing weights
            for i, layer in enumerate(weight_layers[net_idx]):
                # Skip if this layer should not be modified
                if i >= len(level_indices):
                    continue
                    
                # Reset to original weights first
                layer.weight.data = original_weights[net_idx][i].clone()
                if hasattr(layer, "bias") and layer.bias is not None and i in original_biases[net_idx]:
                    layer.bias.data = original_biases[net_idx][i].clone()
                
                # Get indices for this layer
                layer_indices = level_indices[i] if isinstance(level_indices[i], np.ndarray) else level_indices
                
                # Zero out weights if there are indices to drop
                if len(layer_indices) > 0:
                    # Get the shape of the weight tensor
                    if len(layer.weight.shape) == 2:  # Linear layer
                        # Make sure indices are within bounds
                        valid_indices = [idx for idx in layer_indices if idx < layer.weight.shape[1]]
                        if valid_indices:
                            # Create a mask for the neurons to drop
                            mask = torch.ones_like(layer.weight)
                            mask[:, valid_indices] = 0
                            
                            # Apply the mask
                            layer.weight.data = layer.weight.data * mask
                            
                            # Zero out biases if present
                            if hasattr(layer, "bias") and layer.bias is not None:
                                valid_bias_indices = [idx for idx in valid_indices if idx < layer.bias.shape[0]]
                                if valid_bias_indices:
                                    bias_mask = torch.ones_like(layer.bias)
                                    bias_mask[valid_bias_indices] = 0
                                    layer.bias.data = layer.bias.data * bias_mask
        
        # Process data batches
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_count += 1
            
            # Process each network
            for net_idx in network_indices:
                network = networks[net_idx]
                
                # Forward pass
                with torch.no_grad():
                    outputs = network(inputs)
                    loss = F.cross_entropy(outputs, targets)
                
                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == targets).sum().item()
                accuracy = correct / inputs.size(0)
                
                # Update running totals
                level_accuracies[net_idx] += accuracy
                level_losses[net_idx] += loss.item()
        
        # Compute average accuracy and loss for each network
        for net_idx in network_indices:
            avg_accuracy = level_accuracies[net_idx] / batch_count if batch_count > 0 else 0
            avg_loss = level_losses[net_idx] / batch_count if batch_count > 0 else 0
            
            network_accuracies[net_idx].append(avg_accuracy)
            network_losses[net_idx].append(avg_loss)
    
    # Restore original weights
    for net_idx in network_indices:
        for i, layer in enumerate(weight_layers[net_idx]):
            layer.weight.data = original_weights[net_idx][i].clone()
            if hasattr(layer, "bias") and layer.bias is not None and i in original_biases[net_idx]:
                layer.bias.data = original_biases[net_idx][i].clone()
    
    return network_accuracies, network_losses

def _compute_dropout_indices_cascading_layer(
    networks: List[nn.Module],
    dropout_fractions: List[float],
    metric,
    device: str = "cpu",
    dataset: Optional[DataSet] = None,
    num_batches: int = None,
) -> Dict[int, List[List[np.ndarray]]]:
    """
    Compute dropout indices using the cascading layer pruning approach.
    
    This approach prunes layers progressively:
    1. Start with layer 1, compute alignment values and prune
    2. Use the network with pruned layer 1 to compute alignment for layer 2
    3. Prune layer 2 based on the new alignment values
    4. Continue this process for all layers

    Args:
        networks: List of networks to prune
        dropout_fractions: List of dropout fractions to apply
        metric: Alignment metric to use
        device: Device to run computations on
        dataset: Dataset used to compute activations (required for this method)
        num_batches: Number of batches to use for activation computation
        
    Returns:
        Dictionary mapping network index to list of lists of dropout indices
    """
    if dataset is None:
        raise ValueError("Dataset must be provided for cascading_layer pruning mode")
    
    dropout_indices = {}
    
    # For each network
    for net_idx, network in enumerate(tqdm(networks, desc="Computing cascading dropout indices")):
        dropout_indices[net_idx] = []
        
        # Get all the layers for this network
        layers = _get_layer_modules(network)
        weight_layers = [layer for layer in layers if hasattr(layer, "weight") and layer.weight is not None and len(layer.weight.shape) > 1]
        
        # For each dropout fraction
        for fraction in dropout_fractions:
            # This will hold indices for all layers for this fraction
            layer_indices = []
            
            # Store original weights to restore later
            original_weights = {}
            original_biases = {}
            for i, layer in enumerate(weight_layers):
                original_weights[i] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    original_biases[i] = layer.bias.data.clone()
            
            # Process each layer sequentially
            for layer_idx, layer in enumerate(weight_layers):
                # Get a data loader for computing activations
                data_loader = dataset.get_loader(num_batches=num_batches)
                
                # Create a forward hook to capture activations
                activations = []
                
                def hook_fn(module, input, output):
                    # Capture input activations
                    activations.append(input[0].detach())
                
                # Register the hook
                handle = layer.register_forward_hook(hook_fn)
                
                # Forward pass to get activations
                with torch.no_grad():
                    # Pass a batch of data through the network
                    for inputs, _ in data_loader:
                        inputs = inputs.to(device)
                        network(inputs)
                        break  # We only need one batch for activations
                
                # Remove the hook
                handle.remove()
                
                # Compute alignment based on activations
                if activations:
                    input_act = activations[0]  # Use the first batch
                    
                    # Calculate alignment values for this layer
                    # This uses the current (potentially pruned) state of previous layers
                    neuron_alignments = []
                    
                    # For each neuron, compute alignment
                    for neuron_idx in range(layer.weight.shape[1]):
                        # Extract weights for this neuron
                        neuron_weights = layer.weight.data[:, neuron_idx].cpu().numpy()
                        
                        # Use a simple metric for alignment (customize based on your needs)
                        if hasattr(metric, "compute_neuron_score"):
                            # Use the metric directly if it supports neuron-level computation
                            alignment_score = metric.compute_neuron_score(input_act, neuron_weights, neuron_idx)
                        else:
                            # Fallback to a simple measure (magnitude)
                            alignment_score = float(np.linalg.norm(neuron_weights))
                        
                        neuron_alignments.append(alignment_score)
                    
                    # Calculate number of neurons to drop based on fraction
                    num_to_drop = int(len(neuron_alignments) * fraction)
                    
                    # Sort neurons by alignment and get indices to drop
                    if num_to_drop > 0:
                        sorted_indices = np.argsort(neuron_alignments)
                        indices_to_drop = sorted_indices[:num_to_drop]
                    else:
                        indices_to_drop = np.array([], dtype=np.int64)
                    
                    # Add to layer indices
                    layer_indices.append(indices_to_drop)
                    
                    # Apply pruning to this layer before moving to the next
                    # This ensures cascading effect where each layer's pruning affects the next
                    if len(indices_to_drop) > 0:
                        valid_indices = [idx for idx in indices_to_drop if idx < layer.weight.shape[1]]
                        if valid_indices:
                            # Zero out weights for pruned neurons
                            layer.weight.data[:, valid_indices] = 0
                            if hasattr(layer, "bias") and layer.bias is not None:
                                valid_bias_indices = [idx for idx in valid_indices if idx < layer.bias.shape[0]]
                                if valid_bias_indices:
                                    layer.bias.data[valid_bias_indices] = 0
                else:
                    # No activations captured
                    layer_indices.append(np.array([], dtype=np.int64))
            
            # Restore original weights after processing all layers for this fraction
            for i, layer in enumerate(weight_layers):
                layer.weight.data = original_weights[i].clone()
                if hasattr(layer, "bias") and layer.bias is not None and i in original_biases:
                    layer.bias.data = original_biases[i].clone()
            
            # Add indices for this dropout fraction
            dropout_indices[net_idx].append(layer_indices)
    
    return dropout_indices 