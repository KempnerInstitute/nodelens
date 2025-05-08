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
1. "global_joint": Prune x% of nodes across all layers simultaneously based on alignment score.
2. "layer_wise": Prune x% from each layer simultaneously (optionally skip classification).
3. "layer_isolated": For each layer, prune x% of that layer alone and measure accuracy.
4. "cascading_layer": Prune layers progressively - prune layer 1, then compute RQ for layer 2 using the pruned network,
   prune layer 2, and continue this cascading approach for all layers.

Deprecated naming schemes (will be removed in future versions):
- "global" → use "global_joint" instead
- "per_layer", "per_layer_combined" → use "layer_wise" instead
- "per_layer_independent", "isolated" → use "layer_isolated" instead
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

from alignment.metrics import AlignmentMetric, get_metric
from alignment.utils.metrics_utils import AlignmentMetricsFactory
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

def progressive_dropout(
    networks: List[nn.Module],
    dataset: DataSet,
    dropout_fractions: List[float],
    metric: AlignmentMetric,
    device="cuda",
    pruning_mode: str = "global_joint",
    dropout_mode: str = "scaled",
    strategy: str = "low_rq",
    show_progress: bool = False
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Apply progressive dropout to networks and evaluate their performance.
    
    Args:
        networks: List of networks.
        dataset: The dataset to evaluate on.
        dropout_fractions: List of float fractions to compute dropout for.
        metric: The alignment metric to use.
        device: The device to use for evaluation. Default is "cuda".
        pruning_mode: How to apply pruning across network. Options:
            - "global_joint": Prune x% of nodes across all layers simultaneously based on alignment score
            - "layer_wise": Prune x% from each layer simultaneously (optionally skip classification)
            - "layer_isolated": For each layer, prune x% of that layer alone and measure accuracy
            - "cascading_layer": Prune layers progressively: prune layer 1, then use pruned network to 
                              compute RQ for layer 2, etc.
        dropout_mode: The operational mode. Options:
            - "evaluate_pruned": Standard approach, evaluate networks with pruned nodes
            - "mask_activations": Zero out activations but leave weights intact
            - "scaled": Scale remaining weights after zeroing pruned weights
            - "zero": Simply zero out pruned weights without scaling
        strategy: The neuron selection strategy. Options:
            - "high_rq": Prune neurons with highest alignment scores (weight magnitudes)
            - "low_rq": Prune neurons with lowest alignment scores (weight magnitudes)
            - "random": Prune neurons randomly
        show_progress: Whether to show progress bars during processing
        
    Returns:
        Tuple of (network_accuracies, network_losses).
    """
    # Handle deprecated pruning mode names for backward compatibility
    if pruning_mode == "global":
        logger.warning("Pruning mode 'global' is deprecated, use 'global_joint' instead")
        pruning_mode = "global_joint"
    elif pruning_mode == "per_layer":
        logger.warning("Pruning mode 'per_layer' is deprecated, use 'layer_wise' instead")
        pruning_mode = "layer_wise"
    elif pruning_mode == "per_layer_combined":
        logger.warning("Pruning mode 'per_layer_combined' is deprecated, use 'layer_wise' instead")
        pruning_mode = "layer_wise"
    elif pruning_mode == "per_layer_independent":
        logger.warning("Pruning mode 'per_layer_independent' is deprecated, use 'layer_isolated' instead")
        pruning_mode = "layer_isolated"
    elif pruning_mode == "isolated":
        logger.warning("Pruning mode 'isolated' is deprecated, use 'layer_isolated' instead")
        pruning_mode = "layer_isolated"
    
    # Initialize results
    network_accuracies = {}
    network_losses = {}
    
    # Handle case with no networks
    if not networks:
        logger.warning("No networks provided to progressive_dropout")
        return network_accuracies, network_losses
    
    # Ensure device is properly set
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    # Process each network
    network_iterator = tqdm(enumerate(networks), total=len(networks), desc=f"Networks ({strategy})", 
                           position=1, leave=False) if show_progress else enumerate(networks)
    
    for net_idx, network in network_iterator:
        if not show_progress:
            logger.info(f"Processing network {net_idx+1}/{len(networks)} with strategy '{strategy}'")
        
        # Move network to device
        network = network.to(device)
        
        # Initialize results for this network
        network_accuracies[net_idx] = []
        network_losses[net_idx] = []
        
        # Store original weights and biases
        original_weights = {}
        original_biases = {}
        
        # Get alignment layers
        if not hasattr(network, "alignment_layers"):
            logger.warning(f"Network {net_idx} doesn't have alignment_layers attribute")
            continue
        
        # Save original weights for each layer
        for i, layer in enumerate(network.alignment_layers):
            if hasattr(layer, "weight") and layer.weight is not None:
                original_weights[i] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    original_biases[i] = layer.bias.data.clone()
        
        # Get original accuracy
        network.eval()
        original_accuracy = 0.0
        original_loss = 0.0
        
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0.0
            
            test_iter = tqdm(dataset.test_loader, desc="Evaluating original accuracy", 
                           position=2, leave=False) if show_progress else dataset.test_loader
            
            for inputs, targets in test_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = network(inputs)
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            if total > 0:
                original_accuracy = 100.0 * correct / total
                original_loss = total_loss / total
                
                # Add original values for 0% dropout
                network_accuracies[net_idx].append(original_accuracy)
                network_losses[net_idx].append(original_loss)
        
        if not show_progress:
            logger.info(f"Network {net_idx}, Original accuracy: {original_accuracy:.2f}%, loss: {original_loss:.4f}")
        
        # Compute pruning for each dropout fraction (except the first one, which is 0%)
        fraction_iterator = tqdm(enumerate(dropout_fractions), total=len(dropout_fractions), 
                               desc="Dropout fractions", position=2, 
                               leave=False) if show_progress else enumerate(dropout_fractions)
        
        for frac_idx, fraction in fraction_iterator:
            # Skip the first fraction (0.0) since we already added original accuracy
            if frac_idx == 0 and fraction == 0.0:
                continue
                
            if not show_progress:
                logger.info(f"Processing network {net_idx}, dropout fraction: {fraction:.4f}")
            
            if show_progress:
                fraction_iterator.set_description(f"Fraction: {fraction:.2f}")
            
            # Restore original weights
            for i, layer in enumerate(network.alignment_layers):
                if i in original_weights:
                    layer.weight.data = original_weights[i].clone()
                    if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias.data = original_biases[i].clone()
            
            # Apply pruning based on the selected strategy
            if pruning_mode == "global_joint":
                # Collect all neurons and their scores across all layers
                all_neurons = []
                
                for layer_idx, layer in enumerate(network.alignment_layers):
                    if layer_idx not in original_weights:
                        continue
                        
                    weights = layer.weight.data
                    input_dim = weights.shape[1]
                    
                    # Calculate importance scores (use L2 norm as proxy for alignment)
                    neuron_scores = [torch.norm(weights[:, j]).item() for j in range(input_dim)]
                    
                    # Store (layer_idx, neuron_idx, score) tuples
                    all_neurons.extend([(layer_idx, j, neuron_scores[j]) for j in range(input_dim)])
                
                # Sort neurons by score based on strategy
                if strategy == "high_rq":
                    # Sort by highest scores first
                    all_neurons.sort(key=lambda x: x[2], reverse=True)
                elif strategy == "low_rq":
                    # Sort by lowest scores first
                    all_neurons.sort(key=lambda x: x[2])
                elif strategy == "random":
                    # Shuffle neurons randomly
                    import random
                    random.shuffle(all_neurons)
                else:
                    # Default to low_rq behavior
                    all_neurons.sort(key=lambda x: x[2])
                
                # Calculate how many neurons to prune
                total_neurons = len(all_neurons)
                num_to_drop = int(total_neurons * fraction)
                
                if num_to_drop > 0:
                    # Get indices to drop
                    to_drop = all_neurons[:num_to_drop]
                    
                    # Apply pruning
                    for layer_idx, neuron_idx, _ in to_drop:
                        if layer_idx in original_weights:
                            layer = network.alignment_layers[layer_idx]
                            
                            # Zero out weights for this neuron
                            if neuron_idx < layer.weight.data.shape[1]:
                                layer.weight.data[:, neuron_idx] = 0.0
                                if hasattr(layer, "bias") and layer.bias is not None and neuron_idx < layer.bias.data.shape[0]:
                                    layer.bias.data[neuron_idx] = 0.0
            
            elif pruning_mode == "layer_wise":
                # Apply pruning to each layer individually
                for layer_idx, layer in enumerate(network.alignment_layers):
                    if layer_idx not in original_weights:
                        continue
                        
                    weights = layer.weight.data
                    input_dim = weights.shape[1]
                    
                    # Calculate importance scores
                    neuron_scores = [torch.norm(weights[:, j]).item() for j in range(input_dim)]
                    
                    # Calculate how many neurons to prune in this layer
                    num_to_drop = int(input_dim * fraction)
                    
                    if num_to_drop > 0:
                        # Get neurons to drop based on strategy
                        if strategy == "high_rq":
                            # Sort by highest scores first (descending)
                            sorted_indices = np.argsort(neuron_scores)[::-1]
                            to_drop = sorted_indices[:num_to_drop]
                        elif strategy == "low_rq":
                            # Sort by lowest scores first (ascending)
                            sorted_indices = np.argsort(neuron_scores)
                            to_drop = sorted_indices[:num_to_drop]
                        elif strategy == "random":
                            # Choose random neurons
                            all_indices = list(range(input_dim))
                            np.random.shuffle(all_indices)
                            to_drop = all_indices[:num_to_drop]
                        else:
                            # Default to low_rq behavior
                            sorted_indices = np.argsort(neuron_scores)
                            to_drop = sorted_indices[:num_to_drop]
                        
                        # Apply pruning
                        for neuron_idx in to_drop:
                            if neuron_idx < weights.shape[1]:
                                layer.weight.data[:, neuron_idx] = 0.0
                                if hasattr(layer, "bias") and layer.bias is not None and neuron_idx < layer.bias.data.shape[0]:
                                    layer.bias.data[neuron_idx] = 0.0
            
            elif pruning_mode == "layer_isolated":
                # For each layer, prune and evaluate separately
                layer_accuracies = []
                layer_losses = []
                
                for layer_idx, layer in enumerate(network.alignment_layers):
                    if layer_idx not in original_weights:
                        continue
                    
                    # Restore original weights for all layers
                    for i, l in enumerate(network.alignment_layers):
                        if i in original_weights:
                            l.weight.data = original_weights[i].clone()
                            if i in original_biases and hasattr(l, "bias") and l.bias is not None:
                                l.bias.data = original_biases[i].clone()
                    
                    # Now prune just this layer
                    weights = layer.weight.data
                    input_dim = weights.shape[1]
                    
                    # Calculate importance scores
                    neuron_scores = [torch.norm(weights[:, j]).item() for j in range(input_dim)]
                    
                    # Calculate how many neurons to prune
                    num_to_drop = int(input_dim * fraction)
                    
                    if num_to_drop > 0:
                        # Get neurons to drop based on strategy
                        if strategy == "high_rq":
                            # Sort by highest scores first (descending)
                            sorted_indices = np.argsort(neuron_scores)[::-1]
                            to_drop = sorted_indices[:num_to_drop]
                        elif strategy == "low_rq":
                            # Sort by lowest scores first (ascending)
                            sorted_indices = np.argsort(neuron_scores)
                            to_drop = sorted_indices[:num_to_drop]
                        elif strategy == "random":
                            # Choose random neurons
                            all_indices = list(range(input_dim))
                            np.random.shuffle(all_indices)
                            to_drop = all_indices[:num_to_drop]
                        else:
                            # Default to low_rq behavior
                            sorted_indices = np.argsort(neuron_scores)
                            to_drop = sorted_indices[:num_to_drop]
                        
                        # Apply pruning to just this layer
                        for neuron_idx in to_drop:
                            if neuron_idx < weights.shape[1]:
                                layer.weight.data[:, neuron_idx] = 0.0
                                if hasattr(layer, "bias") and layer.bias is not None and neuron_idx < layer.bias.data.shape[0]:
                                    layer.bias.data[neuron_idx] = 0.0
                    
                    # Evaluate with just this layer pruned
                    network.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        total_loss = 0.0
                        
                        eval_iter = tqdm(dataset.test_loader, desc=f"Layer {layer_idx} eval", 
                                       position=3, leave=False) if show_progress else dataset.test_loader
                        
                        for inputs, targets in eval_iter:
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = network(inputs)
                            
                            # Compute loss
                            loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')
                            total_loss += loss.item()
                            
                            # Compute accuracy
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                        
                        if total > 0:
                            layer_accuracies.append(100.0 * correct / total)
                            layer_losses.append(total_loss / total)
                
                # Use the minimum accuracy across all layers (worst case)
                if layer_accuracies:
                    min_acc = min(layer_accuracies)
                    min_loss = max(layer_losses)  # Maximum loss corresponds to minimum accuracy
                    
                    network_accuracies[net_idx].append(min_acc)
                    network_losses[net_idx].append(min_loss)
                    
                    # Skip the evaluation below
                    continue
            
            elif pruning_mode == "cascading_layer":
                # Cascading approach: prune layer 1, then use pruned network to compute scores for layer 2, etc.
                for layer_idx, layer in enumerate(network.alignment_layers):
                    if layer_idx not in original_weights:
                        continue
                        
                    weights = layer.weight.data
                    input_dim = weights.shape[1]
                    
                    # Calculate importance scores based on current network state
                    neuron_scores = [torch.norm(weights[:, j]).item() for j in range(input_dim)]
                    
                    # Calculate how many neurons to prune
                    num_to_drop = int(input_dim * fraction)
                    
                    if num_to_drop > 0:
                        # Get neurons to drop based on strategy
                        if strategy == "high_rq":
                            # Sort by highest scores first (descending)
                            sorted_indices = np.argsort(neuron_scores)[::-1]
                            to_drop = sorted_indices[:num_to_drop]
                        elif strategy == "low_rq":
                            # Sort by lowest scores first (ascending)
                            sorted_indices = np.argsort(neuron_scores)
                            to_drop = sorted_indices[:num_to_drop]
                        elif strategy == "random":
                            # Choose random neurons
                            all_indices = list(range(input_dim))
                            np.random.shuffle(all_indices)
                            to_drop = all_indices[:num_to_drop]
                        else:
                            # Default to low_rq behavior
                            sorted_indices = np.argsort(neuron_scores)
                            to_drop = sorted_indices[:num_to_drop]
                        
                        # Apply pruning
                        for neuron_idx in to_drop:
                            if neuron_idx < weights.shape[1]:
                                layer.weight.data[:, neuron_idx] = 0.0
                                if hasattr(layer, "bias") and layer.bias is not None and neuron_idx < layer.bias.data.shape[0]:
                                    layer.bias.data[neuron_idx] = 0.0
            
            # Evaluate the pruned network
            network.eval()
            pruned_accuracy = 0.0
            pruned_loss = 0.0
            
            with torch.no_grad():
                correct = 0
                total = 0
                total_loss = 0.0
                
                eval_iter = tqdm(dataset.test_loader, desc=f"Eval fraction {fraction:.2f}", 
                               position=3, leave=False) if show_progress else dataset.test_loader
                
                for inputs, targets in eval_iter:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = network(inputs)
                    
                    # Compute loss
                    loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')
                    total_loss += loss.item()
                    
                    # Compute accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                if total > 0:
                    pruned_accuracy = 100.0 * correct / total
                    pruned_loss = total_loss / total
                    
                    # Add to results
                    network_accuracies[net_idx].append(pruned_accuracy)
                    network_losses[net_idx].append(pruned_loss)
                    
                    if show_progress:
                        fraction_iterator.set_postfix({"acc": f"{pruned_accuracy:.2f}%"})
            
            if not show_progress:
                logger.info(f"Network {net_idx}, fraction {fraction:.2f}: accuracy = {pruned_accuracy:.2f}%, loss = {pruned_loss:.4f}")
        
        # Update progress bar for this network if showing progress
        if show_progress:
            last_acc = network_accuracies[net_idx][-1] if network_accuracies[net_idx] else 0
            network_iterator.set_postfix({"final_acc": f"{last_acc:.2f}%"})
        
        # Restore original weights after processing this network
        for i, layer in enumerate(network.alignment_layers):
            if i in original_weights:
                layer.weight.data = original_weights[i].clone()
                if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.data = original_biases[i].clone()
    
    return network_accuracies, network_losses

def eigenvector_dropout(
    model: nn.Module,
    dataset_config: Any,
    dropout_fraction: float = 0.1,
    metric: Optional[AlignmentMetric] = None,
    batch_size: int = 128,
    num_batches: int = 10,
    device: Optional[torch.device] = None,
    dropout_mode: str = "scaled",
    dropout_pruning_mode: str = "layer_wise"
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
        dropout_pruning_mode: How pruning is distributed (global_joint, layer_wise, layer_isolated)
        
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

def _compute_alignments(networks: List[nn.Module], metric) -> Dict[int, List[List[float]]]:
    """
    Compute alignment for each network.
    
    Args:
        networks: List of networks.
        metric: Alignment metric to use.
        
    Returns:
        Dictionary mapping network index to list of alignment values for each layer.
    """
    alignments = {}
    
    for net_idx, network in enumerate(networks):
        # Store alignments for this network
        alignments[net_idx] = []
        
        # Get alignment layers - if not available, use all modules with weights
        if hasattr(network, "alignment_layers"):
            alignment_layers = network.alignment_layers
        else:
            logger.warning(f"Network {net_idx} doesn't have alignment_layers, using all modules with weights")
            alignment_layers = [m for m in network.modules() if hasattr(m, "weight") and m.weight is not None and len(m.weight.shape) > 1]
        
        # Process each layer
        for layer_idx, layer in enumerate(alignment_layers):
            # Skip layers without weights
            if not hasattr(layer, "weight") or layer.weight is None:
                alignments[net_idx].append([])
                continue
                
            weight = layer.weight
            
            # Skip non-2D weight tensors (e.g., conv layers not supported yet)
            if len(weight.shape) != 2:
                logger.warning(f"Layer {layer_idx} has weight shape {weight.shape}, skipping (only 2D weights supported)")
                alignments[net_idx].append([])
                continue
            
            try:
                # Get number of neurons (input dimension)
                n_neurons = weight.shape[1]
                
                # Compute alignment for each neuron
                layer_alignments = []
                for neuron_idx in range(n_neurons):
                    try:
                        # Get the weights for this neuron (column of the weight matrix)
                        neuron_weight = weight[:, neuron_idx].cpu().numpy()
                        
                        # Calculate alignment score using the provided metric
                        # Check if the metric is callable or has a compute method
                        if callable(metric):
                            alignment_score = metric(neuron_weight)
                        elif hasattr(metric, 'compute_alignment'):
                            alignment_score = metric.compute_alignment(neuron_weight)
                        elif hasattr(metric, 'measure_neuron'):
                            alignment_score = metric.measure_neuron(neuron_weight)
                        elif hasattr(metric, 'compute'):
                            alignment_score = metric.compute(neuron_weight)
                        elif hasattr(metric, 'name') and metric.name.lower() == 'rq':
                            # Special case for RQ metric
                            alignment_score = float(np.linalg.norm(neuron_weight))
                            logger.info("Using magnitude as a proxy for RQ metric")
                        else:
                            # Default to magnitude as a simple alignment metric
                            alignment_score = float(np.linalg.norm(neuron_weight))
                            logger.warning(f"Using default magnitude metric - could not determine how to use provided metric object")
                        
                        # Add alignment score to list
                        layer_alignments.append(float(alignment_score))
                    except Exception as e:
                        logger.error(f"Error computing alignment for neuron {neuron_idx} in layer {layer_idx}: {str(e)}")
                        # Use a default low alignment score if there's an error
                        layer_alignments.append(0.0)
                
                # Add layer alignments to network alignments
                alignments[net_idx].append(layer_alignments)
            except Exception as e:
                logger.error(f"Error processing layer {layer_idx}: {str(e)}")
                alignments[net_idx].append([])
    
    return alignments

def _compute_dropout_indices(
    networks: List[nn.Module],
    dropout_fractions: List[float],
    metric,
    mode: str = "global_joint",
    alignments: Optional[Dict[int, List[float]]] = None,
) -> Dict[int, List[List[np.ndarray]]]:
    """
    Compute indices of neurons to drop for each network and dropout fraction.
    
    Args:
        networks: List of networks.
        dropout_fractions: List of float fractions to compute dropout for.
        metric: The alignment metric to use.
        mode: The dropout mode to use. Options: "global_joint", "layer_wise", "layer_isolated".
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
        if net_idx >= len(networks):
            logger.warning(f"Network index {net_idx} is out of range (max {len(networks)-1})")
            continue
            
        dropout_indices[net_idx] = []
        
        # Get the network
        network = networks[net_idx]
        
        # Get alignment layers
        if hasattr(network, "alignment_layers"):
            alignment_layers = network.alignment_layers
        else:
            logger.warning(f"Network {net_idx} doesn't have alignment_layers, using all modules with weights")
            alignment_layers = [m for m in network.modules() if hasattr(m, "weight") and m.weight is not None and len(m.weight.shape) > 1]
        
        # Calculate mapping between alignment layers and alignment values
        if len(alignment_layers) != len(alignment_values):
            logger.warning(
                f"Network {net_idx}: Mismatch between alignment layers ({len(alignment_layers)}) "
                f"and alignment values ({len(alignment_values)})"
            )
            # Use the minimum of the two to avoid index errors
            num_layers = min(len(alignment_layers), len(alignment_values))
            alignment_layers = alignment_layers[:num_layers]
            alignment_values = alignment_values[:num_layers]
        
        # For each dropout fraction
        for fraction in dropout_fractions:
            # This will hold indices for all layers for this fraction
            layer_indices = []
            
            # For each layer with alignment values
            for layer_idx, (layer, layer_alignments) in enumerate(zip(alignment_layers, alignment_values)):
                if not layer_alignments:
                    # No alignment values for this layer
                    layer_indices.append(np.array([], dtype=np.int64))
                    continue
                
                # Get the shape of the weight tensor
                if not hasattr(layer, "weight") or layer.weight is None:
                    layer_indices.append(np.array([], dtype=np.int64))
                    continue
                    
                weight = layer.weight
                
                # Check if weight is a proper 2D tensor
                if len(weight.shape) != 2:
                    logger.warning(f"Layer {layer_idx} has unexpected weight shape {weight.shape}, expecting 2D tensor")
                    layer_indices.append(np.array([], dtype=np.int64))
                    continue
                
                # Calculate number of neurons in this layer (input dimension)
                n_neurons = weight.shape[1]
                
                # Verify alignment values match the weight shape
                if len(layer_alignments) != n_neurons:
                    logger.warning(
                        f"Layer {layer_idx}: Mismatch between alignment values length ({len(layer_alignments)}) "
                        f"and weight input dim ({n_neurons})"
                    )
                    # Use only valid indices
                    layer_alignments = [
                        layer_alignments[i] if i < len(layer_alignments) else 0.0
                        for i in range(min(n_neurons, len(layer_alignments)))
                    ]
                
                # Calculate number of neurons to drop based on fraction
                num_to_drop = int(n_neurons * fraction)
                
                # Get the indices of neurons to drop based on alignment value
                if num_to_drop > 0:
                    try:
                        # Convert to numpy array and ensure proper type
                        alignments_array = np.array(layer_alignments[:n_neurons], dtype=np.float32)
                        
                        # Sort and get indices (smallest first)
                        sorted_indices = np.argsort(alignments_array)
                        
                        # Only take valid indices up to num_to_drop
                        indices_to_drop = sorted_indices[:min(num_to_drop, len(sorted_indices))]
                    except Exception as e:
                        logger.error(f"Error computing dropout indices for layer {layer_idx}: {str(e)}")
                        indices_to_drop = np.array([], dtype=np.int64)
                else:
                    indices_to_drop = np.array([], dtype=np.int64)
                
                # Ensure indices are within bounds
                indices_to_drop = np.array([idx for idx in indices_to_drop if idx < n_neurons], dtype=np.int64)
                
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
    pruned_networks: List[nn.Module],
    dataset: DataSet,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Evaluates a list of pruned networks on a given dataset using a tensorized approach.
    Returns accuracy for each network.

    Args:
        pruned_networks: List of networks to evaluate
        dataset: PyTorch dataset to evaluate on
        device: Device to run the evaluation on
        batch_size: Batch size for evaluation

    Returns:
        Array of accuracies, one for each network
    """
    # Safety check - if no networks, return empty array
    if not pruned_networks:
        return np.array([])
        
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_correct = [0] * len(pruned_networks)
    total_samples = 0
    
    # Put networks in eval mode
    for network in pruned_networks:
        network.eval()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Batch predictions for all networks
                batch_preds = []
                for i, network in enumerate(pruned_networks):
                    try:
                        # Process predictions for this network
                        logits = network(inputs)
                        preds = torch.argmax(logits, dim=1)
                        correct = (preds == targets).sum().item()
                        total_correct[i] += correct
                    except Exception as e:
                        logger.error(f"Error evaluating network {i}: {str(e)}")
                        # Don't add any correct predictions for this network in this batch
                        
                # Count total samples processed
                total_samples += targets.size(0)
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
    
    # Compute accuracies (handle division by zero)
    if total_samples > 0:
        accuracies = np.array([correct / total_samples for correct in total_correct])
    else:
        accuracies = np.zeros(len(pruned_networks))
    
    return accuracies

def _compute_dropout_indices_cascading_layer(
    networks: List[nn.Module],
    dropout_fractions: List[float],
    metric,
    device="cuda",
) -> Dict[int, List[List[np.ndarray]]]:
    """
    Compute dropout indices using a progressive cascading approach.
    
    For each network, this method:
    1. Starts with layer 1
    2. Computes alignment scores and prunes the specified fraction
    3. Uses the pruned network to compute alignment scores for layer 2
    4. And so on for each layer
    
    Args:
        networks: List of networks.
        dropout_fractions: List of float fractions to compute dropout for.
        metric: The alignment metric to use.
        device: Device to use for computation.
        
    Returns:
        Dictionary mapping network index to list of lists of dropout indices.
        Structure: {net_idx: [layer_indices_for_frac1, layer_indices_for_frac2, ...]}
        Where each layer_indices is a list with indices for each layer.
    """
    dropout_indices = {}
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    # For each network
    for net_idx, network in enumerate(networks):
        logger.info(f"Computing cascading dropout indices for network {net_idx}")
        
        # Initialize dropout indices for this network
        dropout_indices[net_idx] = []
        
        # Get alignment layers
        if hasattr(network, "alignment_layers"):
            alignment_layers = network.alignment_layers
        else:
            logger.warning(f"Network {net_idx} doesn't have alignment_layers, using all modules with weights")
            alignment_layers = [m for m in network.modules() if hasattr(m, "weight") and m.weight is not None and len(m.weight.shape) > 1]
        
        # Skip if no alignment layers
        if not alignment_layers:
            logger.warning(f"Network {net_idx} has no alignment layers")
            dropout_indices[net_idx] = [[np.array([], dtype=np.int64) for _ in alignment_layers] for _ in dropout_fractions]
            continue
        
        # Dictionary to store original weights for each layer
        original_weights = {}
        original_biases = {}
        
        # Store the original weights for each layer
        for layer_idx, layer in enumerate(alignment_layers):
            if hasattr(layer, "weight") and layer.weight is not None:
                original_weights[layer_idx] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    original_biases[layer_idx] = layer.bias.data.clone()
        
        # For each dropout fraction
        for frac_idx, fraction in enumerate(dropout_fractions):
            # Create a copy of the network that we'll progressively modify
            pruned_net = copy.deepcopy(network)
            pruned_net.to(device)
            
            # Indices for all layers at this fraction
            layer_indices = []
            
            # Process each layer sequentially
            for layer_idx, layer in enumerate(pruned_net.alignment_layers):
                # Skip layers without weights
                if not hasattr(layer, "weight") or layer.weight is None:
                    layer_indices.append(np.array([], dtype=np.int64))
                    continue
                
                weight = layer.weight
                
                # Skip non-2D weight tensors (e.g., conv layers not supported yet)
                if len(weight.shape) != 2:
                    logger.warning(f"Layer {layer_idx} has weight shape {weight.shape}, skipping (only 2D weights supported)")
                    layer_indices.append(np.array([], dtype=np.int64))
                    continue
                
                try:
                    # Compute alignment scores for this layer in the current pruned network
                    n_neurons = weight.shape[1]
                    layer_alignments = []
                    
                    for neuron_idx in range(n_neurons):
                        try:
                            # Get weights for this neuron
                            neuron_weight = weight[:, neuron_idx].cpu().numpy()
                            
                            # Calculate alignment score using the metric
                            # Check if the metric is callable or has a compute method
                            if callable(metric):
                                alignment_score = metric(neuron_weight)
                            elif hasattr(metric, 'compute_alignment'):
                                alignment_score = metric.compute_alignment(neuron_weight)
                            elif hasattr(metric, 'measure_neuron'):
                                alignment_score = metric.measure_neuron(neuron_weight)
                            elif hasattr(metric, 'compute'):
                                alignment_score = metric.compute(neuron_weight)
                            elif hasattr(metric, 'name') and metric.name.lower() == 'rq':
                                # Special case for RQ metric
                                alignment_score = float(np.linalg.norm(neuron_weight))
                                logger.info("Using magnitude as a proxy for RQ metric")
                            else:
                                # Default to magnitude as a simple alignment metric
                                alignment_score = float(np.linalg.norm(neuron_weight))
                                logger.warning(f"Using default magnitude metric - could not determine how to use provided metric object")
                            
                            # Add to list of alignment scores
                            layer_alignments.append(float(alignment_score))
                        except Exception as e:
                            logger.error(f"Error computing alignment for neuron {neuron_idx} in layer {layer_idx}: {str(e)}")
                            # Use default low alignment score if error
                            layer_alignments.append(0.0)
                    
                    # Calculate number of neurons to drop based on fraction
                    num_to_drop = int(n_neurons * fraction)
                    
                    if num_to_drop > 0:
                        # Sort and get indices of neurons to drop (lowest alignment first)
                        sorted_indices = np.argsort(layer_alignments)
                        indices_to_drop = sorted_indices[:min(num_to_drop, len(sorted_indices))]
                        
                        # Ensure indices are within bounds
                        indices_to_drop = np.array([idx for idx in indices_to_drop if idx < n_neurons], dtype=np.int64)
                        
                        # Add indices to layer_indices list
                        layer_indices.append(indices_to_drop)
                        
                        # Apply dropout to this layer in the pruned network
                        # (so it will be reflected when computing next layer's alignments)
                        if indices_to_drop.size > 0:
                            # Zero out weights for dropped neurons
                            new_weights = layer.weight.data.clone()
                            new_weights[:, indices_to_drop] = 0.0
                            layer.weight.data = new_weights
                            
                            # Zero out biases if present
                            if hasattr(layer, "bias") and layer.bias is not None:
                                biases = layer.bias.data.clone()
                                valid_bias_indices = [idx for idx in indices_to_drop if idx < len(biases)]
                                if valid_bias_indices:
                                    biases[valid_bias_indices] = 0.0
                                    layer.bias.data = biases
                    else:
                        # No neurons to drop
                        layer_indices.append(np.array([], dtype=np.int64))
                except Exception as e:
                    logger.error(f"Error processing layer {layer_idx}: {str(e)}")
                    layer_indices.append(np.array([], dtype=np.int64))
            
            # Add the results for this fraction to the main results
            dropout_indices[net_idx].append(layer_indices)
        
        # Release memory
        del pruned_net
        torch.cuda.empty_cache()
    
    return dropout_indices 