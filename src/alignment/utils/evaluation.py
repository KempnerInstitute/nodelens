"""
Network evaluation utilities.

This module provides functions for evaluating neural networks on datasets,
with support for different metrics and configurations.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm

# Updated import to reflect moved utility functions
from alignment.utils.model_utils import _normalize_device

logger = logging.getLogger(__name__)


def evaluate_on_loader(
    model: nn.Module,
    data_loader,
    device="cuda",
    show_progress: bool = False
) -> Dict[str, float]:
    """
    Evaluate a model on the given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Normalize device
    device = _normalize_device(device)
    
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    loader_iter = tqdm(data_loader, desc="Evaluating") if show_progress else data_loader
    
    with torch.no_grad():
        for inputs, targets in loader_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            if show_progress:
                acc = 100.0 * correct / total
                loader_iter.set_postfix({'loss': f"{total_loss/total:.4f}", 'acc': f"{acc:.2f}%"})
    
    # Calculate final metrics
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate_networks(
    networks: List[nn.Module],
    data_loader,
    device="cuda"
) -> Tuple[float, float]:
    """
    Evaluate multiple networks on the given data loader.
    
    Args:
        networks: List of networks to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        
    Returns:
        Tuple of (average loss, average accuracy)
    """
    # Normalize device
    device = _normalize_device(device)
    
    total_loss = 0.0
    total_acc = 0.0
    
    for network in networks:
        metrics = evaluate_on_loader(network, data_loader, device, show_progress=False)
        total_loss += metrics['loss']
        total_acc += metrics['accuracy']
    
    avg_loss = total_loss / len(networks)
    avg_acc = total_acc / len(networks)
    
    return avg_loss, avg_acc 


# A simple ensemble for evaluation:
class EvaluationEnsemble(nn.Module):
    def __init__(self, networks_list: List[nn.Module]):
        super().__init__()
        # Ensure networks are properly registered as submodules
        self.networks = nn.ModuleList(networks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs_list = [network(x) for network in self.networks]
        return torch.stack(outputs_list, dim=0)

def evaluate_networks_ensemble(
    networks_to_evaluate: List[nn.Module],
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    # Use reduction='sum' for CrossEntropyLoss if averaging loss manually later by total samples
    # Or use reduction='mean' if criterion itself should average over batch for each network
    criterion: nn.Module = nn.CrossEntropyLoss(reduction='sum') 
) -> Tuple[List[float], List[float]]:
    """
    Evaluate a list of (already pruned) networks simultaneously on a dataset.
    Assumes all networks in the list have the same architecture and are on the target device.

    Args:
        networks_to_evaluate (List[nn.Module]): List of PyTorch models to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device to perform evaluation on.
        criterion (nn.Module): The loss function.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
                                         - Average losses per network.
                                         - Average accuracies per network (in percentage).
    """
    if not networks_to_evaluate:
        return [], []

    num_networks = len(networks_to_evaluate)
    
    # Networks should already be in eval() mode and on the correct device before calling this.
    # However, a safety check/assertion can be useful in practice.
    # For example:
    # for net in networks_to_evaluate:
    #     assert not net.training, "Network should be in eval mode for ensemble evaluation"
    #     assert next(net.parameters()).device == device, "Network not on correct device for ensemble evaluation"

    ensemble_model = EvaluationEnsemble(networks_to_evaluate).to(device)
    ensemble_model.eval() # Ensure the ensemble itself is in eval mode

    # Accumulators for sums, will average at the end
    sum_losses_per_network = torch.zeros(num_networks, device=device, dtype=torch.float64)
    sum_correct_per_network = torch.zeros(num_networks, device=device, dtype=torch.float64)
    total_samples_processed = 0

    with torch.no_grad():
        for inputs, targets in data_loader: # Consider tqdm(data_loader, desc="Ensemble Eval") if verbose
            inputs, targets = inputs.to(device), targets.to(device)
            current_batch_size = inputs.size(0)
            total_samples_processed += current_batch_size
            
            # ensemble_outputs: [num_networks, batch_size, num_classes]
            ensemble_outputs = ensemble_model(inputs)
            
            for i in range(num_networks):
                network_outputs = ensemble_outputs[i] # Shape: [batch_size, num_classes]
                
                # Calculate loss for this network's outputs for the current batch
                loss = criterion(network_outputs, targets) # criterion has reduction='sum'
                sum_losses_per_network[i] += loss.item()

                # Calculate correct predictions for this network for the current batch
                _, predicted_classes = network_outputs.max(1)
                sum_correct_per_network[i] += predicted_classes.eq(targets).sum().item()

    if total_samples_processed == 0:
        # Avoid division by zero if data_loader was empty
        avg_losses = [0.0] * num_networks
        avg_accuracies = [0.0] * num_networks
    else:
        avg_losses = (sum_losses_per_network / total_samples_processed).tolist()
        avg_accuracies = (100.0 * sum_correct_per_network / total_samples_processed).tolist()
    
    return avg_losses, avg_accuracies 