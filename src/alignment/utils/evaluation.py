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
import numpy as np
import torch.nn.functional as F

# Updated import to reflect moved utility functions
from alignment.utils.model_utils import _normalize_device

logger = logging.getLogger(__name__)


def evaluate_on_loader(
    model: nn.Module,
    data_loader,
    device="cuda",
    show_progress: bool = False,
    # --- NEW: DDP Parameters ---
    ddp_rank: int = 0,
    ddp_world_size: int = 1
    # --- End NEW ---
) -> Dict[str, float]:
    """
    Evaluate a model on the given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        show_progress: Whether to show progress bar
        ddp_rank: DDP rank, for rank-specific operations (progress bar).
        ddp_world_size: DDP world size.
        
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
    
    is_main_process = (ddp_rank == 0)
    # Disable progress bar for non-main DDP processes if DDP is active
    progress_bar_disabled = not is_main_process if (ddp_world_size > 1 and show_progress) else not show_progress
    
    loader_iter = tqdm(data_loader, desc="Evaluating", disable=progress_bar_disabled) 
    
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
            
            # Update progress bar only on the main process
            if show_progress and is_main_process:
                acc = 100.0 * correct / total if total > 0 else 0.0 # Ensure total > 0 for acc calculation
                loader_iter.set_postfix({'loss': f"{total_loss/total if total > 0 else 0.0:.4f}", 'acc': f"{acc:.2f}%"})
    
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
    device="cuda",
    # --- NEW: DDP Parameters ---
    ddp_rank: int = 0,
    ddp_world_size: int = 1
    # --- End NEW ---
) -> Tuple[float, float]:
    """
    Evaluate multiple networks on the given data loader.
    
    Args:
        networks: List of networks to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        ddp_rank: DDP rank.
        ddp_world_size: DDP world size.
        
    Returns:
        Tuple of (average loss, average accuracy)
    """
    # Normalize device
    device = _normalize_device(device)
    
    total_loss = 0.0
    total_acc = 0.0
    
    for network_idx, network in enumerate(networks):
        # Pass DDP parameters to evaluate_on_loader
        # Progress bar within evaluate_on_loader will be rank-aware
        metrics = evaluate_on_loader(network, data_loader, device, show_progress=False, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size)
        total_loss += metrics['loss']
        total_acc += metrics['accuracy']
    
    avg_loss = total_loss / len(networks) if networks else 0.0
    avg_acc = total_acc / len(networks) if networks else 0.0
    
    # This function doesn't log itself, relies on caller.
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
    criterion: nn.Module = nn.CrossEntropyLoss(reduction='sum'),
    # --- NEW: DDP Parameters (ensemble eval is typically single-process but good for signature consistency) ---
    ddp_rank: int = 0,
    ddp_world_size: int = 1
    # --- End NEW ---
) -> Tuple[List[float], List[float]]:
    """
    Evaluate a list of (already pruned) networks simultaneously on a dataset.
    Assumes all networks in the list have the same architecture and are on the target device.

    Args:
        networks_to_evaluate (List[nn.Module]): List of PyTorch models to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device to perform evaluation on.
        criterion (nn.Module): The loss function.
        ddp_rank: DDP rank.
        ddp_world_size: DDP world size.

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

    is_main_process = (ddp_rank == 0)
    # Progress bar for ensemble evaluation, only on main process
    progress_bar_disabled = not is_main_process if ddp_world_size > 1 else False
    batch_iterator = tqdm(data_loader, desc="Ensemble Eval", disable=progress_bar_disabled)

    with torch.no_grad():
        for inputs, targets in batch_iterator: # Consider tqdm(data_loader, desc="Ensemble Eval") if verbose
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

def _evaluate_model_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Helper function to evaluate a model's accuracy on a given data loader.
    This is used internally by progressive_dropout and other functions.
    Args:
        model: The PyTorch model to evaluate.
        data_loader: DataLoader for the evaluation data.
        device: The device to perform evaluation on.
    Returns:
        Accuracy in percentage.
    """
    model.eval() # Ensure model is in evaluation mode
    # _ensure_model_on_device(model, device) # Assuming model is already on correct device by caller
    # Caller of this low-level util should ensure device consistency.
    # For safety, can add: model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                # Common case: model returns (predictions, other_stuff)
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    if total == 0:
        return 0.0 # Avoid division by zero
    return 100.0 * correct / total 

def evaluate_model(
    model: nn.Module,
    dataset_config: Any, # Or DatasetConfig if imported
    device: Optional[torch.device] = None,
    loader_name: str = 'test_loader',
    # extra_config: Optional[Any] = None, # Assuming this might refer to ExperimentConfig.extra or similar
    with_alignment: bool = False,
    # Adding metric_name and num_batches here if they were from extra_config or implicit
    metric_name_for_eval: str = "RQ",
    num_batches_for_eval: Optional[int] = None, # For sampling if loader is large
    show_progress: bool = True,
    # --- NEW: DDP Parameters ---
    ddp_rank: int = 0,
    ddp_world_size: int = 1
    # --- End NEW ---
) -> Dict[str, Any]:
    """
    Evaluate a neural network model.
    Args:
        model: Model to evaluate
        dataset_config: Configuration for the dataset (can be DatasetConfig instance or dict)
        device: Device to evaluate on
        loader_name: Name of the loader to use ('test_loader', 'val_loader', etc.)
        with_alignment: Whether to measure alignment metrics.
        metric_name_for_eval: Name of the alignment metric if with_alignment is True.
        num_batches_for_eval: Number of batches from loader to use for evaluation (None for all).
        show_progress: Whether to display a progress bar during evaluation.
        ddp_rank: DDP rank.
        ddp_world_size: DDP world size.
    Returns:
        Dictionary containing evaluation metrics (loss, accuracy, optional alignment).
    """
    from alignment.datasets import load_dataset # Moved import here for clarity
    from alignment.metrics import get_metric # Moved import here

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration: # Model has no parameters
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _normalize_device(device)
    model.to(device)
    model.eval()

    # Load dataset
    batch_size_from_config = None
    if hasattr(dataset_config, 'batch_size'): batch_size_from_config = dataset_config.batch_size
    elif isinstance(dataset_config, dict): batch_size_from_config = dataset_config.get('batch_size')
    final_batch_size = batch_size_from_config if batch_size_from_config is not None else 128

    dataset = load_dataset(dataset_config, batch_size=final_batch_size, device=device, 
                           use_ddp=(ddp_world_size > 1), ddp_rank=ddp_rank, ddp_world_size=ddp_world_size) # Pass DDP info
    
    loader = getattr(dataset, loader_name, None)
    if loader is None:
        raise ValueError(f"Data loader '{loader_name}' not found in dataset object created from config: {dataset_config}")
    
    # If num_batches_for_eval is specified, create a sub-loader
    if num_batches_for_eval is not None and num_batches_for_eval > 0:
        sub_dataset = torch.utils.data.Subset(loader.dataset, range(min(num_batches_for_eval * loader.batch_size, len(loader.dataset))))
        final_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=loader.batch_size, sampler=None, shuffle=False) # no sampler for subset
    else:
        final_loader = loader

    total_loss = 0.0
    correct = 0
    total_samples = 0
    alignment_metric_instance = None
    collected_alignment_values = [] # For storing batch_alignment if with_alignment is True

    is_main_process = (ddp_rank == 0)
    progress_bar_disabled_eval_model = not is_main_process if (ddp_world_size > 1 and show_progress) else not show_progress

    if with_alignment:
        alignment_metric_instance = get_metric(metric_name_for_eval)
        if not isinstance(model, AlignmentNetwork):
            logger.warning("Alignment measurement requested, but model is not an AlignmentNetwork. Skipping alignment.")
            with_alignment = False # Disable if not an AlignmentNetwork

    with torch.no_grad():
        for inputs, targets in tqdm(final_loader, desc=f"Evaluating on {loader_name}", leave=False, disable=progress_bar_disabled_eval_model):
            inputs, targets = inputs.to(device), targets.to(device)
            
            current_batch_size = inputs.size(0)
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = F.cross_entropy(outputs, targets, reduction='sum') # Sum loss for batch
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_samples += current_batch_size
            
            if with_alignment and alignment_metric_instance and hasattr(model, 'measure_alignment'):
                # Ensure model.forward was called in a way that populates hidden activations if metric needs them
                # For some metrics, just the inputs and weights are needed. measure_alignment might handle this.
                # This part might need model.forward(inputs, store_hidden=True) called before if metric needs it.
                # For simplicity, assuming measure_alignment can trigger necessary hooks if called on AlignmentNetwork.
                # The AlignmentNetwork.measure_alignment itself calls get_layer_inputs which can store_hidden.
                try:
                    # Alignment measurement should ideally run on rank 0 only or results gathered.
                    # For now, assume it's okay for all ranks to compute if model is DDP (weights are same).
                    # But if it involves data collection (hooks), it should be on one rank or handled carefully.
                    # If measure_alignment_methods from AlignmentNetwork is DDP-aware or called on non-DDP model on rank 0:
                    if is_main_process: # Only measure alignment on main process to avoid redundant computation/hook issues
                        batch_alignment = model.measure_alignment(inputs, method=metric_name_for_eval, precomputed=False) 
                        collected_alignment_values.append(batch_alignment) 
                except Exception as e:
                    logger.error(f"Error measuring alignment during evaluation for batch: {e}", exc_info=True) # Changed from debug_mode
    
    metrics_results = {}
    if total_samples > 0:
        metrics_results['loss'] = total_loss / total_samples
        metrics_results['accuracy'] = 100. * correct / total_samples
    else:
        metrics_results['loss'] = float('inf')
        metrics_results['accuracy'] = 0.0
    
    # Averaging and reporting alignment only on main process
    if is_main_process and with_alignment and collected_alignment_values:
        # Average alignment values across batches for each layer
        # collected_alignment_values is a list of lists: [[L0_b1, L1_b1,...], [L0_b2, L1_b2,...]]
        num_layers_alignment = len(collected_alignment_values[0]) if collected_alignment_values else 0
        avg_alignment_per_layer = []
        for i in range(num_layers_alignment):
            layer_specific_alignments = [batch_align_list[i] for batch_align_list in collected_alignment_values if i < len(batch_align_list)]
            # Each element in layer_specific_alignments can be a tensor (e.g. node scores) or a scalar (layer score)
            # If it's a tensor of node scores, we might want to average them first
            processed_layer_alignments = []
            for item in layer_specific_alignments:
                if isinstance(item, torch.Tensor):
                    processed_layer_alignments.append(item.mean().item()) # Example: mean of node scores
                elif isinstance(item, (float, int)):
                    processed_layer_alignments.append(item)
            
            if processed_layer_alignments:
                avg_alignment_per_layer.append(np.mean(processed_layer_alignments))
            else:
                avg_alignment_per_layer.append(np.nan)
        metrics_results['alignment'] = avg_alignment_per_layer
    
    if is_main_process: # Log final results only on main process
        logger.info(f"Evaluation on {loader_name}: Loss={metrics_results['loss']:.4f}, Acc={metrics_results['accuracy']:.2f}%")
        if 'alignment' in metrics_results:
            alignment_str = ', '.join([f"{val:.4f}" for val in metrics_results['alignment']])
            logger.info(f"Avg Alignment ({metric_name_for_eval}): [{alignment_str}]")
    
    return metrics_results


@torch.no_grad()
def evaluate(
    nets: Union[nn.Module, List[nn.Module]], 
    dataset: Any, # Should be DataSet instance
    device: Optional[torch.device] = None,
    train_set: bool = False,
    measure_alignment: bool = False,
    alignment_methods: Optional[List[str]] = None,
    # measure_expected: bool = True, # This was part of original, seems complex for general eval
    # bins: int = 50 # Also for expected distribution
    num_batches_for_eval: Optional[int] = None, # To limit data for faster eval
    # --- NEW: DDP Parameters ---
    ddp_rank: int = 0,
    ddp_world_size: int = 1
    # --- End NEW ---
) -> Dict[str, Any]:
    """
    Evaluate network(s) on a dataset, with options for alignment measurement.
    Args:
        nets: Neural network or list of neural networks to evaluate.
        dataset: DataSet object to evaluate on.
        device: Device to use.
        train_set: Whether to use the training set instead of the test set.
        measure_alignment: If True, compute alignment metrics for the network(s).
        alignment_methods: List of alignment method names to compute if measure_alignment is True (e.g., ["RQ", "MI"]).
        num_batches_for_eval: Number of batches to use for this evaluation (None for all).
        ddp_rank: DDP rank.
        ddp_world_size: DDP world size.
    Returns:
        Dictionary of evaluation results. If multiple nets, results are lists under keys.
    """
    from alignment.metrics import AlignmentMetrics # Local import for class, consider top-level
    from alignment.models.base import AlignmentNetwork # For isinstance check

    if not isinstance(nets, list):
        nets = [nets]
    if not nets:
        return {}

    num_nets = len(nets)
    if device is None:
        device = getattr(dataset, 'device', _normalize_device(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    device = _normalize_device(device)

    # DDP: This function assumes models are already on their correct DDP devices if DDP is active.
    # The DDP wrapping (which moves model to device) happens in AlignmentExperiment.create_networks.
    for net in nets:
        # net.to(device) # Should already be on correct device due to DDP wrapping or _ensure_model_on_device
        net.eval()

    # DataSet.load_dataset creates DDP-aware dataloaders if `distributed=True` is passed.
    # The `dataset` object here should have been created with DDP awareness if applicable.
    dataloader = dataset.train_loader if train_set else dataset.test_loader
    if dataloader.sampler is not None and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(0) # Set epoch for sampler if DDP

    if num_batches_for_eval is not None and num_batches_for_eval > 0:
        # Note: Using torch.utils.data.Subset with DistributedSampler needs careful handling.
        # The sampler indices might not align with subset indices unless subset is created per rank.
        # For simplicity, if subsetting, assume it's for single process or results are approximate for DDP.
        # A more robust way for DDP subsetting would be to limit iterations.
        if ddp_world_size > 1:
            logger.warning("Subsetting dataloader with num_batches_for_eval and DDP might lead to skewed data per rank. Consider iterating for num_batches directly.")
        sub_dataset = torch.utils.data.Subset(dataloader.dataset, range(min(num_batches_for_eval * dataloader.batch_size, len(dataloader.dataset))))
        # If DDP, sampler should be for the subset, or iterate `num_batches_for_eval` times.
        # For now, let DDP sampler work on full dataset and we just iterate less if num_batches_for_eval is set.
        # This means each rank processes a subset of its DDP-assigned shard.
        actual_sampler = dataloader.sampler # Keep original DDP sampler if any
        dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=dataloader.batch_size, shuffle=False, sampler=None if actual_sampler else None)
        # If actual_sampler was DDP, the new loader doesn't have it, which is problematic.
        # Better: limit loop iterations instead of creating subset loader for DDP.

    results = {}
    all_losses = [[] for _ in range(num_nets)]
    all_accuracies = [[] for _ in range(num_nets)]
    # For alignment: results["alignment_metrics"][net_idx][method_name][layer_idx] = value
    # Or simpler: results["alignment_metrics"][net_idx][layer_idx] = {method1: val, method2: val}
    # Let's use: results["alignment_metrics"] = [{method: [layer_scores_net_1]}, {method: [layer_scores_net_2]}]
    if measure_alignment:
        results["alignment_metrics"] = [{} for _ in range(num_nets)]

    is_main_process = (ddp_rank == 0)
    progress_bar_disabled_evaluate = not is_main_process if ddp_world_size > 1 else False
    actual_dataloader_iterator = iter(dataloader)
    
    # Loop for specified number of batches if num_batches_for_eval is set
    num_batches_to_process = num_batches_for_eval if num_batches_for_eval is not None else len(dataloader)

    for batch_iter_idx in tqdm(range(num_batches_to_process), desc=f"Evaluating ({'train' if train_set else 'test'} set)", leave=False, disable=progress_bar_disabled_evaluate):
        try:
            inputs, targets = next(actual_dataloader_iterator)
        except StopIteration:
            break # Should not happen if num_batches_to_process <= len(dataloader)
        inputs, targets = dataset.unwrap_batch((inputs, targets), device=device)
        
        for idx, net in enumerate(nets):
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = dataset.measure_loss(outputs, targets, reduction="mean") # Use dataset's loss, mean over batch
            all_losses[idx].append(loss.item())
            
            # Use dataset.measure_accuracy for consistency if it exists, else simple acc
            if hasattr(dataset, 'measure_accuracy'):
                acc = dataset.measure_accuracy(outputs, targets, percentage=True)
            else:
                _, predicted = outputs.max(1)
                acc = (predicted == targets).float().mean().item() * 100.0
            all_accuracies[idx].append(acc.item() if isinstance(acc, torch.Tensor) else acc)

            if measure_alignment and isinstance(net, AlignmentNetwork):
                # Resolve alignment_methods before using in the main DDP-aware block
                resolved_alignment_methods = alignment_methods
                if not resolved_alignment_methods: 
                    resolved_alignment_methods = ["RQ"] # Default to RQ if none specified
                
                # Alignment measurement only on main process to avoid redundant work / hook issues with DDP models
                if is_main_process: 
                    model_to_measure = net.module if isinstance(net, nn.parallel.DistributedDataParallel) else net
                    if isinstance(model_to_measure, AlignmentNetwork): 
                        current_net_alignment_batch = model_to_measure.measure_alignment_methods(
                            inputs, methods=resolved_alignment_methods, precomputed=False
                        )
                        # Store once (e.g., from first batch) or average across batches.
                        # Current storage is once.
                        if idx not in results["alignment_metrics"] or not results["alignment_metrics"][idx]:
                            results["alignment_metrics"][idx] = current_net_alignment_batch 
                    else:
                        logger.warning(f"Net {idx} is DDP-wrapped but module is not AlignmentNetwork. Cannot measure alignment.")

    # Finalize metrics
    results["loss"] = [np.mean(losses) if losses else np.nan for losses in all_losses]
    results["accuracy"] = [np.mean(accs) if accs else np.nan for accs in all_accuracies]
    if num_nets == 1:
        results["loss"] = results["loss"][0]
        results["accuracy"] = results["accuracy"][0]
        if measure_alignment and results.get("alignment_metrics"):
            results["alignment_metrics"] = results["alignment_metrics"][0]

    if is_main_process: # Log final results only on main process
        logger.info(f"Final Evaluation ({'train' if train_set else 'test'} set):")
        if num_nets == 1:
            logger.info(f"  Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.2f}%")
            if measure_alignment and results.get("alignment_metrics"):
                alignment_data_to_log = results["alignment_metrics"]
                if isinstance(alignment_data_to_log, list) and len(alignment_data_to_log) > 0: # Check if it is a list (for multiple nets)
                    alignment_data_to_log = alignment_data_to_log[0] # Get data for the first net if single net case was not flattened

                if isinstance(alignment_data_to_log, list): # It should be list of layer_data dicts
                    for layer_idx, method_data in enumerate(alignment_data_to_log):
                        log_str = f"  Layer {layer_idx} Alignments: "
                        for method, val in method_data.items():
                            log_str += f"{method}: {val.mean().item() if isinstance(val, torch.Tensor) else val:.4f} "
                        logger.info(log_str)
                elif isinstance(alignment_data_to_log, dict): # Should not happen if num_nets=1 was handled
                    logger.warning("Unexpected alignment_metrics structure for single net logging.")
        else:
            for i in range(num_nets):
                logger.info(f"  Net {i}: Loss: {results['loss'][i]:.4f}, Accuracy: {results['accuracy'][i]:.2f}%")
            # TODO: Add DDP-aware logging for alignment metrics for multiple networks if needed.
            # Current alignment storage results["alignment_metrics"][net_idx] can be used.
    
    return results 