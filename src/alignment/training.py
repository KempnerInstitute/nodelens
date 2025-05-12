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
from alignment.utils.evaluation import evaluate_networks, evaluate_on_loader, evaluate_model
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
    Train a neural network. (Legacy wrapper around train_model)
    
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
        device = next(network.parameters()).device if hasattr(network, 'parameters') and next(iter(network.parameters()), None) is not None else _normalize_device(None)
    else:
        device = _normalize_device(device)
    
    _ensure_model_on_device(network, device)
    
    # Get data loader from DataSet, respecting batch_size override
    # DataSet.train_loader is created with dataset.batch_size by default.
    # If batch_size is specified here, it implies a different loader config for this specific call.
    # For simplicity in this refactor, we assume dataset.train_loader is used.
    # If batch_size override for train_network is critical, DataSet would need a method to get a loader with a specific batch_size.
    if batch_size is not None and batch_size != dataset.dataloader_parameters.get('batch_size'):
        logger.warning(f"train_network batch_size override ({batch_size}) is provided, but DataSet.train_loader uses {dataset.dataloader_parameters.get('batch_size')}. Using DataSet default.")
        # To truly use the override, one might do:
        # temp_loader_params = dataset.dataloader_parameters.copy()
        # temp_loader_params['batch_size'] = batch_size
        # train_loader = DataLoader(dataset.train_dataset, sampler=dataset.train_sampler, **temp_loader_params)
        # For now, stick to the dataset's pre-configured loader:
    train_loader = dataset.train_loader

    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(
        network.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Adapt the old progress_callback to the new callback system if provided
    # The old callback expects (epoch, epoch_loss, epoch_acc)
    # The new callback system in train_model provides a richer epoch_context dict.
    internal_callbacks = []
    if progress_callback:
        def wrapper_callback(epoch_context: Dict[str, Any]):
            # The old callback expects 0-indexed epoch for its first arg, train_model passes 1-indexed.
            progress_callback(
                epoch_context["epoch"] - 1, 
                epoch_context["train_loss"],
                epoch_context["train_accuracy"]
            )
        internal_callbacks.append(wrapper_callback)
    
    # Call train_model
    # train_model returns a comprehensive history. We need to adapt it.
    # It also handles its own logging and progress bar (via tqdm in train_loader).
    # We set return_history=True to get the history dict.
    # Assuming val_loader=None for train_network as it only reports training loss/acc.
    # Pass dataset.config_object for evaluation if train_model was to do validation.
    train_model_history = train_model(
        model=network,
        train_loader=train_loader,
        val_loader=None, # train_network doesn't do validation in its loop
        optimizer=optimizer,
        num_epochs=epochs,
        device=device,
        return_history=True,
        callbacks=internal_callbacks,
        dataset_config_for_eval=None # No validation here
    )

    # Adapt the returned history to the old format
    # Old format: {'train_loss': [], 'train_acc': []}
    # train_model_history format: {'train_loss': [], 'train_accuracy': [], 'val_loss': [], ...}
    history = {
        'train_loss': train_model_history.get('train_loss', []),
        'train_acc': train_model_history.get('train_accuracy', [])
    }
    
    # The original train_network logged per epoch. train_model also logs per epoch.
    # So, explicit logging here might be redundant if train_model's logging is sufficient.
    # For compatibility, the old progress_callback (if any) handles any specific per-epoch actions.
    
    return history


def test_network(
    network: AlignmentNetwork,
    dataset: DataSet,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Test a neural network. (Legacy wrapper around evaluate_on_loader)
    
    Args:
        network: The neural network to test
        dataset: Dataset to test on
        batch_size: Batch size (uses dataset default if None)
        device: Device to test on
    
    Returns:
        Dictionary with test metrics (loss, accuracy)
    """
    if device is None:
        device = next(network.parameters()).device if hasattr(network, 'parameters') and next(iter(network.parameters()), None) is not None else _normalize_device(None)
    else:
        device = _normalize_device(device)

    _ensure_model_on_device(network, device)
    network.eval() # Ensure eval mode

    # The original test_network created its own test_loader respecting batch_size.
    # evaluate_on_loader takes a data_loader. For consistency, we use dataset.test_loader.
    # If batch_size override for test_network is critical, similar logic as in train_network applies.
    if batch_size is not None and batch_size != dataset.dataloader_parameters.get('batch_size'):
        logger.warning(f"test_network batch_size override ({batch_size}) is provided, but DataSet.test_loader uses {dataset.dataloader_parameters.get('batch_size')}. Using DataSet default.")
    
    test_loader_to_use = dataset.test_loader

    # evaluate_on_loader returns {'loss': avg_loss, 'accuracy': accuracy}
    # which is compatible with what test_network needs to return, but keys are slightly different.
    # original test_network returned {'test_loss': avg_loss, 'test_acc': accuracy}
    # evaluate_on_loader is quiet by default regarding its own final summary if show_progress=False.
    # The original test_network had its own logger.info for the final result.
    
    eval_metrics = evaluate_on_loader(
        model=network, 
        data_loader=test_loader_to_use, 
        device=device, 
        show_progress=False # Keep it quiet, test_network does its own logging
    )
    
    # Adapt keys for return value
    results_to_return = {
        'test_loss': eval_metrics['loss'],
        'test_acc': eval_metrics['accuracy']
    }
    
    logger.info(f"Test - Loss: {results_to_return['test_loss']:.4f}, Accuracy: {results_to_return['test_acc']:.2f}%")
    
    return results_to_return


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
    Train and test a neural network. (Legacy wrapper)
    
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
    # Calls the now refactored train_network and test_network
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
    
    test_metrics = test_network(
        network=network,
        dataset=dataset,
        batch_size=batch_size,
        device=device
    )
    
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
    return_history: bool = False,
    callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    dataset_config_for_eval: Optional[Any] = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1
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
        callbacks: Optional list of callback functions to call at the end of each epoch. 
                   Each callback will receive a dictionary with epoch context.
        dataset_config_for_eval: Optional dataset configuration for the validation loader.
        ddp_rank: Rank of the current DDP process
        ddp_world_size: Total number of DDP processes
        
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
        
        is_main_process = (ddp_rank == 0)
        
        pbar_desc = f"Epoch {epoch+1}/{num_epochs}"
        # Adjust progress bar description for non-main DDP processes if they were to run it
        # if ddp_world_size > 1 and not is_main_process:
        #     pbar_desc = f"Epoch {epoch+1}/{num_epochs} (Rank {ddp_rank})"
        
        # Progress bar: only on main process (rank 0) if DDP is active and world_size > 1
        pbar_disabled = not is_main_process if ddp_world_size > 1 else False
        pbar = tqdm(train_loader, desc=pbar_desc, disable=pbar_disabled)
        
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
            
            # Update progress bar only for main process
            if is_main_process:
                pbar.set_postfix({
                    'loss': f"{total_loss / total:.4f}",
                    'acc': f"{100. * correct / total:.1f}%"
                })
        
        # Calculate epoch metrics (all processes do this, loss/acc are from their own data portion)
        epoch_loss = total_loss / total
        epoch_accuracy = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['learning_rate'].append(current_lr)
        
        # Evaluate on validation set if provided
        if val_loader is not None:
            # evaluate_model is called, it needs to be DDP aware or call DDP aware functions
            # Assume evaluate_model internally calls something like evaluate_on_loader which should be made DDP aware
            val_results = evaluate_model(
                model=model, # DDP model is passed if ddp active
                dataset_config=dataset_config_for_eval, # This should ideally be a DataSet object
                device=device,
                ddp_rank=ddp_rank, # Pass DDP info
                ddp_world_size=ddp_world_size
            )
            
            history['val_loss'].append(val_results['loss'])
            history['val_accuracy'].append(val_results['accuracy'])
            
            # Logging only on main process
            if is_main_process and (epoch == num_epochs - 1 or (epoch + 1) % 5 == 0):
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.2f}%, "
                          f"Val Loss={val_results['loss']:.4f}, Val Acc={val_results['accuracy']:.2f}%")
        else:
            # Logging only on main process
            if is_main_process and (epoch == num_epochs - 1 or (epoch + 1) % 5 == 0):
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Loss={epoch_loss:.4f}, Acc={epoch_accuracy:.2f}%")
        
        # Save checkpoint only on main process
        if is_main_process and checkpoint_dir is not None and (epoch + 1) % checkpoint_freq == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            
            # If model is DDP wrapped, save model.module.state_dict()
            model_state_to_save = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_accuracy': epoch_accuracy,
                'val_loss': history['val_loss'][-1] if val_loader is not None else None,
                'val_accuracy': history['val_accuracy'][-1] if val_loader is not None else None
            }, checkpoint_file)

        # Execute callbacks at the end of the epoch
        if callbacks:
            epoch_context = {
                "epoch": epoch + 1,
                "model": model,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
                "val_loss": history['val_loss'][-1] if val_loader is not None and history['val_loss'] else None,
                "val_accuracy": history['val_accuracy'][-1] if val_loader is not None and history['val_accuracy'] else None,
                "learning_rate": current_lr,
                "optimizer": optimizer,
                "history": history # Provides access to the full history so far
            }
            for callback_fn in callbacks:
                try:
                    callback_fn(epoch_context)
                except Exception as e:
                    logger.error(f"Error in callback function during epoch {epoch+1}: {e}", exc_info=True)
    
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


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to the checkpoint file
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load the checkpoint to
        ddp_rank: Rank of the current DDP process
        ddp_world_size: Total number of DDP processes
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint to CPU first to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    # If the model passed is DDP wrapped, load state to model.module.
    # Otherwise, load directly to the model.
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
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
def evaluate(
    nets: Union[nn.Module, List[nn.Module]], 
    dataset: Any, # Should be DataSet instance
    device: Optional[torch.device] = None,
    train_set: bool = False,
    measure_alignment: bool = False,
    alignment_methods: Optional[List[str]] = None,
    num_batches_for_eval: Optional[int] = None, # To limit data for faster eval
    # ADDED: Parameter to pass down for RQ scaling
    scale_rq_by_norm: bool = False,
    ddp_rank: int = 0,
    ddp_world_size: int = 1
) -> Dict[str, Any]:
    """
    Evaluate network(s) on a dataset, with options for alignment measurement.
    Args:
        nets: Neural network(s) to evaluate
        dataset: Dataset to evaluate on
        device: Device to evaluate on
        train_set: Whether to use training set (default: False)
        measure_alignment: Whether to measure alignment (default: False)
        alignment_methods: List of alignment methods to measure (default: ["RQ"])
        num_batches_for_eval: Number of batches to use for evaluation (default: None)
        scale_rq_by_norm: If True and RQ is an alignment method, scale its covariance by norm.
        ddp_rank: Rank of the current DDP process
        ddp_world_size: Total number of DDP processes
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
            if measure_alignment and isinstance(net, AlignmentNetwork):
                if not alignment_methods:
                    alignment_methods = ["RQ"] 
                
                # MODIFIED: Pass scale_by_norm_for_rq to measure_alignment_methods
                current_net_alignment_batch = net.measure_alignment_methods(
                    images, 
                    methods=alignment_methods, 
                    precomputed=False, 
                    scale_by_norm_for_rq=scale_rq_by_norm # Passed from evaluate function's params
                )
                align_data.append(current_net_alignment_batch)
            
            # Compute distributions of alignment values
            layer_dists = []
            for layer_dict in current_net_alignment_batch:
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
    
    is_main_process = (ddp_rank == 0)

    # Construct a similar return dict to the original `evaluate` if possible
    # Original returned e.g. {'accuracies': [...], 'losses': [...], 'alignment_scores': ...}
    final_results = {
        "loss": eval_results.get("mean_loss", eval_results.get("all_losses", [0.0])[0]),
        "accuracy": eval_results.get("mean_accuracy", eval_results.get("all_accuracies", [0.0])[0])
    }
    
    if is_main_process: # Log only on main process
        logger.info(f"Evaluation on {'train' if train_set else 'test'} set: Accuracy={final_results['accuracy']:.2f}%, Loss={final_results['loss']:.4f}")

    return final_results


def train_networks_fully_tensorized(
    networks: List[nn.Module],
    dataset,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    device=None,
    show_progress: bool = True,
    optimizer_class=torch.optim.Adam,
    weight_decay: float = 0.0,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
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
        ddp_rank: Rank of the current DDP process
        ddp_world_size: Total number of DDP processes
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
    callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    **optimizer_kwargs
) -> Dict[str, List[float]]:
    """
    Train multiple networks on the given dataset sequentially.
    This is also used for training a single network.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _normalize_device(device)

    # History structure to store average metrics across all networks per epoch
    # For detailed history of each network, it would be more complex or rely on callbacks saving their own data.
    aggregated_history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    all_individual_histories = [] # To store history from each train_model call

    net_iter_desc = "Training Networks Sequentially"
    net_iter = tqdm(networks, desc=net_iter_desc, leave=False) if show_progress and len(networks) > 1 else networks

    for net_idx, net in enumerate(net_iter):
        if show_progress and len(networks) > 1:
            net_iter.set_description(f"{net_iter_desc} (Network {net_idx+1}/{len(networks)})")

            _ensure_model_on_device(net, device)
        
        # Setup optimizer for the current network
        optimizer = optimizer_class(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_kwargs
        )

        # Use the dataset's train_loader and test_loader
        # train_model expects DataLoaders, DataSet has .train_loader and .test_loader
        # The original train_networks_sequential used dataset.train_loader directly.
        
        # Call train_model for the current network
        # train_model handles its own progress bar for epochs if show_progress is True (implicitly via tqdm in train_loader)
        # We might want to adjust progress bar descriptions if show_progress is True.
        # For simplicity, train_model's default progress bar will show.
        individual_history = train_model(
            model=net,
            train_loader=dataset.train_loader,
            val_loader=dataset.test_loader, # Assuming DataSet provides a test_loader for validation
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            checkpoint_dir=None, # Checkpointing handled at a higher level or disabled here
            return_history=True,
            callbacks=callbacks, # Pass down the callbacks
            dataset_config_for_eval=dataset.config_object, # MODIFIED: Use dataset.config_object
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size
        )
        all_individual_histories.append(individual_history)

    # Aggregate histories: For each epoch, average the metrics from all networks
    if all_individual_histories:
        num_epochs_trained = len(all_individual_histories[0]['train_loss']) # Assuming all trained for same epochs
        for epoch_idx in range(num_epochs_trained):
            epoch_train_losses = [h['train_loss'][epoch_idx] for h in all_individual_histories if len(h['train_loss']) > epoch_idx]
            epoch_train_accs = [h['train_accuracy'][epoch_idx] for h in all_individual_histories if len(h['train_accuracy']) > epoch_idx]
            epoch_val_losses = [h['val_loss'][epoch_idx] for h in all_individual_histories if h.get('val_loss') and len(h['val_loss']) > epoch_idx]
            epoch_val_accs = [h['val_accuracy'][epoch_idx] for h in all_individual_histories if h.get('val_accuracy') and len(h['val_accuracy']) > epoch_idx]

            if epoch_train_losses: aggregated_history['train_loss'].append(np.mean(epoch_train_losses))
            if epoch_train_accs: aggregated_history['train_acc'].append(np.mean(epoch_train_accs))
            if epoch_val_losses: aggregated_history['test_loss'].append(np.mean(epoch_val_losses)) # Renaming to test_loss for consistency
            if epoch_val_accs: aggregated_history['test_acc'].append(np.mean(epoch_val_accs)) # Renaming to test_acc

        if show_progress and aggregated_history['train_loss']: # Log final aggregated results
            logger.info(f"Sequential Training Avg (Epoch {num_epochs_trained}/{num_epochs}) - "
                        f"train_loss: {aggregated_history['train_loss'][-1]:.4f}, train_acc: {aggregated_history['train_acc'][-1]:.2f}%, "
                        f"test_loss: {aggregated_history['test_loss'][-1]:.4f}, test_acc: {aggregated_history['test_acc'][-1]:.2f}%")
            
    return aggregated_history


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
    callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
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
        callbacks: Optional list of callback functions to call at the end of each epoch for each model (primarily for sequential).
        ddp_rank: Rank of the current DDP process
        ddp_world_size: Total number of DDP processes
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
        "ddp_rank": ddp_rank,
        "ddp_world_size": ddp_world_size,
        **optimizer_kwargs
    }

    if selected_method == "fully_tensorized":
        if not same_architecture and len(networks) > 1:
            logger.warning("Fully_tensorized training requires all networks to have the same architecture. "
                           "Falling back to sequential training.")
            return train_networks_sequential(**common_args, callbacks=callbacks)
        try:
            logger.info(f"Using fully_tensorized training for {len(networks)} networks.")
            # Callbacks are tricky for fully_tensorized if they are model-specific.
            # For now, train_networks_fully_tensorized does not accept model-specific callbacks.
            # If general ensemble-level callbacks were needed, its signature would change.
            if callbacks:
                logger.warning("Model-specific callbacks are not currently supported with 'fully_tensorized' training method. Callbacks will be ignored.")
            return train_networks_fully_tensorized(**common_args)
        except Exception as e:
            logger.error(f"Fully_tensorized training failed: {str(e)}. Falling back to sequential.", exc_info=True)
            return train_networks_sequential(**common_args, callbacks=callbacks)
    elif selected_method == "sequential":
        logger.info(f"Using sequential training for {len(networks)} networks.")
        return train_networks_sequential(**common_args, callbacks=callbacks)
    else:
        logger.warning(f"Unknown training_method: {selected_method}. Defaulting to sequential training.")
        return train_networks_sequential(**common_args, callbacks=callbacks)

# Ensure the older single-network train_network and test_network functions are distinct
# if they are still used elsewhere. For now, the dispatcher `train_networks` is the main entry point
# for AlignmentExperiment.

# The original `train_network` (singular) can remain if it serves a purpose for single network training
# outside the multi-network replicate scenario handled by `AlignmentExperiment`.
# Same for `test_network` and `train_and_test`.
# `train_model` and `evaluate_model` also seem like general utility functions.
# `evaluate` is a more complex evaluation function, also seems distinct. 