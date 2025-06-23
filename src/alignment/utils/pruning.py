"""
Pruning utilities for neural network alignment analysis.

This module provides various pruning strategies and utilities for analyzing
the effect of pruning on network alignment.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Callable
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""
    strategy: str = "magnitude"  # magnitude, random, structured, gradient, sensitivity
    amount: float = 0.5  # Fraction of parameters to prune
    structured: bool = False  # Whether to prune entire channels/filters
    iterative: bool = False  # Whether to prune iteratively
    global_pruning: bool = False  # Whether to prune globally across layers


class PruningUtilities:
    """
    Collection of pruning utilities for neural networks.
    """
    
    @staticmethod
    def get_pruning_mask_magnitude(
        weights: torch.Tensor,
        amount: float,
        structured: bool = False,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get pruning mask based on weight magnitudes.
        
        Args:
            weights: Weight tensor to prune
            amount: Fraction of weights to prune (0 to 1)
            structured: Whether to prune entire structures
            dim: Dimension along which to prune (for structured pruning)
            
        Returns:
            Binary mask (1 = keep, 0 = prune)
        """
        if structured and dim is not None:
            # Compute importance scores for each structure
            importance = weights.abs().sum(dim=dim, keepdim=True)
            # Flatten importance scores
            importance_flat = importance.flatten()
            # Get threshold
            k = int(amount * importance_flat.numel())
            threshold = importance_flat.kthvalue(k).values
            # Create mask
            mask = importance > threshold
            # Broadcast mask to original shape
            shape = [1] * weights.ndim
            shape[dim] = weights.shape[dim]
            mask = mask.reshape(shape).expand_as(weights)
        else:
            # Magnitude-based pruning
            weights_flat = weights.abs().flatten()
            k = int(amount * weights_flat.numel())
            threshold = weights_flat.kthvalue(k).values
            mask = weights.abs() > threshold
        
        return mask.float()
    
    @staticmethod
    def get_pruning_mask_random(
        weights: torch.Tensor,
        amount: float,
        structured: bool = False,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get random pruning mask.
        
        Args:
            weights: Weight tensor to prune
            amount: Fraction of weights to prune (0 to 1)
            structured: Whether to prune entire structures
            dim: Dimension along which to prune (for structured pruning)
            
        Returns:
            Binary mask (1 = keep, 0 = prune)
        """
        if structured and dim is not None:
            # Random structured pruning
            num_structures = weights.shape[dim]
            num_to_prune = int(amount * num_structures)
            indices = torch.randperm(num_structures)[:num_to_prune]
            
            mask = torch.ones_like(weights)
            if dim == 0:
                mask[indices] = 0
            elif dim == 1:
                mask[:, indices] = 0
            elif dim == 2:
                mask[:, :, indices] = 0
            elif dim == 3:
                mask[:, :, :, indices] = 0
        else:
            # Random unstructured pruning
            mask = torch.rand_like(weights) > amount
        
        return mask.float()
    
    @staticmethod
    def get_pruning_mask_gradient(
        weights: torch.Tensor,
        gradients: torch.Tensor,
        amount: float
    ) -> torch.Tensor:
        """
        Get pruning mask based on gradient information.
        
        Prunes weights with smallest gradient magnitudes (least important for task).
        
        Args:
            weights: Weight tensor
            gradients: Gradient tensor
            amount: Fraction to prune
            
        Returns:
            Binary mask
        """
        importance = (weights * gradients).abs()
        importance_flat = importance.flatten()
        k = int(amount * importance_flat.numel())
        threshold = importance_flat.kthvalue(k).values
        mask = importance > threshold
        
        return mask.float()
    
    @staticmethod
    def get_pruning_mask_sensitivity(
        model: nn.Module,
        layer: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        amount: float,
        loss_fn: Callable,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Get pruning mask based on sensitivity analysis.
        
        Prunes weights that have least impact on loss when removed.
        
        Args:
            model: Full model
            layer: Layer to prune
            dataloader: Data for sensitivity analysis
            amount: Fraction to prune
            loss_fn: Loss function
            device: Device to use
            
        Returns:
            Binary mask
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Get original weights
        original_weights = layer.weight.data.clone()
        
        # Compute sensitivity scores
        sensitivity = torch.zeros_like(original_weights)
        
        # For each weight, compute loss increase when zeroed
        with torch.no_grad():
            # Baseline loss
            baseline_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                baseline_loss += loss_fn(outputs, targets).item()
            
            # Compute sensitivity for each weight
            for i in range(original_weights.shape[0]):
                for j in range(original_weights.shape[1]):
                    # Zero out weight
                    layer.weight.data[i, j] = 0
                    
                    # Compute new loss
                    new_loss = 0
                    for inputs, targets in dataloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        new_loss += loss_fn(outputs, targets).item()
                    
                    # Sensitivity is loss increase
                    sensitivity[i, j] = new_loss - baseline_loss
                    
                    # Restore weight
                    layer.weight.data[i, j] = original_weights[i, j]
        
        # Create mask based on sensitivity
        sensitivity_flat = sensitivity.flatten()
        k = int(amount * sensitivity_flat.numel())
        threshold = sensitivity_flat.kthvalue(k).values
        mask = sensitivity > threshold
        
        return mask.float()
    
    @staticmethod
    def apply_pruning_mask(layer: nn.Module, mask: torch.Tensor):
        """
        Apply pruning mask to a layer.
        
        Args:
            layer: Layer to prune
            mask: Binary mask to apply
        """
        if hasattr(layer, 'weight'):
            layer.weight.data *= mask
            
            # Register mask as buffer for forward passes
            if not hasattr(layer, 'weight_mask'):
                layer.register_buffer('weight_mask', mask)
            else:
                layer.weight_mask = mask
            
            # Hook to apply mask during forward pass
            def apply_mask_hook(module, inputs):
                module.weight.data *= module.weight_mask
            
            # Remove old hook if exists
            if hasattr(layer, '_pruning_hook'):
                layer._pruning_hook.remove()
            
            # Register new hook
            layer._pruning_hook = layer.register_forward_pre_hook(apply_mask_hook)
    
    @staticmethod
    def remove_pruning(layer: nn.Module):
        """
        Remove pruning from a layer (make pruning permanent).
        
        Args:
            layer: Layer to remove pruning from
        """
        if hasattr(layer, 'weight_mask'):
            # Apply mask permanently
            layer.weight.data *= layer.weight_mask
            # Remove mask buffer
            delattr(layer, 'weight_mask')
            
        # Remove hook
        if hasattr(layer, '_pruning_hook'):
            layer._pruning_hook.remove()
            delattr(layer, '_pruning_hook')
    
    @staticmethod
    def get_sparsity(layer: nn.Module) -> float:
        """
        Get sparsity level of a layer.
        
        Args:
            layer: Layer to check
            
        Returns:
            Fraction of zero weights
        """
        if hasattr(layer, 'weight'):
            total = layer.weight.numel()
            zeros = (layer.weight == 0).sum().item()
            return zeros / total
        return 0.0
    
    @staticmethod
    def iterative_magnitude_pruning(
        model: nn.Module,
        amount: float,
        iterations: int = 10,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        fine_tune_epochs: int = 0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None
    ) -> List[float]:
        """
        Perform iterative magnitude pruning.
        
        Args:
            model: Model to prune
            amount: Total amount to prune
            iterations: Number of pruning iterations
            dataloader: Data for fine-tuning between iterations
            fine_tune_epochs: Epochs of fine-tuning between pruning
            optimizer: Optimizer for fine-tuning
            loss_fn: Loss function for fine-tuning
            
        Returns:
            List of accuracies after each iteration
        """
        amount_per_iteration = 1 - (1 - amount) ** (1 / iterations)
        accuracies = []
        
        for iteration in range(iterations):
            # Prune each layer
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    current_sparsity = PruningUtilities.get_sparsity(module)
                    additional_sparsity = amount_per_iteration * (1 - current_sparsity)
                    
                    mask = PruningUtilities.get_pruning_mask_magnitude(
                        module.weight.data,
                        additional_sparsity
                    )
                    
                    PruningUtilities.apply_pruning_mask(module, mask)
            
            # Fine-tune if requested
            if fine_tune_epochs > 0 and dataloader is not None:
                model.train()
                for epoch in range(fine_tune_epochs):
                    for inputs, targets in dataloader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                        loss.backward()
                        optimizer.step()
            
            # Evaluate
            if dataloader is not None:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in dataloader:
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                accuracy = correct / total
                accuracies.append(accuracy)
                
                logger.info(f"Iteration {iteration + 1}: Sparsity = {PruningUtilities.get_model_sparsity(model):.2%}, Accuracy = {accuracy:.2%}")
        
        return accuracies
    
    @staticmethod
    def get_model_sparsity(model: nn.Module) -> float:
        """
        Get overall sparsity of a model.
        
        Args:
            model: Model to check
            
        Returns:
            Overall sparsity (fraction of zero weights)
        """
        total_params = 0
        zero_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    @staticmethod
    def structured_pruning(
        layer: nn.Module,
        amount: float,
        dim: int = 0,
        importance_scores: Optional[torch.Tensor] = None
    ):
        """
        Perform structured pruning on a layer.
        
        Args:
            layer: Layer to prune
            amount: Fraction of structures to prune
            dim: Dimension to prune (0 for output channels, 1 for input channels)
            importance_scores: Optional pre-computed importance scores
        """
        if not hasattr(layer, 'weight'):
            return
        
        weights = layer.weight.data
        
        # Compute importance scores if not provided
        if importance_scores is None:
            if isinstance(layer, nn.Conv2d):
                # For conv layers, compute L2 norm of each filter
                importance_scores = weights.norm(dim=(1, 2, 3) if dim == 0 else (0, 2, 3))
            else:
                # For linear layers, compute L2 norm along appropriate dimension
                importance_scores = weights.norm(dim=1 if dim == 0 else 0)
        
        # Get indices to prune
        num_to_prune = int(amount * importance_scores.numel())
        _, indices_to_prune = importance_scores.sort()
        indices_to_prune = indices_to_prune[:num_to_prune]
        
        # Create mask
        mask = torch.ones_like(weights)
        if dim == 0:
            mask[indices_to_prune] = 0
        else:
            mask[:, indices_to_prune] = 0
        
        # Apply mask
        PruningUtilities.apply_pruning_mask(layer, mask)
    
    @staticmethod
    def sensitivity_based_pruning(
        model: nn.Module,
        layers_to_prune: List[str],
        amounts: Union[float, Dict[str, float]],
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable
    ):
        """
        Perform sensitivity-based pruning on specified layers.
        
        Args:
            model: Model containing layers to prune
            layers_to_prune: Names of layers to prune
            amounts: Amount to prune (single value or dict per layer)
            dataloader: Data for sensitivity analysis
            loss_fn: Loss function
        """
        if isinstance(amounts, float):
            amounts = {layer: amounts for layer in layers_to_prune}
        
        for name, module in model.named_modules():
            if name in layers_to_prune:
                amount = amounts[name]
                mask = PruningUtilities.get_pruning_mask_sensitivity(
                    model, module, dataloader, amount, loss_fn
                )
                PruningUtilities.apply_pruning_mask(module, mask)
                logger.info(f"Pruned {name}: sparsity = {PruningUtilities.get_sparsity(module):.2%}")


def create_pruning_schedule(
    initial_sparsity: float = 0.0,
    final_sparsity: float = 0.9,
    begin_step: int = 0,
    end_step: int = 1000,
    frequency: int = 100,
    schedule_type: str = "polynomial"
) -> Callable[[int], float]:
    """
    Create a pruning schedule function.
    
    Args:
        initial_sparsity: Starting sparsity
        final_sparsity: Target sparsity
        begin_step: Step to begin pruning
        end_step: Step to end pruning
        frequency: How often to update sparsity
        schedule_type: Type of schedule ("linear", "polynomial", "exponential")
        
    Returns:
        Function that maps step -> sparsity
    """
    def schedule(step: int) -> float:
        if step < begin_step:
            return initial_sparsity
        if step >= end_step:
            return final_sparsity
        
        # Only update at frequency intervals
        step = (step // frequency) * frequency
        progress = (step - begin_step) / (end_step - begin_step)
        
        if schedule_type == "linear":
            sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * progress
        elif schedule_type == "polynomial":
            sparsity = final_sparsity + (initial_sparsity - final_sparsity) * ((1 - progress) ** 3)
        elif schedule_type == "exponential":
            sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (0.5 ** (5 * progress))
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        return sparsity
    
    return schedule 