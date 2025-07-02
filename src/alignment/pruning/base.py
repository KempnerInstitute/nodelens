"""
Base classes for pruning strategies.

This module defines the abstract base classes and interfaces for all pruning strategies
in the alignment framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Tuple, Any
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""
    amount: float = 0.5  # Fraction of parameters to prune
    structured: bool = False  # Whether to prune entire channels/filters
    iterative: bool = False  # Whether to prune iteratively
    global_pruning: bool = False  # Whether to prune globally across layers
    iterations: int = 1  # Number of pruning iterations
    fine_tune_epochs: int = 0  # Epochs of fine-tuning between iterations
    pruning_mode: str = 'low'  # 'low' to prune low values, 'high' to prune high values


class BasePruningStrategy(ABC):
    """
    Abstract base class for all pruning strategies.
    
    This class defines the interface that all pruning strategies must implement.
    Subclasses should implement the compute_importance_scores method to define
    how importance is calculated for pruning decisions.
    """
    
    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize the pruning strategy.
        
        Args:
            config: Pruning configuration. If None, uses default config.
        """
        self.config = config or PruningConfig()
    
    @abstractmethod
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores for the given module.
        
        Args:
            module: The module to compute importance scores for
            inputs: Optional input tensor for the module
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            Tensor of importance scores with same shape as module weights
        """
        pass
    
    def create_pruning_mask(
        self,
        importance_scores: torch.Tensor,
        amount: Optional[float] = None,
        structured: Optional[bool] = None,
        dim: Optional[int] = None,
        pruning_mode: Optional[str] = None
    ) -> torch.Tensor:
        """
        Create a pruning mask based on importance scores.
        
        Args:
            importance_scores: Tensor of importance scores
            amount: Fraction to prune (overrides config if provided)
            structured: Whether to do structured pruning (overrides config)
            dim: Dimension for structured pruning
            pruning_mode: 'low' to prune low values, 'high' to prune high values
            
        Returns:
            Binary mask tensor (1 = keep, 0 = prune)
        """
        amount = amount if amount is not None else self.config.amount
        structured = structured if structured is not None else self.config.structured
        pruning_mode = pruning_mode if pruning_mode is not None else self.config.pruning_mode
        
        if structured and dim is not None:
            # Aggregate importance scores along non-pruned dimensions
            dims_to_reduce = list(range(importance_scores.ndim))
            dims_to_reduce.pop(dim)
            
            if dims_to_reduce:
                aggregated_scores = importance_scores.abs().sum(dim=dims_to_reduce)
            else:
                aggregated_scores = importance_scores.abs()
            
            # Get threshold
            k = int(amount * aggregated_scores.numel())
            if k == 0:
                return torch.ones_like(importance_scores)
            
            if pruning_mode == 'low':
                threshold = aggregated_scores.flatten().kthvalue(k).values
                mask = aggregated_scores > threshold
            else:  # pruning_mode == 'high'
                threshold = aggregated_scores.flatten().kthvalue(aggregated_scores.numel() - k).values
                mask = aggregated_scores < threshold
            
            # Expand mask to original shape
            shape = [1] * importance_scores.ndim
            shape[dim] = importance_scores.shape[dim]
            mask = mask.reshape(shape).expand_as(importance_scores)
        else:
            # Unstructured pruning
            importance_flat = importance_scores.flatten()
            k = int(amount * importance_flat.numel())
            
            if k == 0:
                return torch.ones_like(importance_scores)
            
            if pruning_mode == 'low':
                threshold = importance_flat.kthvalue(k).values
                mask = importance_scores > threshold
            else:  # pruning_mode == 'high'
                threshold = importance_flat.kthvalue(importance_flat.numel() - k).values
                mask = importance_scores < threshold
        
        return mask.float()
    
    def apply_pruning(
        self,
        module: nn.Module,
        mask: torch.Tensor,
        make_permanent: bool = False
    ):
        """
        Apply pruning mask to a module.
        
        Args:
            module: Module to prune
            mask: Binary mask to apply
            make_permanent: If True, makes pruning permanent (cannot be reverted)
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have a weight parameter")
        
        if not make_permanent:
            # Store original weights BEFORE applying mask
            if not hasattr(module, '_original_weight'):
                module.register_buffer('_original_weight', module.weight.data.clone())
            
            # Register mask as buffer for forward passes
            module.register_buffer('weight_mask', mask)
            
            # Apply mask to weights
            module.weight.data *= mask
            
            # Hook to apply mask during forward pass
            def apply_mask_hook(mod, inputs):
                # Apply mask to original weights to maintain pruning
                mod.weight.data = mod._original_weight * mod.weight_mask
                return inputs
            
            # Hook to mask gradients during backward pass
            def mask_gradient_hook(grad):
                # Mask gradients to prevent updates to pruned weights
                return grad * module.weight_mask
            
            # Remove old hooks if exist
            if hasattr(module, '_pruning_hook'):
                module._pruning_hook.remove()
            if hasattr(module, '_gradient_hook_handle'):
                module._gradient_hook_handle.remove()
            
            # Register hooks
            module._pruning_hook = module.register_forward_pre_hook(apply_mask_hook)
            module._gradient_hook_handle = module.weight.register_hook(mask_gradient_hook)
        else:
            # Apply mask permanently
            module.weight.data *= mask
    
    def remove_pruning(self, module: nn.Module):
        """
        Remove pruning from a module (make pruning permanent).
        
        Args:
            module: Module to remove pruning from
        """
        if hasattr(module, 'weight_mask'):
            # Apply mask permanently
            if hasattr(module, '_original_weight'):
                module.weight.data = module._original_weight * module.weight_mask
                delattr(module, '_original_weight')
            else:
                module.weight.data *= module.weight_mask
            # Remove mask buffer
            delattr(module, 'weight_mask')
        
        # Remove hooks
        if hasattr(module, '_pruning_hook'):
            module._pruning_hook.remove()
            delattr(module, '_pruning_hook')
        
        if hasattr(module, '_gradient_hook_handle'):
            module._gradient_hook_handle.remove()
            delattr(module, '_gradient_hook_handle')
    
    def get_sparsity(self, module: nn.Module) -> float:
        """
        Get the current sparsity level of a module.
        
        Args:
            module: Module to check
            
        Returns:
            Fraction of weights that are zero
        """
        if hasattr(module, 'weight'):
            total = module.weight.numel()
            zeros = (module.weight == 0).sum().item()
            return zeros / total
        return 0.0
    
    def prune(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        amount: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Main pruning method that combines all steps.
        
        Args:
            module: Module to prune
            inputs: Optional inputs for computing importance
            amount: Optional amount to prune (overrides config)
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            The pruning mask that was applied
        """
        # Compute importance scores
        importance_scores = self.compute_importance_scores(module, inputs, **kwargs)
        
        # Create mask
        mask = self.create_pruning_mask(importance_scores, amount)
        
        # Apply mask
        self.apply_pruning(module, mask)
        
        return mask


class IterativePruningStrategy(BasePruningStrategy):
    """
    Base class for iterative pruning strategies.
    
    This class extends BasePruningStrategy to support iterative pruning
    with fine-tuning between iterations.
    """
    
    def iterative_prune(
        self,
        model: nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        fine_tune_fn: Optional[callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform iterative pruning on a model.
        
        Args:
            model: Model to prune
            dataloader: DataLoader for fine-tuning
            optimizer: Optimizer for fine-tuning
            criterion: Loss criterion for fine-tuning
            fine_tune_fn: Custom fine-tuning function
            **kwargs: Additional arguments for pruning
            
        Returns:
            Dictionary containing pruning results and statistics
        """
        results = {
            'sparsity_per_iteration': [],
            'loss_per_iteration': [],
            'accuracy_per_iteration': []
        }
        
        # Calculate amount to prune per iteration
        total_amount = self.config.amount
        iterations = self.config.iterations
        amount_per_iter = 1 - (1 - total_amount) ** (1 / iterations)
        
        for iteration in range(iterations):
            # Prune each prunable module
            for name, module in model.named_modules():
                if self._is_prunable(module):
                    current_sparsity = self.get_sparsity(module)
                    additional_amount = amount_per_iter * (1 - current_sparsity)
                    
                    if additional_amount > 0:
                        self.prune(module, amount=additional_amount, **kwargs)
            
            # Calculate current model sparsity
            total_params = 0
            zero_params = 0
            for module in model.modules():
                if hasattr(module, 'weight'):
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()
            
            current_sparsity = zero_params / total_params if total_params > 0 else 0
            results['sparsity_per_iteration'].append(current_sparsity)
            
            # Fine-tune if requested
            if self.config.fine_tune_epochs > 0 and dataloader is not None:
                if fine_tune_fn is not None:
                    # Use custom fine-tuning function
                    metrics = fine_tune_fn(
                        model, dataloader, optimizer, criterion,
                        epochs=self.config.fine_tune_epochs
                    )
                    if 'loss' in metrics:
                        results['loss_per_iteration'].append(metrics['loss'])
                    if 'accuracy' in metrics:
                        results['accuracy_per_iteration'].append(metrics['accuracy'])
                else:
                    # Simple fine-tuning loop
                    model.train()
                    for epoch in range(self.config.fine_tune_epochs):
                        for inputs, targets in dataloader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
        
        return results
    
    def _is_prunable(self, module: nn.Module) -> bool:
        """Check if a module can be pruned."""
        return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) 