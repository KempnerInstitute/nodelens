"""
Base classes for pruning strategies.

This module defines the abstract base classes and interfaces for all pruning strategies
in the alignment framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""

    amount: float = 0.5  # Fraction of parameters to prune
    structured: bool = False  # Whether to prune entire channels/filters
    iterative: bool = False  # Whether to prune iteratively
    global_pruning: bool = False  # Whether to prune globally across layers
    iterations: int = 1  # Number of pruning iterations
    fine_tune_epochs: int = 0  # Epochs of fine-tuning between iterations
    pruning_mode: str = "low"  # 'low' to prune low values, 'high' to prune high values


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
    def compute_importance_scores(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
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
        pruning_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Create a pruning mask based on importance scores.

        Args:
            importance_scores: Tensor of importance scores
            amount: Fraction to prune (overrides config if provided)
            structured: Whether to do structured pruning (overrides config)
            dim: Dimension for structured pruning (default: 0 for output dimension)
            pruning_mode: 'low' to prune low values, 'high' to prune high values, 'random' for random

        Returns:
            Binary mask tensor (1 = keep, 0 = prune)
        """
        amount = amount if amount is not None else self.config.amount
        structured = structured if structured is not None else self.config.structured
        pruning_mode = pruning_mode if pruning_mode is not None else self.config.pruning_mode

        if structured:
            # Default to dimension 0 (output dimension) for structured pruning
            if dim is None:
                dim = 0

            # Aggregate importance scores along non-pruned dimensions
            dims_to_reduce = list(range(importance_scores.ndim))
            dims_to_reduce.pop(dim)

            if dims_to_reduce:
                aggregated_scores = importance_scores.abs().sum(dim=dims_to_reduce)
            else:
                aggregated_scores = importance_scores.abs()

            # Get number of structures to prune
            k = int(amount * aggregated_scores.numel())
            if k == 0:
                return torch.ones_like(importance_scores)

            # Use topk-based selection to guarantee exactly k neurons are pruned
            # This avoids issues with ties at threshold values that can cause non-monotonic behavior
            flat_scores = aggregated_scores.flatten()
            mask = torch.ones(flat_scores.numel(), dtype=torch.bool, device=aggregated_scores.device)

            if pruning_mode == "random":
                # Random selection of structures
                indices = torch.randperm(flat_scores.numel(), device=flat_scores.device)[:k]
            elif pruning_mode == "low":
                # Prune k neurons with LOWEST scores (keep high-scoring neurons)
                _, indices = torch.topk(flat_scores, k, largest=False)
            else:  # pruning_mode == 'high'
                # Prune k neurons with HIGHEST scores (keep low-scoring neurons)
                _, indices = torch.topk(flat_scores, k, largest=True)

            mask[indices] = False

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

            # Use topk-based selection to guarantee exactly k weights are pruned
            # This avoids issues with ties at threshold values
            mask = torch.ones(importance_flat.numel(), dtype=torch.bool, device=importance_scores.device)

            if pruning_mode == "random":
                # Random selection of weights
                indices = torch.randperm(importance_flat.numel(), device=importance_flat.device)[:k]
            elif pruning_mode == "low":
                # Prune k weights with LOWEST scores (keep high-scoring weights)
                _, indices = torch.topk(importance_flat, k, largest=False)
            else:  # pruning_mode == 'high'
                # Prune k weights with HIGHEST scores (keep low-scoring weights)
                _, indices = torch.topk(importance_flat, k, largest=True)

            mask[indices] = False
            mask = mask.view(importance_scores.shape)

        return mask.float()

    def apply_pruning(self, module: nn.Module, mask: torch.Tensor, make_permanent: bool = False, dim: str = "auto"):
        """
        Apply pruning mask to a module's weights.

        Args:
            module: Module to prune (must have .weight)
            mask: Binary mask to apply (1D or same shape as weights)
            make_permanent: If True, apply pruning permanently (no hooks)
            dim: 'input', 'output', or 'auto' (default).
                - 'output': prunes rows (out_features)
                - 'input': prunes columns (in_features)
                - 'auto': infers based on mask size and weight shape
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have a weight parameter")

        weight = module.weight.data

        # Move mask to same device as weight (important for multi-GPU models)
        mask = mask.to(weight.device)

        # Handle dimension
        if mask.dim() == 1:
            if dim == "auto":
                if mask.numel() == weight.size(0):
                    dim = "output"
                elif mask.numel() == weight.size(1):
                    dim = "input"
                else:
                    raise ValueError(
                        f"Mask length {mask.numel()} doesn't match weight shape {tuple(weight.shape)} " f"for module {module.__class__.__name__}"
                    )
            if dim == "output":
                # Linear: [out, in] -> mask[:, None]
                # ConvNd: [out, in, k...,] -> mask.view(out, 1, 1, 1, ...)
                if weight.dim() == 2:
                    mask_applied = mask[:, None]
                else:
                    mask_applied = mask.view(-1, *([1] * (weight.dim() - 1)))
            elif dim == "input":
                # Linear: [out, in] -> mask[None, :]
                # ConvNd: [out, in, k...,] -> mask.view(1, in, 1, 1, ...)
                if weight.dim() == 2:
                    mask_applied = mask[None, :]
                else:
                    mask_applied = mask.view(1, -1, *([1] * (weight.dim() - 2)))
            else:
                raise ValueError(f"Invalid dim='{dim}', must be 'input', 'output', or 'auto'")
        elif mask.shape == weight.shape:
            mask_applied = mask
        else:
            raise ValueError(f"Mask shape {tuple(mask.shape)} not compatible with weight shape {tuple(weight.shape)}")

        # Apply mask permanently or with hooks
        if make_permanent:
            weight *= mask_applied
            return

        # Store original weights if not already done
        if not hasattr(module, "_original_weight"):
            module.register_buffer("_original_weight", weight.clone())

        module.register_buffer("weight_mask", mask_applied)

        # Apply mask to weights immediately
        weight *= mask_applied

        # Define hooks
        def apply_mask_hook(mod, inputs):
            mod.weight.data = mod._original_weight * mod.weight_mask
            return inputs

        def mask_gradient_hook(grad):
            return grad * module.weight_mask

        # Clean up old hooks if they exist
        if hasattr(module, "_pruning_hook"):
            module._pruning_hook.remove()
        if hasattr(module, "_gradient_hook_handle"):
            module._gradient_hook_handle.remove()

        # Register hooks
        module._pruning_hook = module.register_forward_pre_hook(apply_mask_hook)
        module._gradient_hook_handle = module.weight.register_hook(mask_gradient_hook)

    def remove_pruning(self, module: nn.Module):
        """
        Remove pruning from a module (make pruning permanent).

        Args:
            module: Module to remove pruning from
        """
        if hasattr(module, "weight_mask"):
            # Apply mask permanently
            if hasattr(module, "_original_weight"):
                module.weight.data = module._original_weight * module.weight_mask
                delattr(module, "_original_weight")
            else:
                module.weight.data *= module.weight_mask
            # Remove mask buffer
            delattr(module, "weight_mask")

        # Remove hooks
        if hasattr(module, "_pruning_hook"):
            module._pruning_hook.remove()
            delattr(module, "_pruning_hook")

        if hasattr(module, "_gradient_hook_handle"):
            module._gradient_hook_handle.remove()
            delattr(module, "_gradient_hook_handle")

    def clear_pruning_state(self, module: nn.Module):
        """
        Clear all pruning state from a module WITHOUT making pruning permanent.
        This allows a fresh pruning to be applied later.

        Args:
            module: Module to clear pruning state from
        """
        # Remove hooks first
        if hasattr(module, "_pruning_hook"):
            module._pruning_hook.remove()
            delattr(module, "_pruning_hook")

        if hasattr(module, "_gradient_hook_handle"):
            module._gradient_hook_handle.remove()
            delattr(module, "_gradient_hook_handle")

        # Remove buffers (original weight and mask)
        if hasattr(module, "_original_weight"):
            delattr(module, "_original_weight")

        if hasattr(module, "weight_mask"):
            delattr(module, "weight_mask")

    def get_sparsity(self, module: nn.Module) -> float:
        """
        Get the current sparsity level of a module.

        Args:
            module: Module to check

        Returns:
            Fraction of weights that are zero
        """
        if hasattr(module, "weight"):
            total = module.weight.numel()
            zeros = (module.weight == 0).sum().item()
            return zeros / total
        return 0.0

    def prune(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, amount: Optional[float] = None, **kwargs) -> torch.Tensor:
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


class PrecomputedScorePruning(BasePruningStrategy):
    """
    Pruning strategy for use with pre-computed importance scores.

    This is a concrete implementation that doesn't compute scores itself,
    but can apply masks and pruning using externally computed importance scores
    (e.g., from SCAR metrics or other analysis methods).
    """

    def compute_importance_scores(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Not used for pre-computed scores - raises error if called.

        For PrecomputedScorePruning, importance scores should be passed directly
        to create_pruning_mask() rather than computed here.
        """
        raise NotImplementedError(
            "PrecomputedScorePruning expects scores to be passed directly to create_pruning_mask(). "
            "Do not call compute_importance_scores() on this class."
        )


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
        **kwargs,
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
        results = {"sparsity_per_iteration": [], "loss_per_iteration": [], "accuracy_per_iteration": []}

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
                if hasattr(module, "weight"):
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()

            current_sparsity = zero_params / total_params if total_params > 0 else 0
            results["sparsity_per_iteration"].append(current_sparsity)

            # Fine-tune if requested
            if self.config.fine_tune_epochs > 0 and dataloader is not None:
                if fine_tune_fn is not None:
                    # Use custom fine-tuning function
                    metrics = fine_tune_fn(model, dataloader, optimizer, criterion, epochs=self.config.fine_tune_epochs)
                    if "loss" in metrics:
                        results["loss_per_iteration"].append(metrics["loss"])
                    if "accuracy" in metrics:
                        results["accuracy_per_iteration"].append(metrics["accuracy"])
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
