"""
Gradient-based pruning strategy.

This module implements pruning based on gradient information, removing weights
that have small gradients or small gradient-weight products.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn

from ..base import BasePruningStrategy


class GradientPruning(BasePruningStrategy):
    """
    Gradient-based pruning strategy.

    This strategy prunes weights based on gradient information. It can use
    either gradient magnitudes alone or the product of gradients and weights
    (Taylor approximation of loss change).

    Examples:
        >>> from alignment.pruning.strategies import GradientPruning
        >>>
        >>> # Using gradient magnitude
        >>> strategy = GradientPruning(mode='gradient')
        >>>
        >>> # Forward and backward pass to get gradients
        >>> loss = criterion(model(inputs), targets)
        >>> loss.backward()
        >>>
        >>> # Prune based on gradients
        >>> mask = strategy.prune(layer, amount=0.5)
        >>>
        >>> # Using gradient-weight product (Taylor approximation)
        >>> strategy = GradientPruning(mode='taylor')
        >>> mask = strategy.prune(layer, amount=0.5)
    """

    def __init__(
        self,
        config=None,
        mode: Literal['gradient', 'taylor'] = 'taylor'
    ):
        """
        Initialize gradient pruning strategy.

        Args:
            config: Pruning configuration
            mode: Pruning mode
                - 'gradient': Use gradient magnitude
                - 'taylor': Use gradient-weight product (default)
        """
        super().__init__(config)
        self.mode = mode

    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores based on gradients.

        Args:
            module: Module to compute scores for
            inputs: Not used
            **kwargs: Not used

        Returns:
            Tensor of importance scores

        Raises:
            ValueError: If module has no gradients
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")

        if module.weight.grad is None:
            raise ValueError(
                f"Module {module} has no gradients. "
                "Run backward pass before pruning. "
                "Note: Gradient-based pruning is not suitable for post-training pruning "
                "on converged models as gradients will be near-zero."
            )

        if self.mode == 'gradient':
            # Use gradient magnitude
            importance = module.weight.grad.abs()
        elif self.mode == 'taylor':
            # Use Taylor approximation (gradient * weight)
            importance = (module.weight.grad * module.weight.data).abs()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return importance


class FisherPruning(BasePruningStrategy):
    """
    Fisher information-based pruning strategy.

    This strategy approximates the Fisher information matrix diagonal
    using gradient squares accumulated over multiple batches.

    Examples:
        >>> from alignment.pruning.strategies import FisherPruning
        >>> strategy = FisherPruning()
        >>>
        >>> # Accumulate Fisher information over multiple batches
        >>> for inputs, targets in dataloader:
        >>>     loss = criterion(model(inputs), targets)
        >>>     loss.backward()
        >>>     strategy.accumulate_fisher(model)
        >>>     optimizer.zero_grad()
        >>>
        >>> # Prune based on accumulated Fisher information
        >>> masks = strategy.prune_model(model, amount=0.5)
    """

    def __init__(self, config=None):
        """Initialize Fisher pruning strategy."""
        super().__init__(config)
        self.fisher_info = {}
        self.n_samples = 0

    def accumulate_fisher(self, model: nn.Module):
        """
        Accumulate Fisher information from current gradients.

        Args:
            model: Model to accumulate Fisher information for
        """
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                if name not in self.fisher_info:
                    self.fisher_info[name] = torch.zeros_like(module.weight.data)

                # Accumulate squared gradients
                self.fisher_info[name] += module.weight.grad.data ** 2

        self.n_samples += 1

    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        module_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores based on Fisher information.

        Args:
            module: Module to compute scores for
            inputs: Not used
            module_name: Name of the module in fisher_info dict
            **kwargs: Not used

        Returns:
            Tensor of importance scores
        """
        if module_name is None or module_name not in self.fisher_info:
            # Fall back to gradient magnitude if no Fisher info
            if module.weight.grad is not None:
                return module.weight.grad.abs()
            else:
                return module.weight.data.abs()

        # Use accumulated Fisher information
        fisher = self.fisher_info[module_name] / self.n_samples

        # Importance is Fisher info times weight squared
        importance = fisher * (module.weight.data ** 2)

        return importance

    def prune_model(
        self,
        model: nn.Module,
        amount: Optional[float] = None
    ) -> dict:
        """
        Prune model based on accumulated Fisher information.

        Args:
            model: Model to prune
            amount: Fraction to prune

        Returns:
            Dictionary mapping module names to pruning masks
        """
        masks = {}

        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                importance = self.compute_importance_scores(
                    module, module_name=name
                )
                mask = self.create_pruning_mask(importance, amount)
                self.apply_pruning(module, mask)
                masks[name] = mask

        return masks

    def reset_fisher(self):
        """Reset accumulated Fisher information."""
        self.fisher_info.clear()
        self.n_samples = 0


class MomentumPruning(BasePruningStrategy):
    """
    Momentum-based pruning strategy.

    This strategy maintains a momentum buffer of importance scores,
    providing more stable pruning decisions over time.

    Examples:
        >>> from alignment.pruning.strategies import MomentumPruning
        >>> strategy = MomentumPruning(momentum=0.9)
        >>>
        >>> # Update importance with momentum over multiple iterations
        >>> for epoch in range(epochs):
        >>>     for inputs, targets in dataloader:
        >>>         loss = criterion(model(inputs), targets)
        >>>         loss.backward()
        >>>
        >>>         # Update momentum buffer
        >>>         strategy.update_momentum(model)
        >>>         optimizer.step()
        >>>         optimizer.zero_grad()
        >>>
        >>> # Prune based on momentum buffer
        >>> masks = strategy.prune_model(model, amount=0.5)
    """

    def __init__(self, config=None, momentum: float = 0.9):
        """
        Initialize momentum pruning strategy.

        Args:
            config: Pruning configuration
            momentum: Momentum factor (0-1)
        """
        super().__init__(config)
        self.momentum = momentum
        self.importance_buffer = {}

    def update_momentum(self, model: nn.Module):
        """
        Update momentum buffer with current gradients.

        Args:
            model: Model to update momentum for
        """
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                # Current importance (gradient-weight product)
                current_importance = (
                    module.weight.grad.data * module.weight.data
                ).abs()

                if name not in self.importance_buffer:
                    # Initialize buffer
                    self.importance_buffer[name] = current_importance
                else:
                    # Update with momentum
                    self.importance_buffer[name] = (
                        self.momentum * self.importance_buffer[name] +
                        (1 - self.momentum) * current_importance
                    )

    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        module_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Get importance scores from momentum buffer.

        Args:
            module: Module to get scores for
            inputs: Not used
            module_name: Name of module in buffer
            **kwargs: Not used

        Returns:
            Tensor of importance scores
        """
        if module_name is None or module_name not in self.importance_buffer:
            # Fall back to current gradient if no buffer
            if module.weight.grad is not None:
                return (module.weight.grad * module.weight.data).abs()
            else:
                return module.weight.data.abs()

        return self.importance_buffer[module_name]

    def prune_model(
        self,
        model: nn.Module,
        amount: Optional[float] = None
    ) -> dict:
        """
        Prune model based on momentum buffer.

        Args:
            model: Model to prune
            amount: Fraction to prune

        Returns:
            Dictionary mapping module names to pruning masks
        """
        masks = {}

        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                importance = self.compute_importance_scores(
                    module, module_name=name
                )
                mask = self.create_pruning_mask(importance, amount)
                self.apply_pruning(module, mask)
                masks[name] = mask

        return masks

    def reset_momentum(self):
        """Reset momentum buffer."""
        self.importance_buffer.clear()
