"""
Magnitude-based pruning strategy.

This module implements pruning based on weight magnitudes, removing weights
with the smallest absolute values.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..base import BasePruningStrategy, IterativePruningStrategy


class MagnitudePruning(BasePruningStrategy):
    """
    Magnitude-based pruning strategy.
    
    This strategy prunes weights with the smallest absolute values, based on
    the assumption that small weights contribute less to the network's output.
    
    Examples:
        >>> from alignment.pruning.strategies import MagnitudePruning
        >>> from alignment.pruning import PruningConfig
        >>>
        >>> # Basic usage
        >>> strategy = MagnitudePruning()
        >>> mask = strategy.prune(layer, amount=0.5)
        >>>
        >>> # With configuration
        >>> config = PruningConfig(amount=0.7, structured=True)
        >>> strategy = MagnitudePruning(config)
        >>> mask = strategy.prune(conv_layer)
    """
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores based on weight magnitudes.
        
        Args:
            module: Module to compute scores for
            inputs: Not used for magnitude pruning
            **kwargs: Not used
            
        Returns:
            Tensor of importance scores (absolute weight values)
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        
        # Importance is simply the absolute value of weights
        return module.weight.data.abs()


class IterativeMagnitudePruning(IterativePruningStrategy):
    """
    Iterative magnitude-based pruning strategy.
    
    This strategy gradually prunes weights over multiple iterations,
    allowing the network to adapt between pruning steps.
    
    Examples:
        >>> from alignment.pruning.strategies import IterativeMagnitudePruning
        >>> from alignment.pruning import PruningConfig
        >>>
        >>> config = PruningConfig(
        ...     amount=0.9,  # Final sparsity
        ...     iterations=10,
        ...     fine_tune_epochs=5
        ... )
        >>> strategy = IterativeMagnitudePruning(config)
        >>> 
        >>> results = strategy.iterative_prune(
        ...     model=model,
        ...     dataloader=train_loader,
        ...     optimizer=optimizer,
        ...     criterion=nn.CrossEntropyLoss()
        ... )
    """
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores based on weight magnitudes.
        
        Args:
            module: Module to compute scores for
            inputs: Not used for magnitude pruning
            **kwargs: Not used
            
        Returns:
            Tensor of importance scores (absolute weight values)
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        
        return module.weight.data.abs()


class GlobalMagnitudePruning(BasePruningStrategy):
    """
    Global magnitude-based pruning strategy.
    
    This strategy prunes weights globally across all layers based on their
    magnitudes, ensuring that the specified sparsity is achieved across
    the entire network rather than per-layer.
    
    Examples:
        >>> from alignment.pruning.strategies import GlobalMagnitudePruning
        >>> strategy = GlobalMagnitudePruning()
        >>> 
        >>> # Prune entire model to 70% sparsity
        >>> masks = strategy.prune_model(model, amount=0.7)
    """
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores for a single module.
        
        Note: For global pruning, use prune_model() instead of prune().
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        
        return module.weight.data.abs()
    
    def prune_model(
        self,
        model: nn.Module,
        amount: Optional[float] = None
    ) -> dict:
        """
        Prune entire model globally based on weight magnitudes.
        
        Args:
            model: Model to prune
            amount: Global sparsity level to achieve
            
        Returns:
            Dictionary mapping module names to their pruning masks
        """
        amount = amount if amount is not None else self.config.amount
        
        # Collect all weights and their importance scores
        all_scores = []
        module_info = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                scores = module.weight.data.abs()
                all_scores.append(scores.flatten())
                module_info.append((name, module, scores.shape))
        
        if not all_scores:
            return {}
        
        # Concatenate all scores
        global_scores = torch.cat(all_scores)
        
        # Find global threshold
        k = int(amount * global_scores.numel())
        if k == 0:
            return {}
        
        threshold = global_scores.kthvalue(k).values
        
        # Apply threshold to each module
        masks = {}
        for name, module, shape in module_info:
            scores = module.weight.data.abs()
            mask = (scores > threshold).float()
            self.apply_pruning(module, mask)
            masks[name] = mask
        
        return masks 