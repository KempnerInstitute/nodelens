"""
Movement-based pruning strategies.

Implements movement pruning which identifies parameters that consistently
move toward zero during training.

Reference: Sanh et al., "Movement Pruning: Adaptive Sparsity by Fine-Tuning" (2020)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

from ..base import BasePruningStrategy, PruningConfig
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


class MovementPruning(BasePruningStrategy):
    """
    Movement-based pruning strategy.
    
    Identifies weights that consistently move toward zero during training
    and prunes them. More sophisticated than simple magnitude pruning.
    
    Score = |weight| × sign(weight) × sign(gradient)
    
    Interpretation:
    - Positive score: Weight moving away from zero (important)
    - Negative score: Weight moving toward zero (candidate for pruning)
    
    Reference:
        Sanh et al., "Movement Pruning: Adaptive Sparsity by Fine-Tuning"
        NeurIPS 2020
    
    Example:
        >>> strategy = MovementPruning()
        >>> # During training, accumulate gradients
        >>> for batch in train_loader:
        >>>     loss.backward()
        >>>     # Before optimizer.step():
        >>>     strategy.update_movement_history(model)
        >>>     optimizer.step()
        >>> 
        >>> # After training:
        >>> mask = strategy.prune(model, amount=0.5)
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        use_momentum: bool = True,
        momentum: float = 0.9
    ):
        """
        Initialize movement pruning.
        
        Args:
            config: Pruning configuration
            use_momentum: Whether to use momentum for movement estimation
            momentum: Momentum coefficient for exponential moving average
        """
        super().__init__(config)
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.movement_history: Dict[str, torch.Tensor] = {}
    
    def update_movement_history(self, model: nn.Module):
        """
        Update movement history during training.
        
        Call this BEFORE optimizer.step() to track weight movement.
        
        Args:
            model: Model being trained
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                
                # Compute movement score: sign(weight) × sign(gradient)
                # Negative = moving toward zero
                movement = torch.sign(param.data) * torch.sign(param.grad.data)
                
                if name not in self.movement_history:
                    self.movement_history[name] = movement.clone()
                else:
                    if self.use_momentum:
                        # Exponential moving average
                        self.movement_history[name] = (
                            self.momentum * self.movement_history[name] +
                            (1 - self.momentum) * movement
                        )
                    else:
                        # Simple accumulation
                        self.movement_history[name] += movement
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        module_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute movement-based importance scores.
        
        Args:
            module: Module to score
            inputs: Not used for movement pruning
            module_name: Name of module (for retrieving history)
            **kwargs: Additional arguments
            
        Returns:
            Importance scores (same shape as module.weight)
        """
        weight = module.weight
        
        # Get movement history if available
        if module_name and module_name + '.weight' in self.movement_history:
            movement = self.movement_history[module_name + '.weight']
        else:
            logger.warning(
                f"No movement history for {module_name}, falling back to magnitude. "
                "Use update_movement_history() during training."
            )
            # Fallback to magnitude
            return weight.abs()
        
        # Movement score: magnitude × movement direction
        # High positive = moving away from zero = important
        # Negative = moving toward zero = prune
        scores = weight.abs() * movement
        
        return scores
    
    def reset_history(self):
        """Reset movement history."""
        self.movement_history.clear()
    
    def get_movement_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about weight movement.
        
        Returns:
            Dict with statistics per parameter
        """
        stats = {}
        
        for name, movement in self.movement_history.items():
            moving_away = (movement > 0).sum().item()
            moving_toward = (movement < 0).sum().item()
            total = movement.numel()
            
            stats[name] = {
                'moving_away_zero': moving_away,
                'moving_toward_zero': moving_toward,
                'total_params': total,
                'fraction_toward_zero': moving_toward / total,
                'mean_movement': movement.mean().item(),
                'std_movement': movement.std().item()
            }
        
        return stats


class AdaptiveMovementPruning(MovementPruning):
    """
    Adaptive movement pruning that adjusts pruning amount based on movement.
    
    Automatically determines pruning amount based on how many weights
    are consistently moving toward zero.
    """
    
    def __init__(
        self,
        base_amount: float = 0.5,
        adaptation_strength: float = 0.3,
        **kwargs
    ):
        """
        Initialize adaptive movement pruning.
        
        Args:
            base_amount: Base pruning amount
            adaptation_strength: How much to adapt based on movement (0-1)
        """
        super().__init__(**kwargs)
        self.base_amount = base_amount
        self.adaptation_strength = adaptation_strength
    
    def compute_adaptive_amount(
        self,
        module: nn.Module,
        module_name: str
    ) -> float:
        """
        Compute adaptive pruning amount for this module.
        
        Args:
            module: Module to prune
            module_name: Name of module
            
        Returns:
            Adjusted pruning amount
        """
        if module_name + '.weight' not in self.movement_history:
            return self.base_amount
        
        movement = self.movement_history[module_name + '.weight']
        
        # Fraction moving toward zero
        toward_zero = (movement < 0).float().mean().item()
        
        # Adapt pruning amount
        # More weights moving toward zero → prune more aggressively
        adapted_amount = self.base_amount + self.adaptation_strength * (toward_zero - 0.5)
        
        # Clip to valid range
        adapted_amount = max(0.1, min(0.9, adapted_amount))
        
        logger.info(
            f"{module_name}: {toward_zero:.1%} moving toward zero, "
            f"adapted amount: {adapted_amount:.1%}"
        )
        
        return adapted_amount
    
    def prune(
        self,
        module: nn.Module,
        amount: Optional[float] = None,
        module_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Prune with adaptive amount.
        
        If amount is None, uses adaptive computation.
        """
        if amount is None and module_name:
            amount = self.compute_adaptive_amount(module, module_name)
        elif amount is None:
            amount = self.base_amount
        
        # Use parent's prune method with adapted amount
        return super().prune(module, amount=amount, module_name=module_name, **kwargs)

