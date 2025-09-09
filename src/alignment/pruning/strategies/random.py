"""
Random pruning strategy.

This module implements random pruning, which removes weights randomly
regardless of their values. Used as a baseline for comparison.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..base import BasePruningStrategy


class RandomPruning(BasePruningStrategy):
    """
    Random pruning strategy.
    
    This strategy randomly prunes weights regardless of their values.
    It's primarily used as a baseline to compare against informed
    pruning strategies.
    
    Examples:
        >>> from alignment.pruning.strategies import RandomPruning
        >>> from alignment.pruning import PruningConfig
        >>>
        >>> # Basic random pruning
        >>> strategy = RandomPruning()
        >>> mask = strategy.prune(layer, amount=0.5)
        >>>
        >>> # Structured random pruning
        >>> config = PruningConfig(amount=0.3, structured=True)
        >>> strategy = RandomPruning(config)
        >>> mask = strategy.prune(conv_layer)
        >>>
        >>> # With specific random seed
        >>> strategy = RandomPruning(seed=42)
        >>> mask = strategy.prune(layer, amount=0.5)
    """
    
    def __init__(self, config=None, seed: Optional[int] = None):
        """
        Initialize random pruning strategy.
        
        Args:
            config: Pruning configuration
            seed: Random seed for reproducibility
        """
        super().__init__(config)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute random importance scores.
        
        For structured pruning, generates neuron-level scores.
        For unstructured pruning, generates weight-level scores.
        
        Args:
            module: Module to compute scores for
            inputs: Not used
            **kwargs: Not used
            
        Returns:
            Tensor of random importance scores
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        
        weights = module.weight.data
        
        # For structured pruning, generate neuron-level scores
        if self.config.structured:
            if len(weights.shape) == 2:  # Linear layer
                # Generate random score for each output neuron
                neuron_scores = torch.rand(weights.shape[0], device=weights.device)
                # Expand to weight dimensions (same score for all weights in a neuron)
                importance_scores = neuron_scores.unsqueeze(1).expand_as(weights)
            elif len(weights.shape) >= 3:  # Conv layer
                # Generate random score for each output channel
                channel_scores = torch.rand(weights.shape[0], device=weights.device)
                # Expand to weight dimensions
                shape = [weights.shape[0]] + [1] * (len(weights.shape) - 1)
                importance_scores = channel_scores.view(shape).expand_as(weights)
            else:
                # Fallback to weight-level for unknown shapes
                importance_scores = torch.rand_like(weights)
        else:
            # For unstructured pruning, generate weight-level random scores
            importance_scores = torch.rand_like(weights)
        
        return importance_scores


class LayerwiseRandomPruning(RandomPruning):
    """
    Layerwise random pruning with different sparsity per layer.
    
    This strategy allows specifying different pruning amounts for
    different layers while still using random selection.
    
    Examples:
        >>> from alignment.pruning.strategies import LayerwiseRandomPruning
        >>>
        >>> # Define per-layer sparsity
        >>> layer_sparsity = {
        ...     'features.0': 0.3,  # 30% pruning
        ...     'features.3': 0.5,  # 50% pruning
        ...     'classifier': 0.7   # 70% pruning
        ... }
        >>>
        >>> strategy = LayerwiseRandomPruning(layer_sparsity=layer_sparsity)
        >>> masks = strategy.prune_model(model)
    """
    
    def __init__(
        self,
        config=None,
        layer_sparsity: Optional[dict] = None,
        default_sparsity: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize layerwise random pruning.
        
        Args:
            config: Pruning configuration
            layer_sparsity: Dict mapping layer names to sparsity levels
            default_sparsity: Default sparsity for unnamed layers
            seed: Random seed
        """
        super().__init__(config, seed)
        self.layer_sparsity = layer_sparsity or {}
        self.default_sparsity = default_sparsity
    
    def prune_model(self, model: nn.Module) -> dict:
        """
        Prune model with per-layer random sparsity.
        
        Args:
            model: Model to prune
            
        Returns:
            Dictionary mapping module names to pruning masks
        """
        masks = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                # Get sparsity for this layer
                if name in self.layer_sparsity:
                    amount = self.layer_sparsity[name]
                else:
                    amount = self.default_sparsity
                
                # Generate random mask
                importance = self.compute_importance_scores(module)
                mask = self.create_pruning_mask(importance, amount)
                self.apply_pruning(module, mask)
                masks[name] = mask
        
        return masks


class BernoulliPruning(BasePruningStrategy):
    """
    Bernoulli (probabilistic) pruning strategy.
    
    Instead of pruning exactly k% of weights, each weight is pruned
    with probability p = k/100. This can be useful for theoretical
    analysis and stochastic pruning approaches.
    
    Examples:
        >>> from alignment.pruning.strategies import BernoulliPruning
        >>>
        >>> # Each weight has 50% chance of being pruned
        >>> strategy = BernoulliPruning(probability=0.5)
        >>> mask = strategy.prune(layer)
        >>>
        >>> # Different probabilities per layer type
        >>> prob_map = {
        ...     nn.Conv2d: 0.3,
        ...     nn.Linear: 0.5
        ... }
        >>> strategy = BernoulliPruning(probability_map=prob_map)
        >>> masks = strategy.prune_model(model)
    """
    
    def __init__(
        self,
        config=None,
        probability: float = 0.5,
        probability_map: Optional[dict] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Bernoulli pruning strategy.
        
        Args:
            config: Pruning configuration
            probability: Default pruning probability
            probability_map: Dict mapping module types to probabilities
            seed: Random seed
        """
        super().__init__(config)
        self.probability = probability
        self.probability_map = probability_map or {}
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Bernoulli importance scores.
        
        Higher scores mean higher probability of keeping the weight.
        
        Args:
            module: Module to compute scores for
            inputs: Not used
            **kwargs: Can contain 'probability' override
            
        Returns:
            Tensor of importance scores
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        
        # Get probability for this module
        prob = kwargs.get('probability', self.probability)
        
        # Check if module type has specific probability
        for module_type, type_prob in self.probability_map.items():
            if isinstance(module, module_type):
                prob = type_prob
                break
        
        # Generate Bernoulli samples (inverted: 1 means keep)
        # We return uniform random values and let create_pruning_mask
        # handle the thresholding
        importance = torch.rand_like(module.weight.data)
        
        # Adjust scores so that approximately (1-prob) fraction
        # will be below the threshold
        importance = importance / prob if prob > 0 else importance
        
        return importance
    
    def create_pruning_mask(
        self,
        importance_scores: torch.Tensor,
        amount: Optional[float] = None,
        structured: Optional[bool] = None,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create Bernoulli pruning mask.
        
        Overrides base method to use threshold of 1.0 for Bernoulli sampling.
        
        Args:
            importance_scores: Tensor of importance scores
            amount: Not used (probability is set in compute_importance_scores)
            structured: Whether to do structured pruning
            dim: Dimension for structured pruning
            
        Returns:
            Binary mask tensor
        """
        # For Bernoulli pruning, we threshold at 1.0
        mask = (importance_scores > 1.0).float()
        
        if structured and dim is not None:
            # For structured pruning, prune entire structure if
            # majority of weights would be pruned
            dims_to_reduce = list(range(importance_scores.ndim))
            dims_to_reduce.pop(dim)
            
            if dims_to_reduce:
                structure_scores = mask.sum(dim=dims_to_reduce)
                structure_sizes = torch.ones_like(structure_scores)
                for d in dims_to_reduce:
                    structure_sizes *= importance_scores.shape[d]
                
                # Keep structure if more than half weights would survive
                structure_mask = structure_scores > (structure_sizes / 2)
                
                # Expand to original shape
                shape = [1] * importance_scores.ndim
                shape[dim] = importance_scores.shape[dim]
                mask = structure_mask.reshape(shape).expand_as(importance_scores).float()
        
        return mask 