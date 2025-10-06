"""
Parallel pruning strategies that apply multiple pruning modes simultaneously.

This module provides strategies for applying multiple pruning criteria
in parallel, either as separate experiments or as a combined tensor operation.
"""

import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..base import BasePruningStrategy, PruningConfig


@dataclass
class ParallelPruningResult:
    """Result container for parallel pruning operations."""
    masks: Dict[str, torch.Tensor]  # Pruning mode -> mask
    combined_mask: Optional[torch.Tensor] = None
    sparsities: Optional[Dict[str, float]] = None
    importance_scores: Optional[torch.Tensor] = None


class ParallelModePruning(BasePruningStrategy):
    """
    Apply multiple pruning modes (low, high, random) in parallel.

    This strategy computes importance scores once and generates multiple
    masks for different pruning modes simultaneously.

    Examples:
        >>> strategy = ParallelModePruning(modes=['low', 'high', 'random'])
        >>> result = strategy.prune_parallel(model.fc1, amount=0.5)
        >>>
        >>> # Access individual masks
        >>> low_mask = result.masks['low']
        >>> high_mask = result.masks['high']
        >>>
        >>> # Or get combined analysis
        >>> combined = strategy.combine_masks(result.masks, method='union')
    """

    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        modes: List[str] = ['low', 'high', 'random'],
        base_strategy: str = 'magnitude'
    ):
        """
        Initialize parallel mode pruning.

        Args:
            config: Pruning configuration
            modes: List of pruning modes to apply ('low', 'high', 'random')
            base_strategy: Base strategy for computing importance scores
        """
        super().__init__(config)
        self.modes = modes
        self.base_strategy = base_strategy

        # Import strategies dynamically to avoid circular imports
        from . import GradientPruning, MagnitudePruning, RandomPruning

        self.strategy_map = {
            'magnitude': MagnitudePruning,
            'gradient': GradientPruning,
            'random': RandomPruning,
        }

        # Initialize base strategy for importance computation
        if base_strategy in self.strategy_map:
            self.importance_strategy = self.strategy_map[base_strategy](config)
        else:
            raise ValueError(f"Unknown base strategy: {base_strategy}")

    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute importance scores using the base strategy."""
        return self.importance_strategy.compute_importance_scores(module, inputs, **kwargs)

    def prune_parallel(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        amount: Optional[float] = None,
        **kwargs
    ) -> ParallelPruningResult:
        """
        Apply multiple pruning modes in parallel.

        Args:
            module: Module to prune
            inputs: Optional inputs for importance computation
            amount: Pruning amount
            **kwargs: Additional arguments

        Returns:
            ParallelPruningResult containing all masks and metadata
        """
        amount = amount if amount is not None else self.config.amount

        # Compute importance scores once
        importance_scores = self.compute_importance_scores(module, inputs, **kwargs)

        # Generate masks for each mode in parallel
        masks = {}
        sparsities = {}

        for mode in self.modes:
            if mode == 'random':
                # Random doesn't use importance scores
                mask = self._create_random_mask(module.weight.shape, amount)
            else:
                # Create mask with specific mode
                mask = self.create_pruning_mask(
                    importance_scores,
                    amount=amount,
                    pruning_mode=mode
                )

            masks[mode] = mask
            sparsities[mode] = (mask == 0).float().mean().item()

        return ParallelPruningResult(
            masks=masks,
            sparsities=sparsities,
            importance_scores=importance_scores
        )

    def _create_random_mask(self, shape: torch.Size, amount: float) -> torch.Tensor:
        """Create a random pruning mask."""
        mask = torch.rand(shape) > amount
        return mask.float()

    def combine_masks(
        self,
        masks: Dict[str, torch.Tensor],
        method: str = 'intersection'
    ) -> torch.Tensor:
        """
        Combine multiple masks using specified method.

        Args:
            masks: Dictionary of masks to combine
            method: Combination method ('intersection', 'union', 'majority')

        Returns:
            Combined mask
        """
        mask_list = list(masks.values())

        if method == 'intersection':
            # Keep only weights that all masks agree to keep
            combined = torch.stack(mask_list).prod(dim=0)
        elif method == 'union':
            # Keep weights that any mask wants to keep
            combined = torch.stack(mask_list).max(dim=0)[0]
        elif method == 'majority':
            # Keep weights that majority of masks want to keep
            combined = torch.stack(mask_list).mean(dim=0) > 0.5
            combined = combined.float()
        else:
            raise ValueError(f"Unknown combination method: {method}")

        return combined


class TensorizedPruning(BasePruningStrategy):
    """
    Tensorized pruning that computes all pruning variations as a single tensor operation.

    This strategy is optimized for GPU computation and analysis of pruning patterns.

    Examples:
        >>> strategy = TensorizedPruning()
        >>> # Get a 3D tensor: [num_modes, height, width] for Conv2d
        >>> pruning_tensor = strategy.compute_pruning_tensor(
        ...     conv_layer,
        ...     modes=['low', 'high'],
        ...     amounts=[0.1, 0.3, 0.5, 0.7, 0.9]
        ... )
    """

    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute importance scores (magnitude-based by default)."""
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        return module.weight.data.abs()

    def compute_pruning_tensor(
        self,
        module: nn.Module,
        modes: List[str] = ['low', 'high'],
        amounts: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute a tensor containing all pruning variations.

        Args:
            module: Module to analyze
            modes: Pruning modes to include
            amounts: Sparsity levels to include
            inputs: Optional inputs for importance computation

        Returns:
            Tensor of shape [len(modes), len(amounts), *weight_shape]
        """
        # Get importance scores
        importance_scores = self.compute_importance_scores(module, inputs)
        weight_shape = importance_scores.shape

        # Flatten for efficient computation
        scores_flat = importance_scores.flatten()
        n_params = scores_flat.numel()

        # Pre-compute all thresholds
        thresholds = {}
        for amount in amounts:
            k = int(amount * n_params)
            if k > 0 and k < n_params:
                thresholds[amount] = {
                    'low': scores_flat.kthvalue(k).values,
                    'high': scores_flat.kthvalue(n_params - k).values
                }

        # Create pruning tensor
        pruning_tensor = torch.zeros(
            len(modes), len(amounts), *weight_shape,
            dtype=torch.float32,
            device=importance_scores.device
        )

        for i, mode in enumerate(modes):
            for j, amount in enumerate(amounts):
                if amount in thresholds:
                    if mode == 'low':
                        mask = importance_scores > thresholds[amount]['low']
                    elif mode == 'high':
                        mask = importance_scores < thresholds[amount]['high']
                    elif mode == 'random':
                        mask = torch.rand_like(importance_scores) > amount
                    else:
                        continue

                    pruning_tensor[i, j] = mask.float()
                else:
                    # Edge cases (0% or 100% pruning)
                    if amount == 0:
                        pruning_tensor[i, j] = torch.ones_like(importance_scores)
                    else:
                        pruning_tensor[i, j] = torch.zeros_like(importance_scores)

        return pruning_tensor

    def analyze_pruning_patterns(
        self,
        pruning_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze patterns in the pruning tensor.

        Args:
            pruning_tensor: Tensor from compute_pruning_tensor

        Returns:
            Dictionary of analysis metrics
        """
        analysis = {}

        # Sparsity progression
        analysis['sparsity_progression'] = (pruning_tensor == 0).float().mean(dim=-1).mean(dim=-1)

        # Overlap between modes
        if pruning_tensor.shape[0] >= 2:
            overlap = pruning_tensor[0] * pruning_tensor[1]  # Low & High overlap
            analysis['mode_overlap'] = overlap.mean(dim=-1).mean(dim=-1)

        # Variance across amounts
        analysis['pruning_variance'] = pruning_tensor.var(dim=1)

        return analysis


class AsyncParallelPruning(BasePruningStrategy):
    """
    Asynchronous parallel pruning for multiple models or layers.

    This strategy uses Python's concurrent.futures to prune multiple
    modules in parallel across CPU cores.

    Examples:
        >>> strategy = AsyncParallelPruning()
        >>> modules = [model.layer1, model.layer2, model.layer3]
        >>> results = strategy.prune_modules_parallel(
        ...     modules,
        ...     amounts=[0.5, 0.6, 0.7],
        ...     modes=['low', 'high']
        ... )
    """

    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute importance scores."""
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        return module.weight.data.abs()

    def prune_modules_parallel(
        self,
        modules: List[nn.Module],
        amounts: Union[float, List[float]],
        modes: List[str] = ['low'],
        max_workers: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Prune multiple modules in parallel.

        Args:
            modules: List of modules to prune
            amounts: Single amount or list of amounts per module
            modes: Pruning modes to apply
            max_workers: Maximum number of parallel workers

        Returns:
            List of pruning results per module
        """
        if isinstance(amounts, float):
            amounts = [amounts] * len(modules)

        def prune_single_module(module, amount):
            """Prune a single module with all modes."""
            masks = {}

            importance_scores = self.compute_importance_scores(module)

            for mode in modes:
                if mode == 'random':
                    mask = torch.rand_like(module.weight) > amount
                    masks[mode] = mask.float()
                else:
                    mask = self.create_pruning_mask(
                        importance_scores,
                        amount=amount,
                        pruning_mode=mode
                    )
                    masks[mode] = mask

            return masks

        # Execute in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(prune_single_module, module, amount)
                for module, amount in zip(modules, amounts)
            ]

            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results
