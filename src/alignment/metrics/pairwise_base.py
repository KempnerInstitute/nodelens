"""
Base class and framework for pairwise metrics.

Pairwise metrics compute relationships between pairs of neurons,
then aggregate to single-neuron scores.

Examples:
- Redundancy: How much does neuron i overlap with others?
- Synergy: How complementary is neuron i with others?
- Correlation: How correlated is neuron i with others?
"""

from abc import abstractmethod
from typing import Optional, Any, Literal
import torch
import logging

from ..core.base import BaseMetric

logger = logging.getLogger(__name__)


class PairwiseMetric(BaseMetric):
    """
    Base class for pairwise metrics.
    
    Pairwise metrics compute relationships between neuron pairs (i, j),
    then aggregate to per-neuron scores.
    
    Workflow:
    1. Compute pairwise values: R[i,j] for all pairs
    2. Aggregate to single-neuron: score[i] = aggregate(R[i, :])
    
    Subclasses implement:
    - compute_pairwise(): Returns [N, N] matrix
    - aggregate_pairwise(): Converts matrix to [N] scores
    
    Example:
        >>> class MyPairwiseMetric(PairwiseMetric):
        ...     def compute_pairwise(self, outputs):
        ...         # Compute [N, N] pairwise relationships
        ...         return pairwise_matrix
        ...
        >>> metric = MyPairwiseMetric(aggregation='mean')
        >>> scores = metric.compute(outputs=outputs)
        >>> # Returns [N] - one score per neuron
    """
    
    def __init__(
        self,
        aggregation: Literal['mean', 'median', 'max', 'sum'] = 'mean',
        exclude_diagonal: bool = True,
        sampling_strategy: str = 'all',  # 'all', 'random', 'nearest'
        num_pairs: int = 10,
        **config: Any
    ):
        """
        Initialize pairwise metric.
        
        Args:
            aggregation: How to aggregate pairwise values to single-neuron scores
                - 'mean': Average over partners (default)
                - 'median': Median over partners
                - 'max': Maximum pairwise value
                - 'sum': Sum over partners
            exclude_diagonal: Whether to exclude self-pairings
            sampling_strategy: How to select pairs
                - 'all': Use all N-1 partners (exact but slow)
                - 'random': Sample K random partners (fast)
                - 'nearest': Use K nearest neighbors
            num_pairs: Number of partners to sample (if not 'all')
            **config: Additional configuration
        """
        super().__init__(**config)
        self.aggregation = aggregation
        self.exclude_diagonal = exclude_diagonal
        self.sampling_strategy = sampling_strategy
        self.num_pairs = num_pairs
    
    @abstractmethod
    def compute_pairwise(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute pairwise relationships between all neurons.
        
        Args:
            inputs: Input activations
            weights: Layer weights
            outputs: Layer outputs
            **kwargs: Additional arguments
            
        Returns:
            Pairwise matrix [N, N] where entry [i,j] is relationship between neurons i and j
        """
        pass
    
    def aggregate_pairwise(
        self,
        pairwise_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate pairwise matrix to per-neuron scores.
        
        Args:
            pairwise_matrix: [N, N] pairwise relationships
            
        Returns:
            Per-neuron scores [N]
        """
        N = pairwise_matrix.shape[0]
        
        # Exclude diagonal if requested
        if self.exclude_diagonal:
            # Create mask
            mask = ~torch.eye(N, dtype=torch.bool, device=pairwise_matrix.device)
            
            if self.aggregation == 'mean':
                # Average over non-diagonal
                row_sums = (pairwise_matrix * mask.float()).sum(dim=1)
                scores = row_sums / max(1, N - 1)
            
            elif self.aggregation == 'median':
                scores = torch.zeros(N, device=pairwise_matrix.device)
                for i in range(N):
                    # Get non-diagonal values for row i
                    values = pairwise_matrix[i, mask[i]]
                    scores[i] = values.median()
            
            elif self.aggregation == 'max':
                scores = torch.zeros(N, device=pairwise_matrix.device)
                for i in range(N):
                    values = pairwise_matrix[i, mask[i]]
                    scores[i] = values.max()
            
            elif self.aggregation == 'sum':
                scores = (pairwise_matrix * mask.float()).sum(dim=1)
            
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        else:
            # Include diagonal
            if self.aggregation == 'mean':
                scores = pairwise_matrix.mean(dim=1)
            elif self.aggregation == 'median':
                scores = pairwise_matrix.median(dim=1)[0]
            elif self.aggregation == 'max':
                scores = pairwise_matrix.max(dim=1)[0]
            elif self.aggregation == 'sum':
                scores = pairwise_matrix.sum(dim=1)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return scores
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        return_matrix: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute per-neuron scores from pairwise relationships.
        
        Args:
            inputs: Input activations
            weights: Layer weights
            outputs: Layer outputs
            return_matrix: If True, return full [N, N] matrix instead of aggregated [N]
            **kwargs: Additional arguments
            
        Returns:
            Per-neuron scores [N] or pairwise matrix [N, N] if return_matrix=True
        """
        # Compute pairwise matrix
        pairwise_matrix = self.compute_pairwise(inputs, weights, outputs, **kwargs)
        
        if return_matrix:
            return pairwise_matrix
        
        # Aggregate to per-neuron scores
        scores = self.aggregate_pairwise(pairwise_matrix)
        
        return scores
    
    def compute_for_subset(
        self,
        neuron_indices: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute scores for a subset of neurons (efficient for large layers).
        
        Args:
            neuron_indices: Indices of neurons to score
            inputs, weights, outputs: As usual
            **kwargs: Additional arguments
            
        Returns:
            Scores for specified neurons only
        """
        # Full computation then subset
        # Subclasses can override for efficiency
        all_scores = self.compute(inputs, weights, outputs, **kwargs)
        return all_scores[neuron_indices]


# Register the base class for documentation
__all__ = ['PairwiseMetric']

