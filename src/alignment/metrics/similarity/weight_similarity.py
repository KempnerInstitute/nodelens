"""
Weight similarity metrics for measuring relationships between neuron weight vectors.
"""

import torch
import logging
from typing import Optional
from ...core.base import BaseMetric

logger = logging.getLogger(__name__)


class WeightSimilarityBase(BaseMetric):
    """Base class for weight similarity metrics."""
    
    requires_weights = True
    requires_inputs = False
    requires_outputs = False
    
    @torch.no_grad()
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute weight similarity scores.
        
        Args:
            inputs: Not used
            weights: Weight matrix [num_neurons, num_features]
            outputs: Not used
            **kwargs: Additional arguments
            
        Returns:
            Similarity scores per neuron [num_neurons]
        """
        if weights is None:
            raise ValueError(f"{self.name} requires weights")
        
        # Handle different weight dimensions
        if weights.ndim != 2:
            if weights.ndim > 2:
                # Flatten to 2D
                weights = weights.reshape(weights.shape[0], -1)
            else:
                logger.warning(f"Weights have unexpected shape: {weights.shape}")
                return torch.zeros(1, device=weights.device)
        
        num_neurons = weights.shape[0]
        
        if num_neurons <= 1:
            return torch.zeros(num_neurons, device=weights.device)
        
        return self._compute_similarity(weights)
    
    def _compute_similarity(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores. To be implemented by subclasses."""
        raise NotImplementedError


class WeightCosineSimilarity(WeightSimilarityBase):
    """
    Compute average cosine similarity between each neuron's weights and all others.
    """
    
    name = "weight_cosine_similarity"
    
    def _compute_similarity(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute average cosine similarity for each neuron."""
        num_neurons = weights.shape[0]
        similarity_scores = torch.zeros(num_neurons, device=weights.device)
        
        # Normalize weight vectors
        weight_norms = torch.norm(weights, dim=1, keepdim=True)
        normalized_weights = torch.where(
            weight_norms > 1e-12,
            weights / weight_norms,
            torch.zeros_like(weights)
        )
        
        # Compute pairwise cosine similarities
        cosine_sim_matrix = torch.matmul(normalized_weights, normalized_weights.T)
        
        # Average similarity with other neurons (excluding self)
        for i in range(num_neurons):
            mask = torch.ones(num_neurons, dtype=torch.bool, device=weights.device)
            mask[i] = False
            if mask.sum() > 0:
                similarity_scores[i] = cosine_sim_matrix[i, mask].mean()
        
        return similarity_scores


class WeightDotSimilarity(WeightSimilarityBase):
    """
    Compute average dot product between each neuron's weights and all others.
    """
    
    name = "weight_dot_similarity"
    
    def _compute_similarity(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute average dot product for each neuron."""
        num_neurons = weights.shape[0]
        similarity_scores = torch.zeros(num_neurons, device=weights.device)
        
        # Compute pairwise dot products
        dot_product_matrix = torch.matmul(weights, weights.T)
        
        # Average dot product with other neurons (excluding self)
        for i in range(num_neurons):
            mask = torch.ones(num_neurons, dtype=torch.bool, device=weights.device)
            mask[i] = False
            if mask.sum() > 0:
                similarity_scores[i] = dot_product_matrix[i, mask].mean()
        
        return similarity_scores


class WeightEuclideanDistance(WeightSimilarityBase):
    """
    Compute average Euclidean distance between each neuron's weights and all others.
    Note: Lower values indicate higher similarity.
    """
    
    name = "weight_euclidean_distance"
    
    def _compute_similarity(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute average Euclidean distance for each neuron."""
        num_neurons = weights.shape[0]
        distance_scores = torch.zeros(num_neurons, device=weights.device)
        
        # Compute pairwise Euclidean distances efficiently
        # ||w_i - w_j||^2 = ||w_i||^2 + ||w_j||^2 - 2 * w_i @ w_j
        weight_norms_sq = torch.sum(weights ** 2, dim=1, keepdim=True)
        dot_products = torch.matmul(weights, weights.T)
        
        # Distance matrix
        dist_matrix_sq = weight_norms_sq + weight_norms_sq.T - 2 * dot_products
        dist_matrix = torch.sqrt(torch.clamp(dist_matrix_sq, min=0))
        
        # Average distance to other neurons (excluding self)
        for i in range(num_neurons):
            mask = torch.ones(num_neurons, dtype=torch.bool, device=weights.device)
            mask[i] = False
            if mask.sum() > 0:
                distance_scores[i] = dist_matrix[i, mask].mean()
        
        return distance_scores 