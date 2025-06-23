"""
Cosine similarity metrics for neural network alignment analysis.

These metrics measure alignment through cosine similarity between
weight vectors or activation patterns.
"""

from typing import Optional, Any
import torch
import torch.nn.functional as F
import logging

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("weight_cosine_similarity", aliases=["cosine_similarity"])
class WeightCosineSimilarity(BaseMetric):
    """
    Cosine similarity between weight vectors.
    
    For each neuron, computes the average cosine similarity with
    other neurons in the same layer, measuring weight alignment.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        exclude_self: bool = True,
        **config: Any
    ):
        """
        Initialize weight cosine similarity metric.
        
        Args:
            normalize: Whether to normalize weights before computing similarity
            exclude_self: Whether to exclude self-similarity (always 1.0)
            **config: Additional configuration
        """
        super().__init__(**config)
        self.normalize = normalize
        self.exclude_self = exclude_self
    
    @property
    def requires_inputs(self) -> bool:
        return False
    
    @property
    def requires_weights(self) -> bool:
        return True
    
    @property
    def requires_outputs(self) -> bool:
        return False
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute average cosine similarity for each weight vector.
        
        Args:
            inputs: Not used
            weights: Layer weights [num_neurons, input_features]
            outputs: Not used
            
        Returns:
            Average cosine similarity scores [num_neurons]
        """
        if weights is None:
            raise ValueError("WeightCosineSimilarity requires weights")
        
        if weights.ndim != 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        num_neurons = weights.shape[0]
        
        if num_neurons <= 1:
            logger.warning("WeightCosineSimilarity: Need at least 2 neurons")
            return torch.ones(num_neurons, device=weights.device, dtype=weights.dtype)
        
        # Normalize weight vectors
        if self.normalize:
            weights_norm = F.normalize(weights, p=2, dim=1)
        else:
            # Manual normalization for stability
            weight_norms = torch.norm(weights, p=2, dim=1, keepdim=True)
            weight_norms = torch.where(weight_norms > 1e-12, weight_norms, torch.ones_like(weight_norms))
            weights_norm = weights / weight_norms
        
        # Compute pairwise cosine similarities
        similarity_matrix = torch.matmul(weights_norm, weights_norm.T)
        
        # Compute average similarity for each neuron
        similarity_scores = torch.zeros(num_neurons, device=weights.device)
        
        for i in range(num_neurons):
            if self.exclude_self:
                # Average similarity with other neurons
                mask = torch.ones(num_neurons, dtype=torch.bool, device=weights.device)
                mask[i] = False
                if mask.sum() > 0:
                    similarity_scores[i] = similarity_matrix[i, mask].mean()
            else:
                # Include self-similarity
                similarity_scores[i] = similarity_matrix[i].mean()
        
        return similarity_scores


@register_metric("activation_cosine_similarity")
class ActivationCosineSimilarity(BaseMetric):
    """
    Cosine similarity between activation patterns.
    
    Measures how similar the activation patterns of neurons are
    across the batch dimension.
    """
    
    def __init__(
        self,
        min_samples: int = 2,
        exclude_self: bool = True,
        use_outputs: bool = True,
        **config: Any
    ):
        """
        Initialize activation cosine similarity metric.
        
        Args:
            min_samples: Minimum samples for meaningful patterns
            exclude_self: Whether to exclude self-similarity
            use_outputs: If True, use outputs; if False, compute from inputs/weights
            **config: Additional configuration
        """
        super().__init__(**config)
        self.min_samples = min_samples
        self.exclude_self = exclude_self
        self.use_outputs = use_outputs
    
    @property
    def requires_inputs(self) -> bool:
        return not self.use_outputs
    
    @property
    def requires_weights(self) -> bool:
        return not self.use_outputs
    
    @property
    def requires_outputs(self) -> bool:
        return self.use_outputs
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute average cosine similarity between activation patterns.
        
        Args:
            inputs: Input activations (if use_outputs=False)
            weights: Layer weights (if use_outputs=False)
            outputs: Output activations (if use_outputs=True)
            
        Returns:
            Average activation similarity scores [num_neurons]
        """
        # Get activations
        if self.use_outputs:
            if outputs is None:
                raise ValueError("ActivationCosineSimilarity requires outputs when use_outputs=True")
            activations = outputs
        else:
            if inputs is None or weights is None:
                raise ValueError("ActivationCosineSimilarity requires inputs and weights when use_outputs=False")
            
            # Compute activations
            if inputs.ndim != 2:
                inputs = inputs.reshape(inputs.shape[0], -1)
            if weights.ndim != 2:
                weights = weights.reshape(weights.shape[0], -1)
            
            activations = torch.matmul(inputs, weights.T)
        
        if activations.ndim != 2:
            activations = activations.reshape(activations.shape[0], -1)
        
        batch_size, num_neurons = activations.shape
        
        if batch_size < self.min_samples:
            logger.warning(f"ActivationCosineSimilarity: Only {batch_size} samples")
            return torch.zeros(num_neurons, device=activations.device, dtype=activations.dtype)
        
        if num_neurons <= 1:
            return torch.ones(num_neurons, device=activations.device, dtype=activations.dtype)
        
        # Normalize activation patterns (across batch dimension)
        activations_norm = F.normalize(activations, p=2, dim=0)  # Normalize each neuron's pattern
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(activations_norm.T, activations_norm) / batch_size
        
        # Average similarity for each neuron
        similarity_scores = torch.zeros(num_neurons, device=activations.device)
        
        for i in range(num_neurons):
            if self.exclude_self:
                mask = torch.ones(num_neurons, dtype=torch.bool, device=activations.device)
                mask[i] = False
                if mask.sum() > 0:
                    similarity_scores[i] = similarity_matrix[i, mask].mean()
            else:
                similarity_scores[i] = similarity_matrix[i].mean()
        
        return similarity_scores


@register_metric("weight_activation_alignment")
class WeightActivationAlignment(BaseMetric):
    """
    Alignment between weight vectors and activation covariance.
    
    Measures how well each weight vector aligns with the principal
    components of the activation covariance, using cosine similarity.
    """
    
    def __init__(
        self,
        n_components: int = 5,
        min_samples: int = 10,
        **config: Any
    ):
        """
        Initialize weight-activation alignment metric.
        
        Args:
            n_components: Number of principal components to consider
            min_samples: Minimum samples for PCA
            **config: Additional configuration
        """
        super().__init__(**config)
        self.n_components = n_components
        self.min_samples = min_samples
    
    @property
    def requires_inputs(self) -> bool:
        return True
    
    @property
    def requires_weights(self) -> bool:
        return True
    
    @property
    def requires_outputs(self) -> bool:
        return False
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute alignment between weights and activation PCs.
        
        Returns:
            Maximum cosine similarity with top PCs [num_neurons]
        """
        if inputs is None or weights is None:
            raise ValueError("WeightActivationAlignment requires inputs and weights")
        
        if inputs.ndim != 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim != 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        batch_size, input_features = inputs.shape
        num_neurons = weights.shape[0]
        
        if batch_size < self.min_samples:
            logger.warning(f"WeightActivationAlignment: Only {batch_size} samples")
            return torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)
        
        # Compute input covariance and eigendecomposition
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
        cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
        
        # Get top eigenvectors
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Take top n_components (largest eigenvalues)
            n_comp = min(self.n_components, input_features)
            top_eigenvecs = eigenvectors[:, -n_comp:]  # [input_features, n_components]
        except Exception as e:
            logger.warning(f"WeightActivationAlignment: Eigendecomposition failed: {e}")
            return torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)
        
        # Normalize weights and eigenvectors
        weights_norm = F.normalize(weights, p=2, dim=1)
        eigenvecs_norm = F.normalize(top_eigenvecs, p=2, dim=0)
        
        # Compute cosine similarities with each PC
        similarities = torch.matmul(weights_norm, eigenvecs_norm)  # [num_neurons, n_components]
        
        # Take maximum similarity across PCs for each neuron
        max_similarities, _ = torch.max(torch.abs(similarities), dim=1)
        
        return max_similarities 