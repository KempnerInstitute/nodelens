"""
Partial Information Decomposition (PID) metrics.

This module implements PID-based metrics using the BROJA method for
computing shared, unique, and synergistic information between variables.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np
import logging
from pathlib import Path

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)

# Try to import BROJA module
BROJA_AVAILABLE = False
try:
    # Try to import from the expected location
    from alignment.external.BROJA_2PID import BROJA_2PID
    BROJA_AVAILABLE = True
    logger.info("Successfully loaded BROJA_2PID module")
except ImportError:
    try:
        # Alternative path
        import sys
        broja_path = Path(__file__).parent.parent.parent / "external" / "BROJA_2PID" / "BROJA_2PID.py"
        if broja_path.exists():
            sys.path.insert(0, str(broja_path.parent))
            import BROJA_2PID
            BROJA_AVAILABLE = True
            logger.info("Successfully loaded BROJA_2PID module from alternative path")
        else:
            logger.warning(f"BROJA_2PID not found at {broja_path}")
    except ImportError:
        logger.warning("BROJA_2PID module not available. PID metrics will return zeros.")


class BasePIDMetric(BaseMetric):
    """Base class for PID-based metrics."""
    
    # PID metrics require inputs and outputs
    requires_inputs = True
    requires_weights = False
    requires_outputs = True
    
    def __init__(
        self,
        bins: int = 10,
        normalize: bool = True,
        use_pca_inputs: bool = True,
        **kwargs
    ):
        """
        Initialize PID metric.
        
        Args:
            bins: Number of bins for discretization
            normalize: Whether to normalize the result
            use_pca_inputs: Whether to use PCA on inputs
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.bins = bins
        self.normalize = normalize
        self.use_pca_inputs = use_pca_inputs
    
    def _prepare_pid_input(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y: torch.Tensor
    ) -> Optional[Dict[Tuple[int, int, int], float]]:
        """
        Prepare input for BROJA PID computation.
        
        Args:
            x1: First input variable
            x2: Second input variable  
            y: Output variable
            
        Returns:
            Probability distribution dictionary or None if error
        """
        if not BROJA_AVAILABLE:
            return None
        
        try:
            # Convert to numpy and discretize
            x1_discrete = np.digitize(x1.cpu().numpy(), bins=np.linspace(x1.min(), x1.max(), self.bins))
            x2_discrete = np.digitize(x2.cpu().numpy(), bins=np.linspace(x2.min(), x2.max(), self.bins))
            y_discrete = np.digitize(y.cpu().numpy(), bins=np.linspace(y.min(), y.max(), self.bins))
            
            # Create joint probability distribution
            joint_prob = {}
            n_samples = len(x1_discrete)
            
            for i in range(n_samples):
                key = (int(x1_discrete[i]), int(x2_discrete[i]), int(y_discrete[i]))
                joint_prob[key] = joint_prob.get(key, 0) + 1.0 / n_samples
            
            return joint_prob
            
        except Exception as e:
            logger.error(f"Error preparing PID input: {e}")
            return None
    
    def _compute_pca_features(self, inputs: torch.Tensor, n_components: int = 2) -> torch.Tensor:
        """
        Compute PCA features from inputs.
        
        Args:
            inputs: Input tensor [batch, features]
            n_components: Number of components to keep
            
        Returns:
            PCA transformed features
        """
        # Center the data
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov = torch.mm(inputs_centered.t(), inputs_centered) / (inputs.size(0) - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvectors = eigenvectors[:, idx[:n_components]]
        
        # Project data
        pca_features = torch.mm(inputs_centered, eigenvectors)
        
        return pca_features


@register_metric("pid_shared")
class SharedInformation(BasePIDMetric):
    """
    Computes shared information (redundancy) between two inputs about output.
    
    This measures the information that both inputs provide redundantly
    about the output.
    """
    
    name = "pid_shared"
    
    def compute(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute shared information for each output neuron.
        
        Args:
            inputs: Input activations [batch, input_features]
            outputs: Output activations [batch, output_features]
            weights: Not used for this metric
            **kwargs: Additional arguments
            
        Returns:
            Shared information scores [output_features]
        """
        if not BROJA_AVAILABLE:
            logger.warning("BROJA not available, returning zeros")
            return torch.zeros(outputs.shape[1], device=outputs.device)
        
        batch_size, n_outputs = outputs.shape
        scores = torch.zeros(n_outputs, device=outputs.device)
        
        # Get input features (use PCA if enabled)
        if self.use_pca_inputs and inputs.shape[1] > 2:
            input_features = self._compute_pca_features(inputs, n_components=2)
        else:
            # Use first two features
            input_features = inputs[:, :2]
        
        # Compute PID for each output neuron
        for i in range(n_outputs):
            if input_features.shape[1] < 2:
                continue
                
            # Prepare PID input
            pid_input = self._prepare_pid_input(
                input_features[:, 0],
                input_features[:, 1],
                outputs[:, i]
            )
            
            if pid_input is None:
                continue
            
            try:
                # Compute PID
                pid_result = BROJA_2PID.pid(pid_input)
                scores[i] = pid_result.get("SI", 0.0)  # Shared Information
                
            except Exception as e:
                logger.debug(f"PID computation failed for neuron {i}: {e}")
                continue
        
        if self.normalize and scores.max() > 0:
            scores = scores / scores.max()
        
        return scores


@register_metric("pid_unique_x")
class UniqueInformationX(BasePIDMetric):
    """
    Computes unique information from first input variable about output.
    
    This measures information that only the first input provides
    about the output (not available from the second input).
    """
    
    name = "pid_unique_x"
    
    def compute(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute unique information from first input variable."""
        if not BROJA_AVAILABLE:
            logger.warning("BROJA not available, returning zeros")
            return torch.zeros(outputs.shape[1], device=outputs.device)
        
        batch_size, n_outputs = outputs.shape
        scores = torch.zeros(n_outputs, device=outputs.device)
        
        # Get input features
        if self.use_pca_inputs and inputs.shape[1] > 2:
            input_features = self._compute_pca_features(inputs, n_components=2)
        else:
            input_features = inputs[:, :2]
        
        for i in range(n_outputs):
            if input_features.shape[1] < 2:
                continue
                
            pid_input = self._prepare_pid_input(
                input_features[:, 0],
                input_features[:, 1],
                outputs[:, i]
            )
            
            if pid_input is None:
                continue
            
            try:
                pid_result = BROJA_2PID.pid(pid_input)
                scores[i] = pid_result.get("UIY", 0.0)  # Unique info from Y (first var)
                
            except Exception as e:
                logger.debug(f"PID computation failed for neuron {i}: {e}")
                continue
        
        if self.normalize and scores.max() > 0:
            scores = scores / scores.max()
        
        return scores


@register_metric("pid_unique_y")  
class UniqueInformationY(BasePIDMetric):
    """
    Computes unique information from second input variable about output.
    
    This measures information that only the second input provides
    about the output (not available from the first input).
    """
    
    name = "pid_unique_y"
    
    def compute(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute unique information from second input variable."""
        if not BROJA_AVAILABLE:
            logger.warning("BROJA not available, returning zeros")
            return torch.zeros(outputs.shape[1], device=outputs.device)
        
        batch_size, n_outputs = outputs.shape
        scores = torch.zeros(n_outputs, device=outputs.device)
        
        # Get input features
        if self.use_pca_inputs and inputs.shape[1] > 2:
            input_features = self._compute_pca_features(inputs, n_components=2)
        else:
            input_features = inputs[:, :2]
        
        for i in range(n_outputs):
            if input_features.shape[1] < 2:
                continue
                
            pid_input = self._prepare_pid_input(
                input_features[:, 0],
                input_features[:, 1],
                outputs[:, i]
            )
            
            if pid_input is None:
                continue
            
            try:
                pid_result = BROJA_2PID.pid(pid_input)
                scores[i] = pid_result.get("UIZ", 0.0)  # Unique info from Z (second var)
                
            except Exception as e:
                logger.debug(f"PID computation failed for neuron {i}: {e}")
                continue
        
        if self.normalize and scores.max() > 0:
            scores = scores / scores.max()
        
        return scores


@register_metric("pid_synergy")
class SynergisticInformation(BasePIDMetric):
    """
    Computes synergistic information between inputs about output.
    
    This measures information about the output that is only available
    when both inputs are considered together (emergent information).
    """
    
    name = "pid_synergy"
    
    def compute(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute synergistic information."""
        if not BROJA_AVAILABLE:
            logger.warning("BROJA not available, returning zeros")
            return torch.zeros(outputs.shape[1], device=outputs.device)
        
        batch_size, n_outputs = outputs.shape
        scores = torch.zeros(n_outputs, device=outputs.device)
        
        # Get input features
        if self.use_pca_inputs and inputs.shape[1] > 2:
            input_features = self._compute_pca_features(inputs, n_components=2)
        else:
            input_features = inputs[:, :2]
        
        for i in range(n_outputs):
            if input_features.shape[1] < 2:
                continue
                
            pid_input = self._prepare_pid_input(
                input_features[:, 0],
                input_features[:, 1],
                outputs[:, i]
            )
            
            if pid_input is None:
                continue
            
            try:
                pid_result = BROJA_2PID.pid(pid_input)
                scores[i] = pid_result.get("CI", 0.0)  # Synergistic/Complementary info
                
            except Exception as e:
                logger.debug(f"PID computation failed for neuron {i}: {e}")
                continue
        
        if self.normalize and scores.max() > 0:
            scores = scores / scores.max()
        
        return scores


# Aliases for backward compatibility
PartialInformationDecomposition = SharedInformation 