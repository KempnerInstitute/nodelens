"""
Unit tests for information-theoretic metrics.
"""

import pytest
import torch
import numpy as np
from alignment.metrics.information import (
    MutualInformationGaussian,
    MutualInformationBinning,
    ConditionalMutualInformation,
    MIProjectionVsMeanInput,
)


class TestMutualInformation:
    """Test mutual information metrics."""
    
    def test_gaussian_mi_basic(self):
        """Test Gaussian MI with basic inputs."""
        metric = MutualInformationGaussian()
        
        # Create correlated outputs
        batch_size = 100
        outputs = torch.randn(batch_size, 5)
        # Create target that correlates with first output
        target = outputs[:, 0:1] * 0.8 + torch.randn(batch_size, 1) * 0.2
        
        scores = metric.compute(outputs=outputs, target_outputs=target)
        
        assert scores.shape == (5,)
        assert scores[0] > scores[1]  # First neuron should have highest MI
        assert not torch.isnan(scores).any()
        assert (scores >= 0).all()  # MI should be non-negative
    
    def test_gaussian_mi_no_target(self):
        """Test Gaussian MI without target (uses PC1)."""
        metric = MutualInformationGaussian()
        
        outputs = torch.randn(50, 4)
        scores = metric.compute(outputs=outputs)
        
        assert scores.shape == (4,)
        assert not torch.isnan(scores).any()
        assert (scores >= 0).all()
    
    def test_binning_mi_basic(self):
        """Test binning MI with basic inputs."""
        metric = MutualInformationBinning(bins=5)
        
        # Create discrete-like outputs
        batch_size = 100
        outputs = torch.randn(batch_size, 3)
        # Quantize to make more discrete
        outputs = torch.round(outputs * 2) / 2
        
        target = outputs[:, 0:1] + torch.randn(batch_size, 1) * 0.1
        
        scores = metric.compute(outputs=outputs, target_outputs=target)
        
        assert scores.shape == (3,)
        assert scores[0] > 0  # Should have MI with target
        assert not torch.isnan(scores).any()
    
    def test_insufficient_samples(self):
        """Test with too few samples."""
        metric = MutualInformationGaussian()
        
        outputs = torch.randn(1, 5)  # Only 1 sample
        scores = metric.compute(outputs=outputs)
        
        assert scores.shape == (5,)
        assert (scores == 0).all()  # Should return zeros


class TestConditionalMutualInformation:
    """Test conditional mutual information metric."""
    
    def test_basic_cmi(self):
        """Test basic CMI computation."""
        metric = ConditionalMutualInformation(bins=5)
        
        batch_size = 100
        # Create data where Y correlates with Z given X
        X = torch.randn(batch_size, 5)
        Y = torch.randn(batch_size, 3)
        # Make Y depend on both X and some independent factor
        Y[:, 0] = X[:, 0] * 0.5 + torch.randn(batch_size) * 0.5
        
        scores = metric.compute(inputs=X, outputs=Y)
        
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()
        assert (scores >= 0).all()  # CMI should be non-negative
    
    def test_gaussian_cmi(self):
        """Test CMI with Gaussian approximation."""
        metric = ConditionalMutualInformation(use_gaussian=True)
        
        batch_size = 50
        inputs = torch.randn(batch_size, 4)
        outputs = torch.randn(batch_size, 2)
        
        scores = metric.compute(inputs=inputs, outputs=outputs)
        
        assert scores.shape == (2,)
        assert not torch.isnan(scores).any()
    
    def test_cmi_with_target(self):
        """Test CMI with explicit target."""
        metric = ConditionalMutualInformation()
        
        batch_size = 80
        inputs = torch.randn(batch_size, 3)
        outputs = torch.randn(batch_size, 2)
        target = torch.randn(batch_size, 1)
        
        scores = metric.compute(inputs=inputs, outputs=outputs, target_outputs=target)
        
        assert scores.shape == (2,)
        assert not torch.isnan(scores).any()


class TestMIProjection:
    """Test MI projection vs mean input metric."""
    
    def test_basic_computation(self):
        """Test basic MI projection computation."""
        metric = MIProjectionVsMeanInput(bins=10)
        
        batch_size = 100
        inputs = torch.randn(batch_size, 8)
        # Make inputs have structure
        inputs[:, 1] = inputs[:, 0] * 0.7
        
        weights = torch.randn(5, 8)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()
        assert (scores >= 0).all()  # MI should be non-negative
    
    def test_constant_mean_input(self):
        """Test with constant mean input."""
        metric = MIProjectionVsMeanInput()
        
        # Create inputs where mean is constant
        inputs = torch.ones(50, 5)
        weights = torch.randn(3, 5)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (3,)
        assert (scores == 0).all()  # No variation in mean
    
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatch."""
        metric = MIProjectionVsMeanInput()
        
        inputs = torch.randn(50, 10)
        weights = torch.randn(3, 8)  # Mismatch: 10 vs 8
        
        # Should handle by truncating
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()
    
    def test_force_cpu(self):
        """Test force CPU option."""
        metric = MIProjectionVsMeanInput(force_cpu=True)
        
        if torch.cuda.is_available():
            inputs = torch.randn(50, 6).cuda()
            weights = torch.randn(4, 6).cuda()
        else:
            inputs = torch.randn(50, 6)
            weights = torch.randn(4, 6)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (4,)
        assert not torch.isnan(scores).any()


class TestEdgeCases:
    """Test edge cases for information metrics."""
    
    def test_single_neuron(self):
        """Test with single neuron output."""
        metric = MutualInformationGaussian()
        
        outputs = torch.randn(50, 1)
        scores = metric.compute(outputs=outputs)
        
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()
    
    def test_extreme_values(self):
        """Test with extreme values."""
        metric = MutualInformationBinning(bins=5)
        
        # Very large values
        outputs = torch.randn(50, 3) * 1e6
        scores = metric.compute(outputs=outputs)
        
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        
        # Very small values
        outputs = torch.randn(50, 3) * 1e-6
        scores = metric.compute(outputs=outputs)
        
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
    
    def test_perfect_correlation(self):
        """Test with perfect correlation."""
        metric = MutualInformationGaussian()
        
        outputs = torch.randn(50, 2)
        target = outputs[:, 0:1]  # Perfect correlation
        
        scores = metric.compute(outputs=outputs, target_outputs=target)
        
        assert scores[0] > 1.0  # High MI expected
        assert not torch.isnan(scores).any()
    
    def test_no_correlation(self):
        """Test with no correlation."""
        metric = MutualInformationGaussian()
        
        outputs = torch.randn(100, 3)
        target = torch.randn(100, 1)  # Independent
        
        scores = metric.compute(outputs=outputs, target_outputs=target)
        
        assert scores.max() < 0.5  # Low MI expected
        assert (scores >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 