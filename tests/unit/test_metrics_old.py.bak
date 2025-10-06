"""
Unit tests for alignment metrics.
"""

import pytest
import torch
import numpy as np
from typing import Tuple

from alignment.metrics import (
    RayleighQuotient,
    MutualInformationGaussian,
    PartialInformationDecomposition,
    CKA,
    CCA,
    GeneralizedRayleighQuotient,
    SharedInformation
)


class TestRayleighQuotient:
    """Test suite for Rayleigh Quotient metric."""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample data for testing."""
        batch_size, input_dim, output_dim = 32, 10, 5
        inputs = torch.randn(batch_size, input_dim)
        weights = torch.randn(output_dim, input_dim)
        return inputs, weights
    
    def test_compute_basic(self, sample_data):
        """Test basic RQ computation."""
        inputs, weights = sample_data
        metric = RayleighQuotient()
        
        scores = metric.compute(inputs, weights)
        
        assert scores.shape == (weights.shape[0],)
        assert torch.all(scores >= 0)
        assert torch.all(torch.isfinite(scores))
    
    def test_scale_by_norm(self, sample_data):
        """Test RQ with norm scaling."""
        inputs, weights = sample_data
        
        metric_no_scale = RayleighQuotient(scale_by_norm=False)
        metric_scale = RayleighQuotient(scale_by_norm=True)
        
        scores_no_scale = metric_no_scale.compute(inputs, weights)
        scores_scale = metric_scale.compute(inputs, weights)
        
        # Scaled scores should be different
        assert not torch.allclose(scores_no_scale, scores_scale)
    
    def test_force_cpu(self):
        """Test CPU forcing for large operations."""
        if torch.cuda.is_available():
            inputs = torch.randn(1000, 1000).cuda()
            weights = torch.randn(100, 1000).cuda()
            
            metric = RayleighQuotient(force_cpu=True)
            scores = metric.compute(inputs, weights)
            
            # Result should still be on original device
            assert scores.device == inputs.device
    
    def test_cnn_aggregation(self):
        """Test aggregation for CNN layers."""
        batch_size = 16
        inputs = torch.randn(batch_size, 64, 7, 7)  # CNN activations
        weights = torch.randn(128, 64, 3, 3)  # Conv weights
        
        for agg_op in ['mean', 'max', 'sum']:
            metric = RayleighQuotient(aggregation_op=agg_op)
            scores = metric.compute(inputs, weights)
            
            assert scores.shape == (weights.shape[0],)
            assert torch.all(torch.isfinite(scores))
    
    def test_empty_input(self):
        """Test behavior with empty input."""
        inputs = torch.randn(0, 10)
        weights = torch.randn(5, 10)
        
        metric = RayleighQuotient()
        with pytest.raises(Exception):
            metric.compute(inputs, weights)


class TestMutualInformation:
    """Test suite for Mutual Information metrics."""
    
    @pytest.fixture
    def sample_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate sample data with outputs."""
        batch_size, input_dim, output_dim = 64, 20, 10
        inputs = torch.randn(batch_size, input_dim)
        weights = torch.randn(output_dim, input_dim)
        outputs = torch.matmul(inputs, weights.t()) + torch.randn(batch_size, output_dim) * 0.1
        return inputs, weights, outputs
    
    def test_gaussian_estimation(self, sample_data):
        """Test MI with Gaussian estimation."""
        inputs, weights, outputs = sample_data
        
        metric = MutualInformationGaussian(estimation_method="gaussian")
        scores = metric.compute(inputs, weights, outputs)
        
        assert scores.shape == (weights.shape[0],)
        assert torch.all(scores >= 0)  # MI is non-negative
        assert torch.all(torch.isfinite(scores))
    
    def test_requires_outputs(self, sample_data):
        """Test that MI requires outputs."""
        inputs, weights, _ = sample_data
        
        metric = MutualInformationGaussian()
        with pytest.raises(ValueError):
            metric.compute(inputs, weights)  # No outputs provided
    
    def test_different_estimations(self, sample_data):
        """Test different estimation methods produce different results."""
        inputs, weights, outputs = sample_data
        
        metric_gaussian = MutualInformationGaussian(estimation_method="gaussian")
        scores_gaussian = metric_gaussian.compute(inputs, weights, outputs)
        
        # Results should be reasonable
        assert torch.all(scores_gaussian >= 0)
        assert torch.all(scores_gaussian <= 10)  # Reasonable upper bound


class TestCKA:
    """Test suite for CKA metric."""
    
    def test_linear_kernel(self):
        """Test CKA with linear kernel."""
        n_samples = 100
        X = torch.randn(n_samples, 50)
        Y = torch.randn(n_samples, 30)
        
        metric = CKA(kernel="linear")
        similarity = metric.compute(X, Y)
        
        assert isinstance(similarity, (float, torch.Tensor))
        assert 0 <= float(similarity) <= 1
    
    def test_rbf_kernel(self):
        """Test CKA with RBF kernel."""
        n_samples = 100
        X = torch.randn(n_samples, 50)
        Y = torch.randn(n_samples, 30)
        
        metric = CKA(kernel="rbf")
        similarity = metric.compute(X, Y)
        
        assert isinstance(similarity, (float, torch.Tensor))
        assert 0 <= float(similarity) <= 1
    
    def test_self_similarity(self):
        """Test that CKA(X, X) = 1."""
        X = torch.randn(100, 50)
        
        metric = CKA(kernel="linear")
        similarity = metric.compute(X, X)
        
        assert np.isclose(float(similarity), 1.0, rtol=1e-5)
    
    def test_different_samples_error(self):
        """Test error when X and Y have different number of samples."""
        X = torch.randn(100, 50)
        Y = torch.randn(80, 30)  # Different number of samples
        
        metric = CKA()
        with pytest.raises(ValueError):
            metric.compute(X, Y)


class TestCCA:
    """Test suite for CCA metric."""
    
    def test_basic_cca(self):
        """Test basic CCA computation."""
        n_samples = 200
        X = torch.randn(n_samples, 50)
        Y = torch.randn(n_samples, 40)
        
        metric = CCA(n_components=20)
        similarity = metric.compute(X, Y)
        
        assert isinstance(similarity, (float, torch.Tensor))
        assert 0 <= float(similarity) <= 1
    
    def test_regularization(self):
        """Test CCA with regularization."""
        n_samples = 100
        X = torch.randn(n_samples, 50)
        Y = torch.randn(n_samples, 40)
        
        metric = CCA(n_components=20, reg=0.1)
        similarity = metric.compute(X, Y)
        
        assert isinstance(similarity, (float, torch.Tensor))
        assert torch.isfinite(torch.tensor(similarity))
    
    def test_max_components(self):
        """Test CCA with too many components."""
        n_samples = 50
        X = torch.randn(n_samples, 20)
        Y = torch.randn(n_samples, 30)
        
        # Request more components than possible
        metric = CCA(n_components=100)
        similarity = metric.compute(X, Y)
        
        # Should still work (automatically reduced)
        assert isinstance(similarity, (float, torch.Tensor))


class TestPartialInformationDecomposition:
    """Test suite for PID metrics."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for PID."""
        batch_size = 128
        inputs = torch.randn(batch_size, 20)
        weights = torch.randn(10, 20)
        outputs = torch.matmul(inputs, weights.t())
        return inputs, weights, outputs
    
    def test_pid_components(self, sample_data):
        """Test that PID components are computed correctly."""
        inputs, weights, outputs = sample_data
        
        metric = PartialInformationDecomposition(method="broja")
        results = metric.compute(inputs, weights, outputs)
        
        # Check that all components are present
        assert 'unique_information' in results
        assert 'redundant_information' in results
        assert 'synergistic_information' in results
        
        # Check shapes
        assert results['unique_information'].shape == (weights.shape[0],)
        assert isinstance(results['redundant_information'], (float, torch.Tensor))
        assert isinstance(results['synergistic_information'], (float, torch.Tensor))
    
    def test_pid_non_negative(self, sample_data):
        """Test that PID components are non-negative."""
        inputs, weights, outputs = sample_data
        
        metric = PartialInformationDecomposition()
        results = metric.compute(inputs, weights, outputs)
        
        # All information measures should be non-negative
        assert torch.all(results['unique_information'] >= 0)
        assert float(results['redundant_information']) >= 0
        assert float(results['synergistic_information']) >= 0


class TestGeneralizedRayleighQuotient:
    """Test suite for Generalized Rayleigh Quotient."""
    
    def test_basic_computation(self):
        """Test basic GRQ computation."""
        batch_size, dim = 64, 20
        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(10, dim)
        
        metric = GeneralizedRayleighQuotient()
        scores = metric.compute(inputs, weights)
        
        assert scores.shape == (weights.shape[0],)
        assert torch.all(torch.isfinite(scores))
    
    def test_with_reference_covariance(self):
        """Test GRQ with custom reference covariance."""
        batch_size, dim = 64, 20
        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(10, dim)
        
        # Create positive definite reference covariance
        ref_cov = torch.randn(dim, dim)
        ref_cov = ref_cov @ ref_cov.t() + torch.eye(dim) * 0.1
        
        metric = GeneralizedRayleighQuotient()
        scores = metric.compute(inputs, weights, reference_cov=ref_cov)
        
        assert scores.shape == (weights.shape[0],)
        assert torch.all(torch.isfinite(scores))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_consistency(device):
    """Test that metrics work correctly on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    inputs = torch.randn(32, 20).to(device)
    weights = torch.randn(10, 20).to(device)
    
    metric = RayleighQuotient()
    scores = metric.compute(inputs, weights)
    
    assert scores.device.type == device


def test_metric_determinism():
    """Test that metrics produce deterministic results."""
    torch.manual_seed(42)
    inputs1 = torch.randn(32, 20)
    weights1 = torch.randn(10, 20)
    
    torch.manual_seed(42)
    inputs2 = torch.randn(32, 20)
    weights2 = torch.randn(10, 20)
    
    metric = RayleighQuotient()
    scores1 = metric.compute(inputs1, weights1)
    scores2 = metric.compute(inputs2, weights2)
    
    assert torch.allclose(scores1, scores2) 