"""
Unit tests for experiment classes.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

from alignment.experiments import (
    ProgressiveDropoutExperiment,
    EigenvectorAlignment,
    LayerIsolatedPruning,
    ExperimentConfig
)
from alignment.models.architectures.standard_models import MLP, CNN2P2
from alignment.metrics import RayleighQuotient


class TestExperimentConfig:
    """Test suite for ExperimentConfig."""
    
    def test_basic_config(self):
        """Test basic configuration creation."""
        config = ExperimentConfig(
            name="test_experiment",
            device="cpu",
            seed=42,
            batch_size=32
        )
        
        assert config.name == "test_experiment"
        assert config.device == "cpu"
        assert config.seed == 42
        assert config.batch_size == 32
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'name': 'test',
            'model_name': 'mlp',
            'dataset_name': 'mnist',
            'dropout_fractions': [0.1, 0.5, 0.9]
        }
        
        config = ExperimentConfig.from_dict(config_dict)
        assert config.name == 'test'
        assert config.model_name == 'mlp'
        assert config.dataset_name == 'mnist'
        assert config.dropout_fractions == [0.1, 0.5, 0.9]
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ExperimentConfig(name="test", batch_size=64)
        config_dict = config.to_dict()
        
        assert config_dict['name'] == 'test'
        assert config_dict['batch_size'] == 64


class TestProgressiveDropoutExperiment:
    """Test suite for Progressive Dropout experiments."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return MLP(input_dim=100, hidden_dims=[50, 30], output_dim=10)
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return ExperimentConfig(
            name="test_dropout",
            device="cpu",
            seed=42,
            batch_size=16,
            dropout_fractions=[0.0, 0.3, 0.6, 0.9],
            metrics=['rayleigh_quotient']
        )
    
    def test_initialization(self, simple_model, test_config):
        """Test experiment initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.output_dir = tmpdir
            
            exp = ProgressiveDropoutExperiment(
                model=simple_model,
                config=test_config
            )
            
            assert exp.model is simple_model
            assert exp.config is test_config
            assert len(exp.dropout_fractions) == 4
    
    def test_get_tracked_layers(self, simple_model, test_config):
        """Test getting tracked layers."""
        exp = ProgressiveDropoutExperiment(
            model=simple_model,
            config=test_config
        )
        
        # Should track linear layers
        tracked = exp.get_tracked_layers()
        assert len(tracked) > 0
        assert all(isinstance(layer, nn.Linear) for _, layer in tracked)
    
    def test_apply_dropout(self, simple_model, test_config):
        """Test applying dropout to model."""
        exp = ProgressiveDropoutExperiment(
            model=simple_model,
            config=test_config
        )
        
        # Get initial weights
        layer_name, layer = exp.get_tracked_layers()[0]
        initial_weight = layer.weight.data.clone()
        
        # Apply dropout
        exp.apply_dropout(0.5)
        
        # Check that some weights are masked
        current_weight = layer.weight.data
        num_masked = (current_weight == 0).sum().item()
        total_params = current_weight.numel()
        
        # Should have roughly 50% masked
        assert 0.4 * total_params < num_masked < 0.6 * total_params


class TestEigenvectorAlignment:
    """Test suite for Eigenvector Alignment experiments."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return ExperimentConfig(
            name="test_eigen",
            device="cpu",
            num_components=5,
            metrics=['rayleigh_quotient']
        )
    
    def test_initialization(self, test_config):
        """Test eigenvector experiment initialization."""
        model = MLP(input_dim=50, hidden_dims=[30], output_dim=10)
        
        exp = EigenvectorAlignment(
            model=model,
            config=test_config
        )
        
        assert exp.num_components == 5
        assert exp.model is model
    
    def test_compute_eigenvectors(self, test_config):
        """Test eigenvector computation."""
        model = MLP(input_dim=20, hidden_dims=[10], output_dim=5)
        exp = EigenvectorAlignment(model=model, config=test_config)
        
        # Create mock data
        data = torch.randn(100, 20)
        
        # Compute eigenvectors
        layer_name, layer = exp.get_tracked_layers()[0]
        eigenvectors, eigenvalues = exp.compute_layer_eigenvectors(data, layer)
        
        assert eigenvectors.shape[0] == min(test_config.num_components, 20)
        assert eigenvalues.shape[0] == min(test_config.num_components, 20)
        assert torch.all(eigenvalues[:-1] >= eigenvalues[1:])  # Descending order


class TestLayerIsolatedPruning:
    """Test suite for Layer Isolated Pruning experiments."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return ExperimentConfig(
            name="test_pruning",
            device="cpu",
            pruning_percentages=[0.1, 0.3, 0.5],
            metrics=['rayleigh_quotient']
        )
    
    def test_initialization(self, test_config):
        """Test pruning experiment initialization."""
        model = CNN2P2(output_dim=10, example_input_hw=[28, 28])
        
        exp = LayerIsolatedPruning(
            model=model,
            config=test_config
        )
        
        assert exp.model is model
        assert exp.pruning_percentages == [0.1, 0.3, 0.5]
    
    def test_prune_layer(self, test_config):
        """Test layer pruning."""
        model = MLP(input_dim=50, hidden_dims=[30, 20], output_dim=10)
        exp = LayerIsolatedPruning(model=model, config=test_config)
        
        # Get a layer to prune
        layer_name, layer = exp.get_tracked_layers()[0]
        initial_weight = layer.weight.data.clone()
        
        # Apply magnitude-based pruning
        exp.prune_layer(layer, percentage=0.3, strategy='magnitude')
        
        # Check pruning
        current_weight = layer.weight.data
        num_pruned = (current_weight == 0).sum().item()
        total_params = current_weight.numel()
        
        # Should have roughly 30% pruned
        assert abs(num_pruned / total_params - 0.3) < 0.05


@pytest.mark.integration
class TestExperimentIntegration:
    """Integration tests for experiments."""
    
    def test_full_experiment_run(self):
        """Test running a complete experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config = ExperimentConfig(
                name="integration_test",
                device="cpu",
                seed=42,
                batch_size=8,
                dropout_fractions=[0.0, 0.5],
                metrics=['rayleigh_quotient'],
                output_dir=tmpdir,
                checkpoint_interval=None
            )
            
            # Create model
            model = MLP(input_dim=20, hidden_dims=[10], output_dim=5)
            
            # Create experiment
            exp = ProgressiveDropoutExperiment(
                model=model,
                config=config
            )
            
            # Create mock data
            train_data = [(torch.randn(20), torch.randint(0, 5, (1,))) for _ in range(16)]
            val_data = [(torch.randn(20), torch.randint(0, 5, (1,))) for _ in range(8)]
            
            # Run experiment (simplified)
            results = {}
            for fraction in config.dropout_fractions:
                exp.apply_dropout(fraction)
                
                # Mock evaluation
                metric_results = {'rayleigh_quotient': torch.rand(10)}
                results[f'dropout_{fraction}'] = metric_results
            
            # Check results structure
            assert len(results) == 2
            assert 'dropout_0.0' in results
            assert 'dropout_0.5' in results 