"""
Unit tests for experiment classes.
"""

from nodelens.experiments.base import ExperimentConfig
from nodelens.experiments.general_alignment import GeneralAlignmentConfig
from nodelens.pruning.base import PruningConfig


class TestExperimentConfig:
    """Test suite for ExperimentConfig."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = ExperimentConfig(name="test_experiment", device="cpu", seed=42, batch_size=32)

        assert config.name == "test_experiment"
        assert config.device == "cpu"
        assert config.seed == 42
        assert config.batch_size == 32

    def test_default_values(self):
        """Test that defaults are set correctly."""
        config = ExperimentConfig(name="test")
        assert config.experiment_type == "alignment_analysis"
        assert config.seed == 42
        assert config.task_target_permutation == "none"
        assert config.mi_in_proxy_sigma_mode == "median"


class TestPruningConfig:
    """Test suite for PruningConfig."""

    def test_basic_pruning_config(self):
        """Test basic pruning configuration."""
        config = PruningConfig(amount=0.3, structured=True, pruning_mode="low")

        assert config.amount == 0.3
        assert config.structured is True
        assert config.pruning_mode == "low"

    def test_pruning_config_defaults(self):
        """Test pruning config with defaults."""
        config = PruningConfig()

        assert config.amount == 0.5
        assert config.structured is False
        assert config.iterative is False
        assert config.pruning_mode == "low"


class TestGeneralAlignmentConfig:
    """Test suite for GeneralAlignmentConfig."""

    def test_basic_alignment_config(self):
        """Test basic alignment configuration."""
        config = GeneralAlignmentConfig(
            name="test_alignment",
            dataset_name="mnist",
            model_name="mlp",
            training_epochs=5,
            learning_rate=0.01,
            dropout_rates=[0.2, 0.5],
            pruning_amounts=[0.1, 0.3],
        )

        assert config.name == "test_alignment"
        assert config.dataset_name == "mnist"
        assert config.model_name == "mlp"
        assert config.training_epochs == 5
        assert config.dropout_rates == [0.2, 0.5]
        assert config.pruning_amounts == [0.1, 0.3]
