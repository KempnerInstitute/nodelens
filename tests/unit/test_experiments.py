"""
Unit tests for experiment classes.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from alignment.experiments import (
    BaseExperiment,
    CNNConfig,
    EvaluationConfig,
    ExperimentConfig,
    GeneralAlignmentConfig,
    GeneralAlignmentExperiment,
    PruningConfig,
    TrainingConfig,
    create_config_from_dict,
)


class TestExperimentConfig:
    """Test suite for ExperimentConfig."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = ExperimentConfig(name="test_experiment", device="cpu", seed=42, batch_size=32)

        assert config.name == "test_experiment"
        assert config.device == "cpu"
        assert config.seed == 42
        assert config.batch_size == 32


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_basic_training_config(self):
        """Test basic training configuration."""
        config = TrainingConfig(
            training_epochs=10, learning_rate=0.001, batch_size=64, optimizer="adam"
        )

        assert config.training_epochs == 10
        assert config.learning_rate == 0.001
        assert config.batch_size == 64
        assert config.optimizer == "adam"

    def test_training_config_defaults(self):
        """Test training config with defaults."""
        config = TrainingConfig()

        assert config.training_epochs == 10
        assert config.learning_rate == 0.001
        assert config.batch_size == 32


class TestPruningConfig:
    """Test suite for PruningConfig."""

    def test_basic_pruning_config(self):
        """Test basic pruning configuration."""
        config = PruningConfig(
            dropout_rates=[0.1, 0.3, 0.5], pruning_metric="magnitude", dropout_mode="scaled"
        )

        assert config.dropout_rates == [0.1, 0.3, 0.5]
        assert config.pruning_metric == "magnitude"
        assert config.dropout_mode == "scaled"

    def test_pruning_config_defaults(self):
        """Test pruning config with defaults."""
        config = PruningConfig()

        assert len(config.dropout_rates) == 6  # Default is [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        assert config.pruning_strategy == "low"


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


class TestGeneralAlignmentExperiment:
    """Test suite for GeneralAlignmentExperiment."""

    def test_initialization(self):
        """Test experiment initialization."""
        config = GeneralAlignmentConfig(
            name="test_experiment",
            dataset_name="mnist",
            model_name="mlp",
            training_epochs=2,
            batch_size=16,
            device="cpu",
            seed=42,
        )
        
        exp = GeneralAlignmentExperiment(config=config)

        assert exp.config.name == "test_experiment"
        assert exp.config.model_name == "mlp"
        assert exp.config.device == "cpu"
