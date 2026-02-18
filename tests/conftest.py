"""
Pytest configuration and fixtures.
"""

import random

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_mnist_data():
    """Create sample MNIST-like data for testing."""
    batch_size = 32
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def sample_cifar_data():
    """Create sample CIFAR-like data for testing."""
    batch_size = 32
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def mock_dataloader(sample_mnist_data):
    """Create a mock dataloader for testing."""
    images, labels = sample_mnist_data

    class MockDataLoader:
        def __init__(self):
            self.dataset = list(zip(images, labels))

        def __iter__(self):
            return iter([self.dataset])

        def __len__(self):
            return 1

    return MockDataLoader()


# ---------------------------------------------------------------------------
# Shared fixtures for paper-critical modules
# ---------------------------------------------------------------------------


class _TinyCNN(nn.Module):
    """Minimal 2-conv + linear network for unit tests."""

    def __init__(self, in_channels=3, n_channels=16, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def tiny_cnn():
    """Small 2-conv + linear CNN in eval mode."""
    model = _TinyCNN(in_channels=3, n_channels=16, n_classes=10)
    model.eval()
    return model


@pytest.fixture
def synthetic_cifar_batch():
    """Tiny CIFAR-like batch: [8, 3, 8, 8] images + 10-class labels."""
    images = torch.randn(8, 3, 8, 8)
    labels = torch.randint(0, 10, (8,))
    return images, labels


@pytest.fixture
def synthetic_channel_metrics():
    """Pre-computed RQ / redundancy / synergy for 16 channels."""
    rng = np.random.default_rng(42)
    return {
        "rq": rng.uniform(0.1, 10.0, 16),
        "redundancy": rng.uniform(0.0, 1.0, 16),
        "synergy": rng.uniform(0.0, 1.0, 16),
    }


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "paper_critical: marks tests for paper-critical modules")
