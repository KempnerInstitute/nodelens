"""
Pytest configuration and fixtures.
"""

import random

import numpy as np
import pytest
import torch


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


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
