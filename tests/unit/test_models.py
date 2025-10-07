"""
Unit tests for neural network models.
"""

import pytest
import torch
import torch.nn as nn

from alignment.models.architectures.standard_models import CNN2P2, MLP


class TestMLP:
    """Test suite for MLP model."""

    def test_construction(self):
        """Test MLP construction."""
        model = MLP(input_dim=784, hidden_dims=[300, 200, 100], output_dim=10, dropout_rate=0.5)

        # Check architecture
        assert hasattr(model, "network")
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 4  # 3 hidden + 1 output

    def test_forward_pass(self):
        """Test forward pass."""
        model = MLP(input_dim=100, hidden_dims=[50], output_dim=10)
        x = torch.randn(32, 100)
        output = model(x)

        assert output.shape == (32, 10)
        assert torch.all(torch.isfinite(output))

    def test_dropout_behavior(self):
        """Test dropout in train/eval mode."""
        model = MLP(input_dim=100, hidden_dims=[50], output_dim=10, dropout_rate=0.5)
        x = torch.randn(10, 100)

        # Training mode - outputs should vary
        model.train()
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2)

        # Eval mode - outputs should be same
        model.eval()
        out3 = model(x)
        out4 = model(x)
        assert torch.allclose(out3, out4)


class TestCNN2P2:
    """Test suite for CNN2P2 model."""

    def test_construction(self):
        """Test CNN2P2 construction."""
        model = CNN2P2(in_channels=3, conv_channels=[32, 64], output_dim=10, example_input_hw=[32, 32])

        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "fc_layers")

    def test_forward_pass(self):
        """Test forward pass."""
        model = CNN2P2(in_channels=3, output_dim=10, example_input_hw=[32, 32])
        x = torch.randn(16, 3, 32, 32)
        output = model(x)

        assert output.shape == (16, 10)
        assert torch.all(torch.isfinite(output))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(device):
    """Test models work on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = MLP(input_dim=100, hidden_dims=[50], output_dim=10).to(device)
    x = torch.randn(16, 100).to(device)
    output = model(x)

    assert output.device.type == device
