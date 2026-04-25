"""
Unit tests for checkpoint utilities.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from nodelens.infrastructure.storage.checkpoint import load_checkpoint, save_checkpoint, save_model_for_inference


class DummyModelWithHooks(nn.Module):
    """Dummy model that registers hooks."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

        # Register a hook
        self.hook_called = False
        self.linear1.register_forward_hook(self._dummy_hook)

    def _dummy_hook(self, module, input, output):
        self.hook_called = True
        return output

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""

    def test_basic_save_load(self):
        """Test basic checkpoint saving and loading."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")

            # Save checkpoint
            save_checkpoint(model=model, optimizer=optimizer, epoch=5, filepath=filepath, additional_state={"best_loss": 0.1})

            # Check file exists
            assert os.path.exists(filepath)

            # Load checkpoint
            checkpoint = load_checkpoint(filepath)

            assert checkpoint["epoch"] == 5
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert checkpoint.get("best_loss") == 0.1

    def test_model_state_loading(self):
        """Test loading model state from checkpoint."""
        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")

            # Save original weights
            original_weight = model.weight.clone()

            # Save checkpoint
            save_checkpoint(model, None, 1, filepath)

            # Modify model weights
            model.weight.data.fill_(0)
            assert not torch.equal(model.weight, original_weight)

            # Load checkpoint
            load_checkpoint(filepath, model=model)

            # Check weights restored
            assert torch.equal(model.weight, original_weight)

    def test_optimizer_state_loading(self):
        """Test loading optimizer state from checkpoint."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Do some optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(torch.randn(3, 10)).sum()
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")

            # Save checkpoint
            save_checkpoint(model, optimizer, 5, filepath)

            # Create new optimizer
            new_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Load checkpoint
            load_checkpoint(filepath, optimizer=new_optimizer)

            # Check state loaded
            assert len(new_optimizer.state) > 0

    def test_save_with_hooks_warning(self):
        """Test saving model with hooks generates warning."""
        model = DummyModelWithHooks()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")

            # Save with hooks should work but may warn
            save_checkpoint(model, None, 1, filepath, save_hooks=True)
            assert os.path.exists(filepath)

    def test_file_not_found_error(self):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("non_existent_file.pt")


class TestInferenceModelSaving:
    """Test saving models for inference."""

    def test_save_for_inference_basic(self):
        """Test basic inference model saving."""
        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model_inference.pt")

            save_model_for_inference(model, filepath)

            # Check file exists
            assert os.path.exists(filepath)

            # Load and check
            state_dict = torch.load(filepath)
            assert "weight" in state_dict
            assert "bias" in state_dict

    def test_save_for_inference_remove_hooks(self):
        """Test removing hooks when saving for inference."""
        model = DummyModelWithHooks()

        # Verify hook works
        x = torch.randn(1, 10)
        model(x)
        assert model.hook_called

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model_inference.pt")

            # Save with hook removal
            save_model_for_inference(model, filepath, remove_hooks=True)

            # Hooks should be restored after saving
            model.hook_called = False
            model(x)
            assert model.hook_called  # Hook should still work

            # Load saved model into new instance
            new_model = DummyModelWithHooks()
            # Remove all hooks from all modules
            new_model._forward_hooks.clear()
            new_model.linear1._forward_hooks.clear()
            new_model.linear2._forward_hooks.clear()
            new_model.load_state_dict(torch.load(filepath))

            # New model should work without hooks
            new_model.hook_called = False
            new_model(x)
            assert not new_model.hook_called  # No hook

    def test_save_for_inference_keep_hooks(self):
        """Test saving without removing hooks."""
        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model_inference.pt")

            save_model_for_inference(model, filepath, remove_hooks=False)

            assert os.path.exists(filepath)


class TestEdgeCases:
    """Test edge cases for checkpoint utilities."""

    def test_empty_model(self):
        """Test with model that has no parameters."""

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        model = EmptyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty_model.pt")

            save_checkpoint(model, None, 1, filepath)
            checkpoint = load_checkpoint(filepath)

            assert checkpoint["epoch"] == 1
            assert checkpoint["model_state_dict"] == {}

    def test_cuda_to_cpu_loading(self):
        """Test loading CUDA checkpoint on CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = nn.Linear(10, 5).cuda()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "cuda_checkpoint.pt")

            save_checkpoint(model, None, 1, filepath)

            # Load on CPU
            cpu_model = nn.Linear(10, 5)
            checkpoint = load_checkpoint(filepath, model=cpu_model, map_location="cpu")

            assert checkpoint is not None
            assert next(cpu_model.parameters()).device.type == "cpu"

    def test_strict_loading_fallback(self):
        """Test fallback to non-strict loading."""
        # Create models with different architectures
        model1 = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

        model2 = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))  # Different output size

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")

            save_checkpoint(model1, None, 1, filepath)

            # Loading into incompatible model should use strict=False
            load_checkpoint(filepath, model=model2)

            # First layer should be loaded
            assert torch.equal(model1[0].weight, model2[0].weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
