"""
Unit tests for HookManager.
"""

import pytest
import torch
import torch.nn as nn

from alignment.models.hooks import HookManager, PersistentHookManager


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestHookManager:
    """Tests for HookManager class."""

    def test_initialization(self):
        """Test HookManager initialization."""
        mgr = HookManager()
        assert len(mgr.hooks) == 0
        assert len(mgr.cache) == 0

    def test_temporary_hooks_capture_outputs(self):
        """Test that temporary hooks capture outputs correctly."""
        model = SimpleModel()
        mgr = HookManager()

        input_tensor = torch.randn(2, 3, 10, 10)

        with mgr.temporary_hooks(model, ["conv1", "conv2"], track_outputs=True) as cache:
            model(input_tensor)

            # Check outputs were captured
            assert "conv1_output" in cache
            assert "conv2_output" in cache

            # Check shapes
            assert cache["conv1_output"].shape[0] == 2  # batch size
            assert cache["conv1_output"].shape[1] == 16  # out channels
            assert cache["conv2_output"].shape[1] == 32

        # Check hooks were cleaned up
        assert len(mgr.hooks) == 0
        assert len(mgr.cache) == 0

    def test_temporary_hooks_capture_inputs(self):
        """Test that temporary hooks capture inputs correctly."""
        model = SimpleModel()
        mgr = HookManager()

        input_tensor = torch.randn(2, 3, 10, 10)

        with mgr.temporary_hooks(model, ["conv1", "fc"], track_inputs=True) as cache:
            model(input_tensor)

            # Check inputs were captured
            assert "conv1_input" in cache
            assert "fc_input" in cache

            # Check shapes
            assert cache["conv1_input"].shape == (2, 3, 10, 10)

    def test_temporary_hooks_cleanup_on_exception(self):
        """Test that hooks are cleaned up even when exception occurs."""
        model = SimpleModel()
        mgr = HookManager()

        input_tensor = torch.randn(2, 3, 10, 10)

        with pytest.raises(RuntimeError):
            with mgr.temporary_hooks(model, ["conv1"]):
                model(input_tensor)
                raise RuntimeError("Test exception")

        # Hooks should still be cleaned up
        assert len(mgr.hooks) == 0

    def test_manual_cleanup(self):
        """Test manual cleanup method."""
        model = SimpleModel()
        mgr = HookManager()

        # Register some hooks manually
        def dummy_hook(mod, inp, out):
            pass

        mgr.register_forward_hook(model.conv1, dummy_hook)
        mgr.register_forward_hook(model.conv2, dummy_hook)

        assert len(mgr.hooks) == 2

        # Cleanup
        mgr.cleanup()

        assert len(mgr.hooks) == 0
        assert len(mgr.cache) == 0

    def test_context_manager_protocol(self):
        """Test HookManager as context manager."""
        mgr = HookManager()

        with mgr:
            # Do something
            mgr.cache["test"] = torch.tensor([1, 2, 3])

        # Cache should be cleared
        assert len(mgr.cache) == 0


class TestPersistentHookManager:
    """Tests for PersistentHookManager class."""

    def test_persistent_hooks_across_forward_passes(self):
        """Test that persistent hooks work across multiple forwards."""
        model = SimpleModel()
        mgr = PersistentHookManager(auto_clear_cache=False)

        # Register hooks
        mgr.register_persistent_hooks(model, ["conv1", "conv2"], track_outputs=True)

        # First forward
        input1 = torch.randn(2, 3, 10, 10)
        _ = model(input1)

        activations1 = mgr.get_cached_activations(clear_after=False)
        assert "conv1_output" in activations1

        # Second forward (activations should persist)
        input2 = torch.randn(2, 3, 10, 10)
        _ = model(input2)

        activations2 = mgr.get_cached_activations(clear_after=False)
        assert "conv1_output" in activations2

        # Should be from second forward (updated)
        assert not torch.allclose(activations1["conv1_output"], activations2["conv1_output"])

        # Cleanup
        mgr.cleanup()
        assert len(mgr.hooks) == 0

    def test_auto_clear_cache(self):
        """Test auto_clear_cache functionality."""
        model = SimpleModel()
        mgr = PersistentHookManager(auto_clear_cache=True)

        mgr.register_persistent_hooks(model, ["conv1"], track_outputs=True)

        # First forward
        input1 = torch.randn(2, 3, 10, 10)
        _ = model(input1)

        # Cache should have activations
        assert "conv1_output" in mgr.cache

        # Cleanup
        mgr.cleanup()


def test_hook_registration_logging(caplog):
    """Test that hook registration is logged."""
    model = SimpleModel()
    mgr = HookManager()

    def dummy_hook(mod, inp, out):
        pass

    with caplog.at_level("DEBUG"):
        mgr.register_forward_hook(model.conv1, dummy_hook, name="test_layer")

    assert "test_layer" in caplog.text


def test_multiple_layers_capture():
    """Test capturing multiple layers simultaneously."""
    model = SimpleModel()
    mgr = HookManager()

    input_tensor = torch.randn(4, 3, 10, 10)
    layers = ["conv1", "conv2", "fc"]

    with mgr.temporary_hooks(model, layers, track_inputs=True, track_outputs=True) as cache:
        model(input_tensor)

        # All layers should be captured
        for layer in layers:
            assert f"{layer}_input" in cache
            assert f"{layer}_output" in cache

    # Verify cleanup
    assert len(mgr.hooks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
