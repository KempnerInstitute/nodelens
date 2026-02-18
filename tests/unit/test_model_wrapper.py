"""
Tests for models/base.py: BaseModelWrapper, and models/hooks.py: HookManager.
"""

import pytest
import torch
import torch.nn as nn

from alignment.models.base import BaseModelWrapper
from alignment.models.hooks import HookManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# =========================================================================
# HookManager
# =========================================================================


class TestHookManager:
    def test_temporary_hooks_captures_activations(self):
        model = _SimpleCNN()
        hm = HookManager()
        x = torch.randn(2, 3, 8, 8)

        with hm.temporary_hooks(model, ["conv1", "fc"], track_inputs=True, track_outputs=True) as acts:
            model(x)

        assert "conv1_output" in acts or "conv1" in acts
        hm.cleanup()

    def test_cleanup_removes_hooks(self):
        model = _SimpleCNN()
        hm = HookManager()
        x = torch.randn(2, 3, 8, 8)

        with hm.temporary_hooks(model, ["conv1"]) as acts:
            model(x)

        hm.cleanup()
        # After cleanup, running forward should not error
        model(x)

    def test_register_forward_hook_and_cleanup(self):
        model = _SimpleCNN()
        hm = HookManager()
        conv1 = dict(model.named_modules())["conv1"]
        hm.register_forward_hook(conv1, lambda mod, inp, out: None, name="conv1")
        assert len(hm.hooks) > 0
        hm.cleanup()
        assert len(hm.hooks) == 0


# =========================================================================
# BaseModelWrapper
# =========================================================================


class TestBaseModelWrapper:
    def test_auto_discover_layers(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model)
        # Should auto-discover conv1, conv2, fc
        assert len(wrapper._tracked_layers) >= 3

    def test_tracked_layers_explicit(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1", "fc"])
        assert set(wrapper._tracked_layers) == {"conv1", "fc"}

    def test_get_layer_weights(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1", "fc"])
        weights = wrapper.get_layer_weights()
        assert "conv1" in weights
        assert "fc" in weights
        # Conv weights should be flattened to 2D
        assert weights["conv1"].ndim == 2

    def test_get_layer_weights_no_flatten(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1"])
        weights = wrapper.get_layer_weights(flatten=False)
        assert weights["conv1"].ndim == 4  # [out, in, kH, kW]

    def test_get_layer_weights_with_bias(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["fc"])
        weights = wrapper.get_layer_weights(include_bias=True)
        assert "fc" in weights
        assert "fc_bias" in weights

    def test_preprocess_activations_flatten(self):
        wrapper = BaseModelWrapper(_SimpleCNN(), tracked_layers=["conv1"])
        acts = {"conv1": torch.randn(2, 8, 4, 4)}
        processed = wrapper.preprocess_activations(acts, mode="flatten")
        assert processed["conv1"].shape == (2, 128)

    def test_preprocess_activations_none(self):
        wrapper = BaseModelWrapper(_SimpleCNN(), tracked_layers=["conv1"])
        acts = {"conv1": torch.randn(2, 8, 4, 4)}
        processed = wrapper.preprocess_activations(acts, mode="none")
        assert processed["conv1"].shape == (2, 8, 4, 4)

    def test_get_layer_info(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1", "fc"])
        info = wrapper.get_layer_info("conv1")
        assert info["type"] == "Conv2d"
        assert "weight_shape" in info
        assert info["in_channels"] == 3
        assert info["out_channels"] == 8

    def test_get_layer_info_linear(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["fc"])
        info = wrapper.get_layer_info("fc")
        assert info["type"] == "Linear"
        assert info["in_features"] == 16
        assert info["out_features"] == 10

    def test_get_layer_info_missing(self):
        wrapper = BaseModelWrapper(_SimpleCNN(), tracked_layers=["conv1"])
        info = wrapper.get_layer_info("nonexistent")
        assert "error" in info

    def test_get_layer(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1"])
        layer = wrapper.get_layer("conv1")
        assert isinstance(layer, nn.Conv2d)

    def test_get_layer_missing(self):
        wrapper = BaseModelWrapper(_SimpleCNN())
        assert wrapper.get_layer("nonexistent") is None

    def test_forward_with_activations(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1", "fc"])
        x = torch.randn(4, 3, 8, 8)
        output, acts = wrapper.forward_with_activations(x)
        assert output.shape == (4, 10)
        assert len(acts) > 0

    def test_capture_activations_safe(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1"])
        x = torch.randn(2, 3, 8, 8)
        acts = wrapper.capture_activations_safe(x)
        assert len(acts) > 0

    def test_apply_structured_dropout(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1"])
        mask = torch.ones(8)
        mask[0] = 0  # prune first filter
        wrapper.apply_structured_dropout({"conv1": mask}, permanent=False)
        # First filter should be zeroed
        assert (model.conv1.weight.data[0] == 0).all()

    def test_restore_weights(self):
        model = _SimpleCNN()
        wrapper = BaseModelWrapper(model, tracked_layers=["conv1"])
        original_w = model.conv1.weight.data.clone()
        mask = torch.ones(8)
        mask[0] = 0
        wrapper.apply_structured_dropout({"conv1": mask}, permanent=False)
        wrapper.restore_weights()
        torch.testing.assert_close(model.conv1.weight.data, original_w)
