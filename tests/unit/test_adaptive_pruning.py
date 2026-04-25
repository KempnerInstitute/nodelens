"""
Tests for pruning/strategies/adaptive.py: AdaptiveSensitivityPruning.
"""

import pytest
import torch
import torch.nn as nn

from nodelens.pruning.strategies.adaptive import AdaptiveSensitivityPruning, LayerSensitivity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_loader(batch_size=8, n_batches=4):
    """Create a simple data loader for testing."""
    data = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, 3, 8, 8)
        y = torch.randint(0, 4, (batch_size,))
        data.append((x, y))
    return data


# =========================================================================
# LayerSensitivity dataclass
# =========================================================================


class TestLayerSensitivity:
    def test_basic(self):
        ls = LayerSensitivity(name="conv1", sensitivity=0.5, size=100, recommended_amount=0.3)
        assert ls.name == "conv1"
        assert ls.sensitivity == 0.5
        assert ls.size == 100
        assert ls.recommended_amount == 0.3


# =========================================================================
# AdaptiveSensitivityPruning init
# =========================================================================


class TestAdaptiveSensitivityPruningInit:
    def test_defaults(self):
        asp = AdaptiveSensitivityPruning()
        assert asp.target_sparsity == 0.5
        assert asp.sensitivity_method == "perturbation"
        assert asp.min_amount == 0.1
        assert asp.max_amount == 0.9

    def test_custom_params(self):
        asp = AdaptiveSensitivityPruning(
            target_sparsity=0.7,
            sensitivity_method="weight_magnitude",
            min_amount=0.2,
            max_amount=0.8,
        )
        assert asp.target_sparsity == 0.7
        assert asp.sensitivity_method == "weight_magnitude"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown sensitivity_method"):
            AdaptiveSensitivityPruning(sensitivity_method="bogus")

    def test_all_valid_methods(self):
        for method in AdaptiveSensitivityPruning.SENSITIVITY_METHODS:
            asp = AdaptiveSensitivityPruning(sensitivity_method=method)
            assert asp.sensitivity_method == method


# =========================================================================
# measure_layer_sensitivity
# =========================================================================


class TestMeasureLayerSensitivity:
    def test_weight_magnitude(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="weight_magnitude")
        sens = asp.measure_layer_sensitivity(model, "conv1")
        assert sens > 0

    def test_activation_variance_with_cached(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="activation_variance")
        cached = {"conv1": torch.randn(8, 8, 8, 8)}
        sens = asp.measure_layer_sensitivity(model, "conv1", cached_activations=cached)
        assert sens > 0

    def test_activation_variance_no_cached_falls_back(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="activation_variance")
        sens = asp.measure_layer_sensitivity(model, "conv1")
        assert sens > 0  # Falls back to weight_magnitude

    def test_gradient_method(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="gradient")
        loader = _make_loader()
        sens = asp.measure_layer_sensitivity(model, "conv1", data_loader=loader)
        assert sens >= 0

    def test_gradient_no_loader_falls_back(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="gradient")
        sens = asp.measure_layer_sensitivity(model, "conv1")
        assert sens > 0  # Falls back to weight_magnitude

    def test_fisher_method(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="fisher")
        loader = _make_loader()
        sens = asp.measure_layer_sensitivity(model, "conv1", data_loader=loader)
        assert sens >= 0

    def test_perturbation_method(self):
        model = _TinyModel()
        model.eval()
        asp = AdaptiveSensitivityPruning(
            sensitivity_method="perturbation",
            num_trials=2,
        )

        def eval_fn(m):
            with torch.no_grad():
                x = torch.randn(4, 3, 8, 8)
                out = m(x)
                return (out.argmax(1) == torch.randint(0, 4, (4,))).float().mean().item()

        sens = asp.measure_layer_sensitivity(model, "conv1", eval_fn=eval_fn)
        assert sens >= 0

    def test_perturbation_no_eval_fn_raises(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="perturbation")
        with pytest.raises(ValueError, match="requires eval_fn"):
            asp.measure_layer_sensitivity(model, "conv1")


# =========================================================================
# compute_all_sensitivities
# =========================================================================


class TestComputeAllSensitivities:
    def test_weight_magnitude_all_layers(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(
            target_sparsity=0.5,
            sensitivity_method="weight_magnitude",
        )
        result = asp.compute_all_sensitivities(model, ["conv1", "conv2", "fc"])
        assert "conv1" in result
        assert "conv2" in result
        assert "fc" in result
        for ls in result.values():
            assert 0.0 <= ls.recommended_amount <= 1.0

    def test_perturbation_requires_eval_fn(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="perturbation")
        with pytest.raises(ValueError, match="requires eval_fn"):
            asp.compute_all_sensitivities(model, ["conv1"])

    def test_gradient_requires_data_loader(self):
        model = _TinyModel()
        asp = AdaptiveSensitivityPruning(sensitivity_method="gradient")
        with pytest.raises(ValueError, match="requires data_loader"):
            asp.compute_all_sensitivities(model, ["conv1"])


# =========================================================================
# _compute_adaptive_amounts
# =========================================================================


class TestComputeAdaptiveAmounts:
    def test_inverse_relationship(self):
        """More sensitive layers should get lower pruning amounts."""
        asp = AdaptiveSensitivityPruning(
            target_sparsity=0.5,
            min_amount=0.1,
            max_amount=0.9,
        )
        sensitivities = {
            "low_sens": LayerSensitivity("low_sens", sensitivity=0.1, size=100, recommended_amount=0.0),
            "high_sens": LayerSensitivity("high_sens", sensitivity=0.9, size=100, recommended_amount=0.0),
        }
        result = asp._compute_adaptive_amounts(sensitivities)
        # High sensitivity should get lower amount
        assert result["high_sens"].recommended_amount <= result["low_sens"].recommended_amount

    def test_empty_sensitivities(self):
        asp = AdaptiveSensitivityPruning()
        result = asp._compute_adaptive_amounts({})
        assert result == {}

    def test_amounts_within_bounds(self):
        asp = AdaptiveSensitivityPruning(min_amount=0.2, max_amount=0.8)
        sensitivities = {f"layer{i}": LayerSensitivity(f"layer{i}", sensitivity=i * 0.1, size=100, recommended_amount=0.0) for i in range(5)}
        result = asp._compute_adaptive_amounts(sensitivities)
        for ls in result.values():
            assert ls.recommended_amount >= 0.2
            assert ls.recommended_amount <= 0.8


# =========================================================================
# print_sensitivity_report
# =========================================================================


class TestPrintSensitivityReport:
    def test_no_data(self, capsys):
        asp = AdaptiveSensitivityPruning()
        asp.print_sensitivity_report()
        out = capsys.readouterr().out
        assert "No sensitivity data" in out

    def test_with_data(self, capsys):
        asp = AdaptiveSensitivityPruning(target_sparsity=0.5)
        asp.layer_sensitivities = {
            "conv1": LayerSensitivity("conv1", sensitivity=0.5, size=100, recommended_amount=0.6),
            "fc": LayerSensitivity("fc", sensitivity=0.8, size=50, recommended_amount=0.3),
        }
        asp.print_sensitivity_report()
        out = capsys.readouterr().out
        assert "conv1" in out
        assert "fc" in out
        assert "Overall sparsity" in out
