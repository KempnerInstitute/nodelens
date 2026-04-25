"""
Tests for training/evaluation.py and dataops/processing/covariance.py.
"""

import pytest
import torch
import torch.nn as nn

from nodelens.dataops.processing.covariance import CovarianceEstimator, estimate_covariance
from nodelens.training.evaluation import EvaluationManager, evaluate_classification, evaluate_model, evaluate_regression

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


class _TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc(x)


def _make_classification_loader(n=32, in_f=8, n_classes=4, batch_size=8):
    images = torch.randn(n, in_f)
    labels = torch.randint(0, n_classes, (n,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def _make_regression_loader(n=32, in_f=8, batch_size=8):
    x = torch.randn(n, in_f)
    y = torch.randn(n, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# =========================================================================
# evaluate_classification
# =========================================================================


class TestEvaluateClassification:
    def test_returns_loss_and_accuracy(self):
        model = _TinyClassifier()
        loader = _make_classification_loader()
        result = evaluate_classification(model, loader, device="cpu")
        assert "loss" in result
        assert "accuracy" in result

    def test_accuracy_range(self):
        model = _TinyClassifier()
        loader = _make_classification_loader()
        result = evaluate_classification(model, loader, device="cpu")
        assert 0 <= result["accuracy"] <= 100.0

    def test_loss_nonneg(self):
        model = _TinyClassifier()
        loader = _make_classification_loader()
        result = evaluate_classification(model, loader, device="cpu")
        assert result["loss"] >= 0

    def test_custom_criterion(self):
        model = _TinyClassifier()
        loader = _make_classification_loader()
        criterion = nn.CrossEntropyLoss()
        result = evaluate_classification(model, loader, device="cpu", criterion=criterion)
        assert "loss" in result


# =========================================================================
# evaluate_regression
# =========================================================================


class TestEvaluateRegression:
    def test_returns_mse_and_mae(self):
        model = _TinyRegressor()
        loader = _make_regression_loader()
        result = evaluate_regression(model, loader, device="cpu")
        assert "mse" in result
        assert "mae" in result
        assert result["mse"] >= 0
        assert result["mae"] >= 0


# =========================================================================
# evaluate_model (dispatcher)
# =========================================================================


class TestEvaluateModel:
    def test_classification_dispatch(self):
        model = _TinyClassifier()
        loader = _make_classification_loader()
        result = evaluate_model(model, loader, task="classification", device="cpu")
        assert "accuracy" in result

    def test_regression_dispatch(self):
        model = _TinyRegressor()
        loader = _make_regression_loader()
        result = evaluate_model(model, loader, task="regression", device="cpu")
        assert "mse" in result

    def test_unknown_task_raises(self):
        model = _TinyClassifier()
        loader = _make_classification_loader()
        with pytest.raises(ValueError, match="Unknown task"):
            evaluate_model(model, loader, task="unknown", device="cpu")


# =========================================================================
# EvaluationManager
# =========================================================================


class TestEvaluationManager:
    def test_evaluate_and_history(self):
        manager = EvaluationManager(task="classification")
        model = _TinyClassifier()
        loader = _make_classification_loader()
        result = manager.evaluate(model, loader, device="cpu", step=0)
        assert "accuracy" in result
        assert len(manager.get_history()) == 1

    def test_get_best_accuracy(self):
        manager = EvaluationManager(task="classification")
        model = _TinyClassifier()
        loader = _make_classification_loader()
        for i in range(3):
            manager.evaluate(model, loader, device="cpu", step=i)
        best = manager.get_best("accuracy")
        assert "accuracy" in best

    def test_get_best_empty(self):
        manager = EvaluationManager()
        assert manager.get_best() == {}


# =========================================================================
# CovarianceEstimator
# =========================================================================


class TestCovarianceEstimator:
    def test_none_method(self):
        X = torch.randn(50, 4)
        est = CovarianceEstimator(method="none", regularization=0.0)
        cov = est.estimate(X)
        assert cov.shape == (4, 4)

    def test_diagonal_method(self):
        X = torch.randn(50, 4)
        est = CovarianceEstimator(method="diagonal")
        cov = est.estimate(X)
        assert cov.shape == (4, 4)
        # Diagonal elements should be >= regularization
        assert (cov.diag() >= est.regularization).all()

    def test_ledoit_wolf(self):
        X = torch.randn(50, 4)
        est = CovarianceEstimator(method="ledoit_wolf")
        cov = est.estimate(X)
        assert cov.shape == (4, 4)
        # Should be symmetric
        torch.testing.assert_close(cov, cov.T)

    def test_oas_method(self):
        X = torch.randn(50, 4)
        est = CovarianceEstimator(method="oas")
        cov = est.estimate(X)
        assert cov.shape == (4, 4)

    def test_unknown_method_raises(self):
        est = CovarianceEstimator(method="bogus")
        with pytest.raises(ValueError, match="Unknown shrinkage"):
            est.estimate(torch.randn(10, 3))

    def test_1d_input_raises(self):
        est = CovarianceEstimator()
        with pytest.raises(ValueError, match="Expected 2D"):
            est.estimate(torch.randn(10))

    def test_shrinkage_improves_conditioning(self):
        """Shrinkage should improve condition number vs raw sample cov."""
        rng = torch.Generator().manual_seed(42)
        X = torch.randn(10, 20, generator=rng)  # n < d case

        raw_est = CovarianceEstimator(method="none", regularization=1e-8)
        raw_cov = raw_est.estimate(X)
        raw_kappa = CovarianceEstimator.estimate_condition_number(raw_cov)

        lw_est = CovarianceEstimator(method="ledoit_wolf")
        lw_cov = lw_est.estimate(X)
        lw_kappa = CovarianceEstimator.estimate_condition_number(lw_cov)

        assert lw_kappa < raw_kappa

    def test_no_centering(self):
        X = torch.randn(50, 4)
        est = CovarianceEstimator(method="none", regularization=0.0)
        cov = est.estimate(X, center=False)
        # Should still be 4x4 positive semi-definite
        assert cov.shape == (4, 4)

    def test_estimate_condition_number_identity(self):
        identity = torch.eye(4)
        kappa = CovarianceEstimator.estimate_condition_number(identity)
        assert kappa == pytest.approx(1.0, abs=1e-4)

    def test_compare_methods(self):
        X = torch.randn(30, 5)
        results = CovarianceEstimator.compare_methods(X)
        assert "none" in results
        assert "ledoit_wolf" in results
        assert "condition_number" in results["none"]


# =========================================================================
# estimate_covariance convenience function
# =========================================================================


class TestEstimateCovariance:
    def test_default(self):
        X = torch.randn(50, 4)
        cov = estimate_covariance(X)
        assert cov.shape == (4, 4)
