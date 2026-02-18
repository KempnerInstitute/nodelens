"""
Tests for training/base.py: BaseTrainer and TrainingConfig.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alignment.training.base import TrainingConfig, BaseTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_loaders(n_train=64, n_val=32, batch_size=16):
    x_train = torch.randn(n_train, 8)
    y_train = torch.randint(0, 4, (n_train,))
    x_val = torch.randn(n_val, 8)
    y_val = torch.randint(0, 4, (n_val,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


# =========================================================================
# TrainingConfig
# =========================================================================


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 10
        assert cfg.learning_rate == 0.001
        assert cfg.optimizer == "adam"
        assert cfg.device == "cuda"

    def test_custom(self):
        cfg = TrainingConfig(epochs=5, learning_rate=0.01, optimizer="sgd")
        assert cfg.epochs == 5
        assert cfg.optimizer == "sgd"


# =========================================================================
# BaseTrainer init
# =========================================================================


class TestBaseTrainerInit:
    def test_default_config(self):
        model = _TinyModel()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu"))
        assert trainer.config.epochs == 10
        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)
        assert trainer.current_epoch == 0

    def test_custom_loss_fn(self):
        model = _TinyModel()
        loss_fn = nn.MSELoss()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu"), loss_fn=loss_fn)
        assert isinstance(trainer.loss_fn, nn.MSELoss)

    def test_optimizer_adam(self):
        model = _TinyModel()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu", optimizer="adam"))
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_optimizer_sgd(self):
        model = _TinyModel()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu", optimizer="sgd"))
        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_optimizer_adamw(self):
        model = _TinyModel()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu", optimizer="adamw"))
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_unknown_optimizer_raises(self):
        model = _TinyModel()
        with pytest.raises(ValueError, match="Unknown optimizer"):
            BaseTrainer(model, config=TrainingConfig(device="cpu", optimizer="bogus"))


# =========================================================================
# Scheduler creation
# =========================================================================


class TestSchedulerCreation:
    def test_no_scheduler(self):
        model = _TinyModel()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu"))
        assert trainer.scheduler is None

    def test_cosine_scheduler(self):
        model = _TinyModel()
        trainer = BaseTrainer(
            model,
            config=TrainingConfig(device="cpu", scheduler="cosine"),
        )
        assert trainer.scheduler is not None

    def test_plateau_scheduler(self):
        model = _TinyModel()
        trainer = BaseTrainer(
            model,
            config=TrainingConfig(device="cpu", scheduler="plateau"),
        )
        assert trainer.scheduler is not None

    def test_step_scheduler(self):
        model = _TinyModel()
        trainer = BaseTrainer(
            model,
            config=TrainingConfig(device="cpu", scheduler="step", scheduler_kwargs={"step_size": 5}),
        )
        assert trainer.scheduler is not None

    def test_unknown_scheduler_raises(self):
        model = _TinyModel()
        with pytest.raises(ValueError, match="Unknown scheduler"):
            BaseTrainer(model, config=TrainingConfig(device="cpu", scheduler="bogus"))


# =========================================================================
# Training
# =========================================================================


class TestTraining:
    def test_train_one_epoch(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", epochs=1)
        trainer = BaseTrainer(model, config=config)
        train_loader, _ = _make_loaders()
        history = trainer.train(train_loader)
        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_train_with_validation(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", epochs=2, eval_interval=1)
        trainer = BaseTrainer(model, config=config)
        train_loader, val_loader = _make_loaders()
        history = trainer.train(train_loader, val_loader=val_loader)
        assert len(history["val_loss"]) == 2
        assert all(v >= 0 for v in history["val_loss"])

    def test_train_with_metric_fn(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", epochs=1)
        trainer = BaseTrainer(model, config=config)
        train_loader, _ = _make_loaders()

        def accuracy_fn(outputs, targets):
            preds = outputs.argmax(dim=1)
            return {"accuracy": (preds == targets).float().mean().item()}

        history = trainer.train(train_loader, metric_fn=accuracy_fn)
        assert len(history["train_metrics"]) == 1
        assert "accuracy" in history["train_metrics"][0]

    def test_train_with_callbacks(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", epochs=1)
        callback_calls = []

        def my_callback(trainer, epoch):
            callback_calls.append(epoch)

        trainer = BaseTrainer(model, config=config, callbacks=[my_callback])
        train_loader, _ = _make_loaders()
        trainer.train(train_loader)
        assert len(callback_calls) == 1

    def test_gradient_clipping(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", epochs=1, gradient_clip_val=1.0)
        trainer = BaseTrainer(model, config=config)
        train_loader, _ = _make_loaders()
        history = trainer.train(train_loader)
        assert len(history["train_loss"]) == 1

    def test_learning_rate_tracked(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", epochs=2)
        trainer = BaseTrainer(model, config=config)
        train_loader, _ = _make_loaders()
        history = trainer.train(train_loader)
        assert len(history["learning_rates"]) == 2

    def test_epoch_times_tracked(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", epochs=1)
        trainer = BaseTrainer(model, config=config)
        train_loader, _ = _make_loaders()
        history = trainer.train(train_loader)
        assert len(history["epoch_times"]) == 1
        assert history["epoch_times"][0] > 0


# =========================================================================
# Early stopping
# =========================================================================


class TestEarlyStopping:
    def test_no_early_stopping(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu")
        trainer = BaseTrainer(model, config=config)
        assert trainer._should_stop_early(1.0) is False

    def test_improving_loss(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", early_stopping_patience=3)
        trainer = BaseTrainer(model, config=config)
        assert trainer._should_stop_early(0.5) is False  # Improves
        assert trainer._should_stop_early(0.3) is False  # Improves

    def test_plateau_triggers_stop(self):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", early_stopping_patience=2)
        trainer = BaseTrainer(model, config=config)
        trainer._should_stop_early(0.5)  # Best
        trainer._should_stop_early(0.6)  # Worse (patience=1)
        assert trainer._should_stop_early(0.7) is True  # Worse (patience=2) -> stop


# =========================================================================
# Helper methods
# =========================================================================


class TestHelperMethods:
    def test_average_metrics_empty(self):
        model = _TinyModel()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu"))
        assert trainer._average_metrics([]) == {}

    def test_average_metrics(self):
        model = _TinyModel()
        trainer = BaseTrainer(model, config=TrainingConfig(device="cpu"))
        metrics = [{"acc": 0.8, "loss": 0.3}, {"acc": 0.9, "loss": 0.2}]
        avg = trainer._average_metrics(metrics)
        assert avg["acc"] == pytest.approx(0.85)
        assert avg["loss"] == pytest.approx(0.25)

    def test_save_checkpoint(self, tmp_path):
        model = _TinyModel()
        config = TrainingConfig(device="cpu", checkpoint_dir=str(tmp_path))
        trainer = BaseTrainer(model, config=config)
        trainer._save_checkpoint(0)
        assert (tmp_path / "checkpoint_epoch_0.pt").exists()
