"""
Tests for configs/config_validator.py (validate_config, validate_experiment_config,
check_compatibility).
"""

from alignment.configs.config_validator import check_compatibility, validate_config, validate_experiment_config

# =========================================================================
# validate_config
# =========================================================================


class TestValidateConfig:
    def test_valid_minimal(self):
        errors = validate_config({"name": "test"})
        assert errors == []

    def test_missing_name(self):
        errors = validate_config({})
        assert any("name" in e for e in errors)

    def test_invalid_device(self):
        errors = validate_config({"name": "t", "device": "tpu"})
        assert any("device" in e.lower() for e in errors)

    def test_valid_cpu(self):
        errors = validate_config({"name": "t", "device": "cpu"})
        device_errors = [e for e in errors if "device" in e.lower()]
        assert device_errors == []

    def test_valid_cuda(self):
        errors = validate_config({"name": "t", "device": "cuda:0"})
        device_errors = [e for e in errors if "device" in e.lower()]
        assert device_errors == []

    def test_numeric_out_of_range(self):
        errors = validate_config({"name": "t", "batch_size": -1})
        assert any("batch_size" in e for e in errors)

    def test_numeric_valid(self):
        errors = validate_config({"name": "t", "batch_size": 64})
        batch_errors = [e for e in errors if "batch_size" in e]
        assert batch_errors == []

    def test_numeric_wrong_type(self):
        errors = validate_config({"name": "t", "learning_rate": "fast"})
        assert any("learning_rate" in e for e in errors)

    def test_dropout_fractions_valid(self):
        errors = validate_config({"name": "t", "dropout_fractions": [0.0, 0.5, 1.0]})
        df_errors = [e for e in errors if "dropout" in e.lower()]
        assert df_errors == []

    def test_dropout_fractions_invalid(self):
        errors = validate_config({"name": "t", "dropout_fractions": [1.5]})
        assert any("dropout" in e.lower() for e in errors)

    def test_dropout_fractions_not_list(self):
        errors = validate_config({"name": "t", "dropout_fractions": 0.5})
        assert any("dropout" in e.lower() for e in errors)

    def test_metrics_not_list(self):
        errors = validate_config({"name": "t", "metrics": "rq"})
        assert any("metrics" in e.lower() for e in errors)


# =========================================================================
# validate_experiment_config
# =========================================================================


class TestValidateExperimentConfig:
    def test_progressive_dropout_needs_fractions(self):
        errors = validate_experiment_config({"name": "t"}, "progressive_dropout")
        assert any("dropout_fractions" in e for e in errors)

    def test_progressive_dropout_valid(self):
        errors = validate_experiment_config(
            {"name": "t", "dropout_fractions": [0.1, 0.5]},
            "progressive_dropout",
        )
        assert not any("dropout_fractions" in e for e in errors)

    def test_eigenvector_components(self):
        errors = validate_experiment_config(
            {"name": "t", "num_components": 0},
            "eigenvector",
        )
        assert any("num_components" in e for e in errors)

    def test_layer_isolated_needs_percentages(self):
        errors = validate_experiment_config({"name": "t"}, "layer_isolated")
        assert any("pruning_percentages" in e for e in errors)


# =========================================================================
# check_compatibility
# =========================================================================


class TestCheckCompatibility:
    def test_cnn2p2_mnist_wrong_channels(self):
        warnings = check_compatibility(
            {
                "model_name": "cnn2p2",
                "dataset_name": "mnist",
                "model_config": {"in_channels": 3},
            }
        )
        assert any("in_channels" in w for w in warnings)

    def test_large_batch_cpu(self):
        warnings = check_compatibility(
            {
                "batch_size": 1024,
                "device": "cpu",
            }
        )
        assert any("batch" in w.lower() for w in warnings)

    def test_no_warnings_for_normal_config(self):
        warnings = check_compatibility(
            {
                "model_name": "resnet18",
                "dataset_name": "cifar10",
                "batch_size": 64,
                "device": "cuda:0",
            }
        )
        assert warnings == []
