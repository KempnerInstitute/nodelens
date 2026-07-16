"""
Tests for configs/config_loader.py: format detection, conversion, load/save.
"""

import json

import pytest
import yaml

from nodelens.configs.config_loader import (
    METRIC_ORIGINAL_TO_UNIFIED,
    METRIC_UNIFIED_TO_ORIGINAL,
    _convert_unified_to_original,
    _is_unified_format,
    _map_nested_to_flat_config,
    load_config,
    save_config,
)

# =========================================================================
# _is_unified_format
# =========================================================================


class TestIsUnifiedFormat:
    def test_original_format(self):
        cfg = {"metrics": {"enabled": ["rayleigh_quotient"]}}
        assert _is_unified_format(cfg) is False

    def test_unified_format_with_enabled_key(self):
        cfg = {"metrics": {"rayleigh_quotient": {"enabled": True}}}
        assert _is_unified_format(cfg) is True

    def test_unified_format_with_extra(self):
        cfg = {"extra": {"analysis": {}}}
        assert _is_unified_format(cfg) is True

    def test_no_metrics(self):
        cfg = {"name": "test"}
        assert _is_unified_format(cfg) is False

    def test_metrics_not_dict(self):
        cfg = {"metrics": ["rq", "red"]}
        assert _is_unified_format(cfg) is False


# =========================================================================
# _convert_unified_to_original
# =========================================================================


class TestConvertUnifiedToOriginal:
    def test_experiment_section(self):
        unified = {
            "experiment": {"name": "my_exp", "type": "cluster_analysis", "seed": 123, "device": "cpu"},
        }
        result = _convert_unified_to_original(unified)
        assert result["experiment"]["name"] == "my_exp"
        assert result["seed"] == 123
        assert result["device"] == "cpu"

    def test_model_section(self):
        unified = {"model": {"name": "resnet18", "pretrained": True}}
        result = _convert_unified_to_original(unified)
        assert result["model"]["name"] == "resnet18"

    def test_dataset_section(self):
        unified = {"dataset": {"name": "cifar100", "batch_size": 64}}
        result = _convert_unified_to_original(unified)
        assert result["dataset"]["name"] == "cifar100"
        assert result["dataset"]["batch_size"] == 64

    def test_metrics_conversion(self):
        unified = {
            "metrics": {
                "rayleigh_quotient": {"enabled": True},
                "redundancy": {"enabled": True},
            }
        }
        result = _convert_unified_to_original(unified)
        enabled = result["metrics"]["enabled"]
        assert "rayleigh_quotient" in enabled

    def test_metrics_disabled_excluded(self):
        unified = {
            "metrics": {
                "rayleigh_quotient": {"enabled": False},
                "redundancy": {"enabled": True},
            }
        }
        result = _convert_unified_to_original(unified)
        enabled = result["metrics"]["enabled"]
        assert "rayleigh_quotient" not in enabled

    def test_pruning_section(self):
        unified = {
            "pruning": {
                "enabled": True,
                "ratios": [0.3, 0.5, 0.7],
                "methods": ["magnitude", "rayleigh_quotient"],
                "distribution": "uniform",
            }
        }
        result = _convert_unified_to_original(unified)
        assert result["pruning"]["enabled"] is True
        assert result["pruning"]["sparsity_levels"] == [0.3, 0.5, 0.7]
        assert "magnitude" in result["pruning"]["methods"]

    def test_halo_analysis_section(self):
        unified = {"halo_analysis": {"enabled": True, "n_permutations": 50}}
        result = _convert_unified_to_original(unified)
        assert result["do_halo_analysis"] is True

    def test_clustering_section(self):
        unified = {"clustering": {"n_clusters": 4}}
        result = _convert_unified_to_original(unified)
        assert result["clustering"]["n_clusters"] == 4

    def test_evaluation_perplexity(self):
        unified = {"evaluation": {"enabled": True, "perplexity_enabled": True}}
        result = _convert_unified_to_original(unified)
        assert result["do_perplexity_computation"] is True

    def test_output_section(self):
        unified = {"output": {"dir": "/tmp/results"}}
        result = _convert_unified_to_original(unified)
        assert result["results_path"] == "/tmp/results"

    def test_extra_section(self):
        unified = {"extra": {"analysis": {"enabled": True}, "pretrain_epochs": 5}}
        result = _convert_unified_to_original(unified)
        assert result["analysis"]["enabled"] is True
        assert result["pretrain_epochs"] == 5

    def test_training_passthrough(self):
        unified = {"training": {"epochs": 10, "lr": 0.001}}
        result = _convert_unified_to_original(unified)
        assert result["training"]["epochs"] == 10

    def test_learning_rule_passthrough(self):
        unified = {"learning_rule": {"method": "bp_tard", "lambda": 0.001}}
        result = _convert_unified_to_original(unified)
        assert result["learning_rule"]["method"] == "bp_tard"

    def test_scar_metrics(self):
        unified = {
            "metrics": {
                "scar": {"enabled": True, "num_samples": 32, "max_length": 256},
            }
        }
        result = _convert_unified_to_original(unified)
        assert result["do_scar_metrics"] is True
        assert result["scar_num_samples"] == 32

    def test_calibration_sample_count_survives_metrics_rebuild(self):
        unified = {
            "calibration": {"num_samples": 128, "max_length": 512},
            "metrics": {
                "scar": {"enabled": True, "num_samples": 64},
            },
        }
        result = _convert_unified_to_original(unified)
        assert result["metrics"]["num_samples"] == 128


# =========================================================================
# _map_nested_to_flat_config
# =========================================================================


class TestMapNestedToFlatConfig:
    def test_experiment_name(self):
        result = _map_nested_to_flat_config({"experiment": {"name": "my_exp"}})
        assert result["name"] == "my_exp"

    def test_flat_experiment_name(self):
        result = _map_nested_to_flat_config({"experiment_name": "flat_exp"})
        assert result["name"] == "flat_exp"

    def test_dataset_mapping(self):
        result = _map_nested_to_flat_config(
            {
                "dataset": {"name": "CIFAR10", "batch_size": 64},
            }
        )
        assert result["dataset_name"] == "cifar10"  # lowercased
        assert result["batch_size"] == 64

    def test_model_mapping(self):
        result = _map_nested_to_flat_config(
            {
                "model": {"name": "resnet18", "pretrained": True},
            }
        )
        assert result["model_name"] == "resnet18"
        assert result["pretrained"] is True

    def test_flat_metrics_list(self):
        result = _map_nested_to_flat_config({"metrics": ["rq", "red"]})
        assert result["metrics"] == ["rq", "red"]

    def test_metrics_dict_with_enabled(self):
        result = _map_nested_to_flat_config(
            {
                "metrics": {"enabled": ["rq"], "rayleigh_quotient": {"definition": "standard"}},
            }
        )
        assert result["metrics"] == ["rq"]
        assert result["rq_definition"] == "standard"

    def test_clustering_ablation(self):
        result = _map_nested_to_flat_config(
            {
                "clustering": {"ablation": {"enabled": True, "modes": ["RQ_Red", "RQ_Syn"]}},
            }
        )
        assert result["run_metric_ablation"] is True
        assert result["metric_ablations"] == ["RQ_Red", "RQ_Syn"]

    def test_halo_permutation(self):
        result = _map_nested_to_flat_config(
            {
                "halo_analysis": {"permutation_baseline": {"enabled": True, "n_permutations": 100}},
            }
        )
        assert result["run_permutation_baseline"] is True
        assert result["n_permutations"] == 100

    def test_learning_rule_mapping(self):
        result = _map_nested_to_flat_config(
            {
                "learning_rule": {
                    "method": "bp_tard",
                    "lambda": 0.001,
                    "warmup_epochs": 5,
                    "max_layers": 2,
                }
            }
        )
        assert result["learning_rule_method"] == "bp_tard"
        assert result["learning_rule_lambda"] == 0.001
        assert result["learning_rule_warmup_epochs"] == 5
        assert result["learning_rule_max_layers"] == 2


# =========================================================================
# load_config / save_config
# =========================================================================


class TestLoadSaveConfig:
    def test_load_yaml(self, tmp_path):
        config = {
            "experiment": {"name": "test_exp", "type": "alignment_analysis"},
            "model": {"name": "cnn2p2"},
            "dataset": {"name": "cifar10"},
        }
        fpath = tmp_path / "test.yaml"
        fpath.write_text(yaml.dump(config))
        loaded = load_config(fpath)
        assert loaded.name == "test_exp"

    def test_load_json(self, tmp_path):
        config = {
            "experiment": {"name": "json_exp"},
            "model": {"name": "mlp"},
            "dataset": {"name": "mnist"},
        }
        fpath = tmp_path / "test.json"
        fpath.write_text(json.dumps(config))
        loaded = load_config(fpath)
        assert loaded.name == "json_exp"

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_unsupported_format(self, tmp_path):
        fpath = tmp_path / "test.toml"
        fpath.write_text("[experiment]\nname='test'")
        with pytest.raises(ValueError, match="Unsupported"):
            load_config(fpath)

    def test_save_yaml(self, tmp_path):
        from nodelens.experiments.base import ExperimentConfig

        config = ExperimentConfig(name="save_test")
        fpath = tmp_path / "saved.yaml"
        save_config(config, fpath, format="yaml")
        assert fpath.exists()
        loaded = yaml.safe_load(fpath.read_text())
        assert loaded["name"] == "save_test"

    def test_save_json(self, tmp_path):
        from nodelens.experiments.base import ExperimentConfig

        config = ExperimentConfig(name="save_json")
        fpath = tmp_path / "saved.json"
        save_config(config, fpath, format="json")
        loaded = json.loads(fpath.read_text())
        assert loaded["name"] == "save_json"

    def test_save_unsupported_format(self, tmp_path):
        from nodelens.experiments.base import ExperimentConfig

        config = ExperimentConfig(name="test")
        with pytest.raises(ValueError, match="Unsupported"):
            save_config(config, tmp_path / "test.toml", format="toml")

    def test_unified_format_auto_detected(self, tmp_path):
        config = {
            "experiment": {"name": "unified_test"},
            "model": {"name": "resnet18"},
            "dataset": {"name": "cifar10"},
            "metrics": {"rayleigh_quotient": {"enabled": True}},
        }
        fpath = tmp_path / "unified.yaml"
        fpath.write_text(yaml.dump(config))
        loaded = load_config(fpath)
        assert loaded.name == "unified_test"


# =========================================================================
# Metric name mappings
# =========================================================================


class TestMetricMappings:
    def test_unified_to_original_has_rq(self):
        assert "rayleigh_quotient" in METRIC_UNIFIED_TO_ORIGINAL

    def test_original_to_unified_has_rq(self):
        assert "rayleigh_quotient" in METRIC_ORIGINAL_TO_UNIFIED
