"""
Tests for configs/unified_config.py: unified config dataclasses and helpers.
"""

import pytest
import yaml

from nodelens.configs.unified_config import (
    CalibrationConfig,
    CascadeConfig,
    ClusteringConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    HaloConfig,
    MetricsConfig,
    ModelConfig,
    OutputConfig,
    PruningConfig,
    SupernodeConfig,
    UnifiedConfig,
    VisualizationConfig,
    _deep_merge,
    _normalize_metric_name,
    create_config_template,
    load_unified_config,
)

# =========================================================================
# ExperimentConfig
# =========================================================================


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.name == "experiment"
        assert cfg.type == "alignment_analysis"
        assert cfg.seed == 42
        assert cfg.device == "cuda"

    def test_from_dict(self):
        cfg = ExperimentConfig.from_dict({"name": "my_exp", "seed": 123})
        assert cfg.name == "my_exp"
        assert cfg.seed == 123

    def test_from_dict_old_style(self):
        cfg = ExperimentConfig.from_dict(
            {
                "experiment_name": "old_exp",
                "experiment_type": "cluster_analysis",
            }
        )
        assert cfg.name == "old_exp"
        assert cfg.type == "cluster_analysis"


# =========================================================================
# ModelConfig
# =========================================================================


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.name == "resnet18"
        assert cfg.pretrained is True
        assert cfg.dtype == "bfloat16"

    def test_from_dict(self):
        cfg = ModelConfig.from_dict({"name": "vgg16", "pretrained": False})
        assert cfg.name == "vgg16"
        assert cfg.pretrained is False

    def test_from_dict_ignores_unknown(self):
        cfg = ModelConfig.from_dict({"name": "resnet18", "unknown_field": 42})
        assert cfg.name == "resnet18"


# =========================================================================
# DatasetConfig
# =========================================================================


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.name == "cifar10"
        assert cfg.batch_size == 128

    def test_from_dict(self):
        cfg = DatasetConfig.from_dict({"name": "cifar100", "batch_size": 64})
        assert cfg.name == "cifar100"
        assert cfg.batch_size == 64


# =========================================================================
# CalibrationConfig
# =========================================================================


class TestCalibrationConfig:
    def test_defaults(self):
        cfg = CalibrationConfig()
        assert cfg.num_samples == 5000
        assert cfg.batch_size == 4

    def test_from_dict_old_style(self):
        cfg = CalibrationConfig.from_dict({"n_calibration_samples": 128})
        assert cfg.num_samples == 128


# =========================================================================
# MetricsConfig
# =========================================================================


class TestMetricsConfig:
    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.rayleigh_quotient.enabled is True
        assert cfg.redundancy.enabled is True
        assert cfg.synergy.enabled is True
        assert cfg.magnitude.enabled is True

    def test_from_dict_enabled_list(self):
        cfg = MetricsConfig.from_dict({"enabled": ["rq", "redundancy"]})
        assert cfg.rayleigh_quotient.enabled is True

    def test_from_dict_individual_configs(self):
        cfg = MetricsConfig.from_dict(
            {
                "rayleigh_quotient": {"relative": False},
            }
        )
        assert cfg.rayleigh_quotient.params.get("relative") is False

    def test_from_dict_boolean_flags(self):
        cfg = MetricsConfig.from_dict(
            {
                "rayleigh_quotient": False,
            }
        )
        assert cfg.rayleigh_quotient.enabled is False

    def test_from_dict_old_style_flags(self):
        cfg = MetricsConfig.from_dict(
            {
                "compute_rq": True,
                "compute_redundancy": True,
                "compute_synergy": True,
            }
        )
        assert cfg.rayleigh_quotient.enabled is True
        assert cfg.redundancy.enabled is True
        assert cfg.synergy.enabled is True

    def test_composite_weights(self):
        cfg = MetricsConfig.from_dict(
            {
                "composite_weights": {"rayleigh_quotient": 0.5},
            }
        )
        assert cfg.composite_weights["rayleigh_quotient"] == 0.5


# =========================================================================
# ClusteringConfig
# =========================================================================


class TestClusteringConfig:
    def test_defaults(self):
        cfg = ClusteringConfig()
        assert cfg.n_clusters == 4
        assert cfg.normalize_features is True
        assert cfg.stability_enabled is True

    def test_from_dict_old_style(self):
        cfg = ClusteringConfig.from_dict({"compute_stability": False, "n_clusters": 3})
        assert cfg.stability_enabled is False
        assert cfg.n_clusters == 3


# =========================================================================
# Other sub-configs
# =========================================================================


class TestSubConfigs:
    def test_supernode_defaults(self):
        cfg = SupernodeConfig()
        assert cfg.enabled is False
        assert cfg.core_fraction == 0.01

    def test_supernode_from_dict(self):
        cfg = SupernodeConfig.from_dict({"enabled": True, "core_fraction": 0.05})
        assert cfg.enabled is True
        assert cfg.core_fraction == 0.05

    def test_halo_defaults(self):
        cfg = HaloConfig()
        assert cfg.percentile == 90.0
        assert cfg.use_activation_weight is True

    def test_halo_from_dict(self):
        cfg = HaloConfig.from_dict({"percentile": 95.0, "max_refs": 256})
        assert cfg.percentile == 95.0
        assert cfg.max_refs == 256

    def test_cascade_defaults(self):
        cfg = CascadeConfig()
        assert cfg.n_remove_per_group == 5

    def test_cascade_from_dict_old_style(self):
        cfg = CascadeConfig.from_dict({"n_remove_per_cluster": 10})
        assert cfg.n_remove_per_group == 10

    def test_output_defaults(self):
        cfg = OutputConfig()
        assert cfg.dir == "./results"
        assert cfg.save_metrics is True

    def test_output_from_dict(self):
        cfg = OutputConfig.from_dict({"dir": "/tmp/out", "save_checkpoints": True})
        assert cfg.dir == "/tmp/out"
        assert cfg.save_checkpoints is True


# =========================================================================
# PruningConfig
# =========================================================================


class TestPruningConfig:
    def test_defaults(self):
        cfg = PruningConfig()
        assert cfg.enabled is True
        assert 0.5 in cfg.ratios
        assert cfg.distribution == "uniform"

    def test_from_dict_ratios(self):
        cfg = PruningConfig.from_dict({"ratios": [0.1, 0.3, 0.5]})
        assert cfg.ratios == [0.1, 0.3, 0.5]

    def test_from_dict_sparsity_levels(self):
        cfg = PruningConfig.from_dict({"sparsity_levels": [0.2, 0.4]})
        assert cfg.ratios == [0.2, 0.4]

    def test_from_dict_methods_strings(self):
        cfg = PruningConfig.from_dict({"methods": ["magnitude", "random"]})
        assert len(cfg.methods) == 2
        assert cfg.methods[0].name == "magnitude"

    def test_from_dict_methods_dicts(self):
        cfg = PruningConfig.from_dict(
            {
                "methods": [{"name": "cap", "selection": "high"}],
            }
        )
        assert cfg.methods[0].name == "cap"
        assert cfg.methods[0].selection == "high"

    def test_from_dict_fine_tune(self):
        cfg = PruningConfig.from_dict(
            {
                "fine_tune": {"enabled": True, "epochs": 5, "lr": 0.001},
            }
        )
        assert cfg.fine_tune_enabled is True
        assert cfg.fine_tune_epochs == 5
        assert cfg.fine_tune_lr == 0.001


# =========================================================================
# EvaluationConfig
# =========================================================================


class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.perplexity_enabled is False
        assert cfg.benchmarks_enabled is False

    def test_from_dict_perplexity(self):
        cfg = EvaluationConfig.from_dict(
            {
                "perplexity": {"enabled": True, "datasets": ["wikitext", "c4"]},
            }
        )
        assert cfg.perplexity_enabled is True
        assert "wikitext" in cfg.perplexity_datasets
        assert "c4" in cfg.perplexity_datasets

    def test_from_dict_benchmarks(self):
        cfg = EvaluationConfig.from_dict(
            {
                "benchmarks": ["hellaswag", "arc"],
            }
        )
        assert cfg.benchmarks_enabled is True
        assert "hellaswag" in cfg.benchmark_tasks

    def test_from_dict_perplexity_dict_datasets(self):
        cfg = EvaluationConfig.from_dict(
            {
                "perplexity": {
                    "enabled": True,
                    "datasets": [{"name": "wikitext"}, {"name": "c4"}],
                },
            }
        )
        assert "wikitext" in cfg.perplexity_datasets


# =========================================================================
# VisualizationConfig
# =========================================================================


class TestVisualizationConfig:
    def test_defaults(self):
        cfg = VisualizationConfig()
        assert cfg.format == "png"
        assert cfg.dpi == 300
        assert cfg.histograms is True

    def test_from_dict_direct(self):
        cfg = VisualizationConfig.from_dict({"format": "pdf", "dpi": 150})
        assert cfg.format == "pdf"
        assert cfg.dpi == 150

    def test_from_dict_plots_nested(self):
        cfg = VisualizationConfig.from_dict(
            {
                "plots": {"histograms": False, "violin_plots": False},
            }
        )
        assert cfg.histograms is False
        assert cfg.violin_plots is False

    def test_from_dict_figures_list(self):
        cfg = VisualizationConfig.from_dict(
            {
                "figures": ["correlation-heatmap", "cluster-scatter"],
            }
        )
        assert cfg.correlation_heatmap is True
        assert cfg.cluster_scatter is True


# =========================================================================
# UnifiedConfig
# =========================================================================


class TestUnifiedConfig:
    def test_defaults(self):
        cfg = UnifiedConfig()
        assert cfg.experiment.name == "experiment"
        assert cfg.model.name == "resnet18"
        assert cfg.clustering.enabled is True

    def test_from_dict_minimal(self):
        cfg = UnifiedConfig.from_dict(
            {
                "experiment": {"name": "test_exp"},
            }
        )
        assert cfg.experiment.name == "test_exp"

    def test_from_dict_flat_style(self):
        cfg = UnifiedConfig.from_dict(
            {
                "experiment_name": "flat_exp",
            }
        )
        assert cfg.experiment.name == "flat_exp"

    def test_from_dict_all_sections(self):
        cfg = UnifiedConfig.from_dict(
            {
                "experiment": {"name": "full", "type": "cluster_analysis"},
                "model": {"name": "vgg16"},
                "dataset": {"name": "cifar100", "batch_size": 64},
                "calibration": {"num_samples": 1000},
                "metrics": {"rayleigh_quotient": {"relative": True}},
                "clustering": {"n_clusters": 3},
                "supernode": {"enabled": True},
                "halo_analysis": {"percentile": 95.0},
                "cascade_analysis": {"n_remove_per_group": 3},
                "pruning": {"ratios": [0.3, 0.5]},
                "evaluation": {"perplexity": {"enabled": True}},
                "visualization": {"format": "pdf"},
                "output": {"dir": "/tmp/out"},
            }
        )
        assert cfg.experiment.name == "full"
        assert cfg.model.name == "vgg16"
        assert cfg.dataset.batch_size == 64
        assert cfg.clustering.n_clusters == 3
        assert cfg.supernode.enabled is True
        assert cfg.output.dir == "/tmp/out"

    def test_extra_fields_captured(self):
        cfg = UnifiedConfig.from_dict(
            {
                "experiment": {"name": "test"},
                "custom_field": {"key": "value"},
            }
        )
        assert "custom_field" in cfg.extra
        assert cfg.extra["custom_field"]["key"] == "value"

    def test_to_dict(self):
        cfg = UnifiedConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "experiment" in d
        assert d["experiment"]["name"] == "experiment"

    def test_save_and_load(self, tmp_path):
        cfg = UnifiedConfig()
        cfg.experiment.name = "save_test"
        fpath = tmp_path / "config.yaml"
        cfg.save(fpath)
        assert fpath.exists()
        loaded = load_unified_config(fpath)
        assert loaded.experiment.name == "save_test"

    def test_validate_valid_config(self):
        cfg = UnifiedConfig()
        warnings = cfg.validate()
        assert isinstance(warnings, list)

    def test_validate_unknown_type(self):
        cfg = UnifiedConfig()
        cfg.experiment.type = "totally_bogus"
        warnings = cfg.validate()
        assert any("Unknown experiment type" in w for w in warnings)

    def test_validate_llm_missing_model_id(self):
        cfg = UnifiedConfig()
        cfg.experiment.type = "llm_alignment"
        cfg.model.model_id = None
        warnings = cfg.validate()
        assert any("model.model_id" in w for w in warnings)

    def test_validate_clustering_metric_disabled(self):
        cfg = UnifiedConfig()
        cfg.clustering.enabled = True
        cfg.clustering.features = ["rayleigh_quotient"]
        cfg.metrics.rayleigh_quotient.enabled = False
        warnings = cfg.validate()
        assert any("not enabled" in w for w in warnings)


# =========================================================================
# Helper functions
# =========================================================================


class TestNormalizeMetricName:
    def test_known_aliases(self):
        assert _normalize_metric_name("rq") == "rayleigh_quotient"
        assert _normalize_metric_name("rayleigh") == "rayleigh_quotient"
        assert _normalize_metric_name("gaussian_mi") == "redundancy"
        assert _normalize_metric_name("synergy_gaussian_mmi") == "synergy"
        assert _normalize_metric_name("activation_l2_norm") == "magnitude"

    def test_passthrough(self):
        assert _normalize_metric_name("rayleigh_quotient") == "rayleigh_quotient"
        assert _normalize_metric_name("custom_metric") == "custom_metric"


class TestDeepMerge:
    def test_basic_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 10, "z": 20}}
        _deep_merge(base, override)
        assert base["a"] == {"x": 1, "y": 10, "z": 20}
        assert base["b"] == 3


class TestLoadUnifiedConfig:
    def test_load_yaml(self, tmp_path):
        config = {
            "experiment": {"name": "load_test", "type": "cluster_analysis"},
            "model": {"name": "resnet18"},
        }
        fpath = tmp_path / "config.yaml"
        fpath.write_text(yaml.dump(config))
        loaded = load_unified_config(fpath)
        assert loaded.experiment.name == "load_test"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_unified_config("/nonexistent/config.yaml")

    def test_inheritance(self, tmp_path):
        base = {
            "experiment": {"name": "base", "type": "cluster_analysis"},
            "model": {"name": "resnet18"},
        }
        child = {
            "_inherit": "base.yaml",
            "experiment": {"name": "child"},
        }
        (tmp_path / "base.yaml").write_text(yaml.dump(base))
        (tmp_path / "child.yaml").write_text(yaml.dump(child))
        loaded = load_unified_config(tmp_path / "child.yaml")
        assert loaded.experiment.name == "child"
        assert loaded.model.name == "resnet18"  # Inherited


class TestCreateConfigTemplate:
    def test_cluster_analysis(self):
        cfg = create_config_template("cluster_analysis")
        assert cfg.experiment.type == "cluster_analysis"
        assert cfg.model.name == "resnet18"
        assert cfg.clustering.enabled is True

    def test_llm_alignment(self):
        cfg = create_config_template("llm_alignment")
        assert cfg.experiment.type == "llm_alignment"
        assert cfg.model.model_id == "meta-llama/Llama-3.1-8B"
        assert cfg.evaluation.perplexity_enabled is True
        assert cfg.clustering.enabled is False
