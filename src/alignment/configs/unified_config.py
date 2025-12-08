"""
Unified configuration schema for the alignment framework.

This module provides a single, validated configuration structure that works
for all experiment types (vision, LLM, custom). Configuration can be loaded
from YAML files and validated against the schema.

Features:
- Unified structure for vision and LLM experiments
- Automatic validation with helpful error messages
- Default values with override capability
- Support for config inheritance/presets
- Registry-aware: validates that referenced components exist

Usage:
    from alignment.configs.unified_config import load_unified_config, UnifiedConfig
    
    # From YAML file
    config = load_unified_config("configs/my_experiment.yaml")
    
    # Programmatically
    config = UnifiedConfig(
        experiment=ExperimentConfig(name="my_exp", type="cluster_analysis"),
        model=ModelConfig(name="resnet18"),
        ...
    )
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ExperimentConfig:
    """Experiment-level configuration."""
    name: str = "experiment"
    type: str = "alignment_analysis"  # alignment_analysis, cluster_analysis, llm_alignment
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./results"
    num_networks: int = 1
    
    # Backward compatibility aliases
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        # Handle old-style flat configs
        return cls(
            name=d.get("name") or d.get("experiment_name", "experiment"),
            type=d.get("type") or d.get("experiment_type", "alignment_analysis"),
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
            output_dir=d.get("output_dir", "./results"),
            num_networks=d.get("num_networks", 1),
        )


@dataclass
class ModelConfig:
    """Model configuration (works for both vision and LLM)."""
    name: str = "resnet18"
    pretrained: bool = True
    num_classes: Optional[int] = None  # Vision only
    
    # HuggingFace / LLM specific
    model_id: Optional[str] = None  # e.g., "meta-llama/Llama-3.1-8B"
    dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"
    trust_remote_code: bool = True
    
    # Layer tracking
    tracked_layers: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "cifar10"
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    
    # LLM specific
    subset: Optional[str] = None  # e.g., "wikitext-2-raw-v1"
    split: str = "train"
    max_length: int = 2048
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CalibrationConfig:
    """Calibration data configuration."""
    num_samples: int = 5000  # Vision: 5000, LLM: 64-128
    max_length: int = 2048   # LLM only
    batch_size: int = 4
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationConfig":
        # Handle old-style nested configs
        if "n_calibration_samples" in d:
            d["num_samples"] = d.pop("n_calibration_samples")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MetricItemConfig:
    """Configuration for a single metric."""
    enabled: bool = True
    # Metric-specific parameters stored as dict
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """Metrics configuration (unified naming)."""
    # Core metrics available for both vision and LLM
    rayleigh_quotient: MetricItemConfig = field(default_factory=lambda: MetricItemConfig(enabled=True, params={"relative": True}))
    redundancy: MetricItemConfig = field(default_factory=lambda: MetricItemConfig(enabled=True, params={"sampling": "all"}))
    synergy: MetricItemConfig = field(default_factory=lambda: MetricItemConfig(enabled=True, params={"target": "logit_margin", "num_pairs": 10}))
    magnitude: MetricItemConfig = field(default_factory=lambda: MetricItemConfig(enabled=True, params={"type": "l2_norm"}))
    
    # Additional metrics (optional)
    additional: Dict[str, MetricItemConfig] = field(default_factory=dict)
    
    # Composite weights for combined scoring
    composite_weights: Dict[str, float] = field(default_factory=lambda: {
        "rayleigh_quotient": 0.33,
        "redundancy": -0.33,
        "synergy": 0.33,
    })
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricsConfig":
        config = cls()
        
        # Handle 'enabled' list format
        if "enabled" in d and isinstance(d["enabled"], list):
            for metric_name in d["enabled"]:
                canonical = _normalize_metric_name(metric_name)
                if hasattr(config, canonical):
                    getattr(config, canonical).enabled = True
        
        # Handle individual metric configs
        for metric_name in ["rayleigh_quotient", "redundancy", "synergy", "magnitude"]:
            if metric_name in d:
                if isinstance(d[metric_name], dict):
                    getattr(config, metric_name).params.update(d[metric_name])
                elif isinstance(d[metric_name], bool):
                    getattr(config, metric_name).enabled = d[metric_name]
        
        # Handle old-style boolean flags
        if d.get("compute_rq"):
            config.rayleigh_quotient.enabled = True
        if d.get("compute_redundancy"):
            config.redundancy.enabled = True
        if d.get("compute_synergy"):
            config.synergy.enabled = True
        
        if "composite_weights" in d:
            config.composite_weights.update(d["composite_weights"])
        
        return config


@dataclass
class ClusteringConfig:
    """Clustering configuration."""
    enabled: bool = True
    n_clusters: int = 4
    type_names: List[str] = field(default_factory=lambda: ["critical", "redundant", "synergistic", "background"])
    normalize_features: bool = True
    features: List[str] = field(default_factory=lambda: ["rayleigh_quotient", "redundancy", "synergy"])
    
    # Stability analysis
    stability_enabled: bool = True
    n_bootstrap: int = 50
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClusteringConfig":
        if "compute_stability" in d:
            d["stability_enabled"] = d.pop("compute_stability")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SupernodeConfig:
    """Supernode detection configuration."""
    enabled: bool = False
    score_metric: str = "synergy"  # Or: rayleigh_quotient, magnitude, scar_loss_proxy
    core_fraction: float = 0.01   # Top 1% as supernodes
    halo_fraction: float = 0.10   # 10% as halo
    follower_fraction: float = 0.10
    protect_core: bool = True
    
    # Cross-layer analysis
    cross_layer_analysis: bool = True
    compare_by_connection: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SupernodeConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HaloConfig:
    """Halo/cross-layer analysis configuration."""
    enabled: bool = True
    percentile: float = 90.0
    use_activation_weight: bool = True
    compute_influence_matrix: bool = True
    
    # For LLM
    max_refs: int = 512
    sample_pairs: int = 2000
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HaloConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CascadeConfig:
    """Cascade/damage analysis configuration."""
    enabled: bool = True
    n_remove_per_group: int = 5
    damage_sample_fraction: float = 0.2
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CascadeConfig":
        if "n_remove_per_cluster" in d:
            d["n_remove_per_group"] = d.pop("n_remove_per_cluster")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PruningMethodConfig:
    """Configuration for a single pruning method."""
    name: str
    selection: str = "low"  # low, high, random
    weights: Optional[Dict[str, float]] = None  # For composite methods
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningConfig:
    """Pruning configuration."""
    enabled: bool = True
    ratios: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    # Methods to compare
    methods: List[PruningMethodConfig] = field(default_factory=list)
    
    # Selection modes
    selection_modes: List[str] = field(default_factory=lambda: ["low", "high"])
    
    # Fine-tuning
    fine_tune_enabled: bool = True
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 0.0001
    
    # Distribution strategy
    distribution: str = "uniform"  # uniform, global_threshold, adaptive
    structured: bool = True
    dependency_aware: bool = False
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PruningConfig":
        config = cls()
        
        if "ratios" in d:
            config.ratios = d["ratios"]
        elif "sparsity_levels" in d:
            config.ratios = d["sparsity_levels"]
        
        if "methods" in d:
            config.methods = []
            for method in d["methods"]:
                if isinstance(method, str):
                    config.methods.append(PruningMethodConfig(name=method))
                elif isinstance(method, dict):
                    config.methods.append(PruningMethodConfig(**method))
        
        if "fine_tune" in d or "fine_tuning" in d:
            ft = d.get("fine_tune") or d.get("fine_tuning", {})
            config.fine_tune_enabled = ft.get("enabled", True)
            config.fine_tune_epochs = ft.get("epochs", 10)
            config.fine_tune_lr = ft.get("lr") or ft.get("learning_rate", 0.0001)
        
        for key in ["enabled", "distribution", "structured", "dependency_aware", "selection_modes"]:
            if key in d:
                setattr(config, key, d[key])
        
        return config


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    enabled: bool = True
    accuracy: bool = True
    loss: bool = True
    
    # LLM specific
    perplexity_enabled: bool = False
    perplexity_datasets: List[str] = field(default_factory=lambda: ["wikitext"])
    benchmarks_enabled: bool = False
    benchmark_tasks: List[str] = field(default_factory=list)
    benchmark_fewshot: int = 0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationConfig":
        config = cls()
        
        if "perplexity" in d:
            ppl = d["perplexity"]
            config.perplexity_enabled = ppl.get("enabled", True)
            if "datasets" in ppl:
                config.perplexity_datasets = [
                    ds["name"] if isinstance(ds, dict) else ds 
                    for ds in ppl["datasets"]
                ]
        
        if "benchmarks" in d:
            config.benchmarks_enabled = True
            config.benchmark_tasks = [
                b["name"] if isinstance(b, dict) else b
                for b in d["benchmarks"]
            ]
        
        return config


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    enabled: bool = True
    format: str = "png"  # png, pdf, svg
    dpi: int = 300
    
    # Plot types to generate
    histograms: bool = True
    violin_plots: bool = True
    correlation_heatmap: bool = True
    cluster_scatter: bool = True
    cluster_evolution: bool = True
    influence_matrix: bool = True
    halo_properties: bool = True
    pruning_comparison: bool = True
    pruning_recovery: bool = True
    cascade_test: bool = True
    
    # Scatter plot pairs
    scatter_pairs: List[List[str]] = field(default_factory=lambda: [
        ["rayleigh_quotient", "redundancy"],
        ["rayleigh_quotient", "synergy"],
        ["redundancy", "synergy"],
    ])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualizationConfig":
        config = cls()
        
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle nested 'plots' dict
        if "plots" in d:
            for key, value in d["plots"].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Handle 'figures' list
        if "figures" in d:
            for fig_type in d["figures"]:
                attr_name = fig_type.replace("-", "_")
                if hasattr(config, attr_name):
                    setattr(config, attr_name, True)
        
        return config


@dataclass
class OutputConfig:
    """Output configuration."""
    dir: str = "./results"
    save_metrics: bool = True
    save_clusters: bool = True
    save_figures: bool = True
    save_checkpoints: bool = False
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# UNIFIED CONFIG
# =============================================================================

@dataclass
class UnifiedConfig:
    """
    Unified configuration for all experiment types.
    
    This config structure works for:
    - Vision experiments (ResNet, VGG, etc. on CIFAR, ImageNet)
    - LLM experiments (Llama, Qwen, etc.)
    - Custom experiments
    
    All fields have sensible defaults, so you only need to specify
    what you want to change.
    """
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    supernode: SupernodeConfig = field(default_factory=SupernodeConfig)
    halo_analysis: HaloConfig = field(default_factory=HaloConfig)
    cascade_analysis: CascadeConfig = field(default_factory=CascadeConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Extra fields for custom extensions
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedConfig":
        """Create config from dictionary (e.g., loaded from YAML)."""
        config = cls()
        
        # Handle experiment config (with backward compat for flat style)
        if "experiment" in d:
            config.experiment = ExperimentConfig.from_dict(d["experiment"])
        else:
            config.experiment = ExperimentConfig.from_dict(d)
        
        # Model
        if "model" in d:
            config.model = ModelConfig.from_dict(d["model"])
        
        # Dataset
        if "dataset" in d:
            config.dataset = DatasetConfig.from_dict(d["dataset"])
        
        # Calibration (can be in 'calibration' or 'metrics')
        if "calibration" in d:
            config.calibration = CalibrationConfig.from_dict(d["calibration"])
        elif "metrics" in d:
            config.calibration = CalibrationConfig.from_dict(d["metrics"])
        
        # Metrics
        if "metrics" in d:
            config.metrics = MetricsConfig.from_dict(d["metrics"])
        
        # Clustering
        if "clustering" in d:
            config.clustering = ClusteringConfig.from_dict(d["clustering"])
        
        # Supernode
        if "supernode" in d:
            config.supernode = SupernodeConfig.from_dict(d["supernode"])
        
        # Halo
        if "halo_analysis" in d:
            config.halo_analysis = HaloConfig.from_dict(d["halo_analysis"])
        
        # Cascade
        if "cascade_analysis" in d:
            config.cascade_analysis = CascadeConfig.from_dict(d["cascade_analysis"])
        
        # Pruning
        if "pruning" in d:
            config.pruning = PruningConfig.from_dict(d["pruning"])
        
        # Evaluation
        if "evaluation" in d:
            config.evaluation = EvaluationConfig.from_dict(d["evaluation"])
        
        # Visualization
        if "visualization" in d:
            config.visualization = VisualizationConfig.from_dict(d["visualization"])
        
        # Output
        if "output" in d:
            config.output = OutputConfig.from_dict(d["output"])
        
        # Store any extra fields
        known_keys = {
            "experiment", "model", "dataset", "calibration", "metrics",
            "clustering", "supernode", "halo_analysis", "cascade_analysis",
            "pruning", "evaluation", "visualization", "output",
            "experiment_name", "experiment_type", "name", "type", "seed", "device",
        }
        config.extra = {k: v for k, v in d.items() if k not in known_keys}
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        import dataclasses
        
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        return convert(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved config to {path}")
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation warnings/errors (empty if valid)
        """
        warnings = []
        
        # Check experiment type
        valid_types = ["alignment_analysis", "cluster_analysis", "llm_alignment", "general_alignment"]
        if self.experiment.type not in valid_types:
            warnings.append(f"Unknown experiment type: {self.experiment.type}")
        
        # Check LLM-specific requirements
        if self.experiment.type == "llm_alignment":
            if not self.model.model_id:
                warnings.append("LLM experiments require model.model_id")
            if self.calibration.num_samples > 256:
                warnings.append("LLM calibration typically uses < 256 samples")
        
        # Check metric consistency
        if self.clustering.enabled:
            for feature in self.clustering.features:
                normalized = _normalize_metric_name(feature)
                if hasattr(self.metrics, normalized):
                    if not getattr(self.metrics, normalized).enabled:
                        warnings.append(f"Clustering uses '{feature}' but it's not enabled in metrics")
        
        return warnings


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_metric_name(name: str) -> str:
    """Normalize metric names to canonical form."""
    aliases = {
        "rq": "rayleigh_quotient",
        "rayleigh": "rayleigh_quotient",
        "alignment": "rayleigh_quotient",
        "gaussian_mi": "redundancy",
        "gaussian_mi_analytic": "redundancy",
        "average_redundancy": "redundancy",
        "pairwise_redundancy": "redundancy",
        "pairwise_redundancy_gaussian": "redundancy",
        "synergy_gaussian_mmi": "synergy",
        "synergy_continuous": "synergy",
        "activation_l2_norm": "magnitude",
        "l2_norm": "magnitude",
    }
    return aliases.get(name.lower(), name.lower())


def load_unified_config(path: Union[str, Path]) -> UnifiedConfig:
    """
    Load a unified config from a YAML file.
    
    Supports:
    - New unified format
    - Old LLM format (with 'experiment:' block)
    - Old vision format (with flat 'experiment_name:')
    - Config inheritance via '_inherit' key
    
    Args:
        path: Path to YAML config file
        
    Returns:
        UnifiedConfig instance
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        raw = yaml.safe_load(f)
    
    # Handle inheritance
    if "_inherit" in raw:
        base_path = path.parent / raw.pop("_inherit")
        base_config = load_unified_config(base_path)
        # Merge: raw overrides base
        base_dict = base_config.to_dict()
        _deep_merge(base_dict, raw)
        raw = base_dict
    
    config = UnifiedConfig.from_dict(raw)
    
    # Validate
    warnings = config.validate()
    for warning in warnings:
        logger.warning(f"Config validation: {warning}")
    
    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def create_config_template(experiment_type: str = "cluster_analysis") -> UnifiedConfig:
    """
    Create a template config for a specific experiment type.
    
    Args:
        experiment_type: Type of experiment
        
    Returns:
        UnifiedConfig with appropriate defaults
    """
    config = UnifiedConfig()
    config.experiment.type = experiment_type
    
    if experiment_type == "llm_alignment":
        # LLM defaults
        config.model.name = "hf_causal_lm"
        config.model.model_id = "meta-llama/Llama-3.1-8B"
        config.dataset.name = "wikitext"
        config.calibration.num_samples = 128
        config.calibration.max_length = 2048
        config.supernode.enabled = True
        config.clustering.enabled = False
        config.evaluation.perplexity_enabled = True
    
    elif experiment_type == "cluster_analysis":
        # Vision clustering defaults
        config.model.name = "resnet18"
        config.model.num_classes = 10
        config.dataset.name = "cifar10"
        config.calibration.num_samples = 5000
        config.clustering.enabled = True
        config.supernode.enabled = False
    
    return config
