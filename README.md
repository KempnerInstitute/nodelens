# Alignment Framework

Neural network analysis and structured pruning using alignment metrics and information theory.

## Overview

This framework provides tools for analyzing and pruning neural networks through:

- **Alignment metrics**: Rayleigh quotient, activation-based importance
- **Information-theoretic analysis**: Mutual information, redundancy, synergy
- **Cluster-based analysis**: Functional type identification, cross-layer halo tracking
- **Structured pruning**: Channel/neuron removal with multiple scoring strategies

**Supported architectures**: MLPs, CNNs (ResNet, VGG, MobileNet), Transformers, LLMs (LLaMA, Mistral, Qwen)

## Installation

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

## Quick Start

```bash
# Vision model analysis
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# CNN pruning
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml

# LLM analysis
python scripts/run_experiment.py --config configs/paper/llama3_8b_full.yaml

# Cluster-based analysis
python scripts/run_experiment.py --config configs/cluster_analysis/resnet18_cifar10_full.yaml
```

## Experiment Types

| Type | Description | Config Example |
|------|-------------|----------------|
| `alignment_analysis` | General alignment metrics | `mnist_basic.yaml` |
| `llm_alignment` | LLM supernode/SCAR analysis | `llama3_8b_full.yaml` |
| `cluster_analysis` | Metric-space clustering with halos | `resnet18_cifar10_full.yaml` |

## Metrics

| Category | Metrics |
|----------|---------|
| Activation | `activation_l2_norm`, `activation_variance`, `activation_outlier_index` |
| Alignment | `rayleigh_quotient`, `delta_alignment` |
| Information | `mutual_information_gaussian`, `pairwise_redundancy_gaussian`, `gaussian_pid_synergy_mmi` |
| SCAR (LLM) | `scar_activation_power`, `scar_taylor`, `scar_curvature`, `scar_loss_proxy` |
| Synergy | `synergy_continuous_target` (with logit margin) |

## Cluster-Based Analysis

The cluster analysis framework groups channels/neurons into functional types:

| Type | Characteristics | Pruning Implication |
|------|-----------------|---------------------|
| Critical | High RQ, Low Redundancy, High Synergy | Protect |
| Redundant | Moderate RQ, High Redundancy | Target for pruning |
| Synergistic | Moderate RQ, High Synergy | Preserve pairs |
| Background | Low on all metrics | Safe to remove |

Cross-layer halo analysis tracks downstream dependencies to predict cascade effects.

## Pruning Strategies

| Strategy | Description |
|----------|-------------|
| `magnitude` | Prune by weight magnitude |
| `alignment` | Prune by alignment score |
| `composite` | Combine multiple metrics |
| `cluster_aware` | Use cluster membership and halo analysis |
| `random` | Random baseline |

## Project Structure

```
alignment/
├── configs/
|   ├── cluster_analysis/   # Cluster-based analysis configs
|   ├── paper/              # Paper experiment configs
|   └── examples/           # Example configs
├── scripts/
|   ├── run_experiment.py   # Main entry point
|   └── run_analysis.py     # Post-hoc analysis
├── src/alignment/
|   ├── analysis/           # Visualization, clustering, cascade analysis
|   ├── experiments/        # Experiment classes
|   ├── metrics/            # Importance metrics
|   ├── models/             # Model wrappers
|   └── pruning/            # Pruning strategies
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Key Modules

### Analysis
- `MetricSpaceClustering`: K-means clustering in (RQ, Redundancy, Synergy) space
- `CrossLayerHaloAnalysis`: Track downstream channel dependencies
- `CascadeAnalysis`: Validate importance via ablation
- `UnifiedVisualizer`: Generate analysis plots

### Experiments
- `GeneralAlignmentExperiment`: Vision model analysis
- `LLMAlignmentExperiment`: LLM supernode and SCAR analysis
- `ClusterAnalysisExperiment`: Cluster-based analysis for any architecture

### Metrics
- `RayleighQuotient`: Input-weight alignment
- `PairwiseRedundancyGaussian`: Gaussian MI-based redundancy
- `SynergyContinuousTarget`: PID synergy with continuous target
- SCAR metrics for LLMs

## Documentation

- [Usage Guide](docs/usage.md) - Running experiments and configuration
- [API Reference](docs/api_reference.md) - Core classes and functions
- [LLM Guide](docs/llm_guide.md) - LLM-specific analysis
- [Metric Consistency](docs/METRIC_CONSISTENCY.md) - Theory-code verification

## Configuration

```yaml
experiment_type: cluster_analysis  # or llm_alignment, alignment_analysis

model:
  name: resnet18
  pretrained: true

dataset:
  name: cifar10
  batch_size: 128

clustering:
  n_clusters: 4
  compute_stability: true

halo_analysis:
  percentile: 90.0

pruning:
  ratios: [0.3, 0.5, 0.7]
  methods: [magnitude, taylor, cluster_aware]
```

See `configs/template.yaml` for complete parameter reference.

## Testing

```bash
pytest tests/
pytest tests/unit/ -v
```

## License

See LICENSE file.
