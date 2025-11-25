# Usage Guide

This document describes how to run experiments, configure analysis and pruning, and generate visualizations.

## Running Experiments

Experiments are driven by YAML configuration files:

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

Example configurations:

| Config | Description |
|--------|-------------|
| `configs/examples/mnist_basic.yaml` | MLP on MNIST with alignment metrics |
| `configs/examples/resnet_pruning.yaml` | ResNet-18 pruning on CIFAR-10 |
| `configs/examples/llama3_scoring.yaml` | LLaMA importance scoring |
| `configs/examples/llama3_pruning.yaml` | LLaMA pruning |
| `configs/projects/llm_supernode.yaml` | LLM SCAR analysis |
| `configs/projects/vision_synergy.yaml` | Vision redundancy/synergy analysis |

## Command-Line Options

```bash
python scripts/run_experiment.py \
  --config configs/examples/resnet_pruning.yaml \
  --device cuda:0 \
  --seed 42 \
  --output-dir results/my_run
```

| Option | Description |
|--------|-------------|
| `--config PATH` | YAML configuration file (required) |
| `--device STRING` | Override device (cuda:0, cpu) |
| `--seed INT` | Override random seed |
| `--output-dir PATH` | Override output directory |
| `--analysis-only` | Regenerate plots from existing results |
| `--experiment-dir PATH` | Existing experiment directory (with --analysis-only) |

## Configuration Structure

```yaml
experiment:
  name: "my_experiment"
  type: "general_alignment"
  seed: 42
  device: "cuda"

model:
  name: "resnet18"
  pretrained: true

dataset:
  name: "cifar10"
  data_path: "./data"
  batch_size: 128

alignment_methods:
  - "rayleigh_quotient"
  - "pairwise_redundancy_gaussian"

pruning:
  enabled: true
  algorithms: ["alignment"]
  sparsity_levels: [0.3, 0.5, 0.7]
  selection_modes: ["low"]
  structured: true
  dependency_aware: true
  fine_tune_after_pruning: true

visualization:
  enabled: true
  format: "png"
  dpi: 300
  training_curves: true
  pruning_plots: true

# Post-experiment analysis (optional)
post_analysis:
  analyses:
    - histograms
    - scatter_plots
    - pruning_curves
  histograms:
    bins: 100
    top_k: 5
```

See `configs/template.yaml` for all available options.

## Visualization Options

### Inline Visualization (visualization block)

Controls plots generated during experiment execution:

| Option | Description |
|--------|-------------|
| `enabled` | Enable/disable plot generation |
| `format` | Output format (png, pdf, svg) |
| `dpi` | Resolution |
| `training_curves` | Training loss/accuracy plots |
| `alignment_curves` | Alignment metric evolution |
| `dropout_plots` | Dropout analysis plots |
| `eigen_plots` | Eigenvalue heatmaps |
| `pruning_plots` | Pruning performance plots |

### Post-Experiment Analysis (post_analysis block)

Runs additional analysis after experiment completes:

```yaml
post_analysis:
  analyses:
    - histograms        # Importance score distributions
    - scatter_plots     # Metric correlations
    - heatmaps          # Layer-metric heatmaps
    - layer_distributions  # Violin/box plots
    - pruning_curves    # Sparsity vs performance
    - scar_analysis     # SCAR metrics (LLM)
  
  histograms:
    bins: 100
    top_k: 5
    metrics: ["rayleigh_quotient", "activation_l2_norm"]
  
  scatter_plots:
    pairs:
      - ["activation_l2_norm", "rayleigh_quotient"]
```

## Standalone Analysis

For analysis of existing results without running experiments:

```bash
# Run analysis from config
python scripts/run_analysis.py --config configs/analysis/vision_figures.yaml

# Quick analysis
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick

# Specific analyses
python scripts/run_analysis.py --results-dir ./results \
    --analyses histograms pruning_curves \
    --output-dir ./custom_plots
```

### Analysis Configuration Files

| Config | Description |
|--------|-------------|
| `configs/analysis/llm_paper_figures.yaml` | LLM paper figures |
| `configs/analysis/vision_figures.yaml` | Vision experiment figures |
| `configs/analysis/llm_layer_analysis.yaml` | LLM layer analysis |

### Programmatic Analysis

```python
from alignment.analysis import AnalysisRunner, AnalysisConfig

config = AnalysisConfig(
    results_dir="./results",
    output_dir="./plots",
    analyses=["histograms", "pruning_curves"],
)
runner = AnalysisRunner(config)
outputs = runner.run()
```

## Pruning Configuration

### Basic Pruning

```yaml
pruning:
  enabled: true
  algorithms: ["alignment"]
  sparsity_levels: [0.3, 0.5, 0.7]
  selection_modes: ["low"]
  alignment_metric: "rayleigh_quotient"
```

### Dependency-Aware Pruning

For models with skip connections (ResNet, DenseNet):

```yaml
pruning:
  enabled: true
  structured: true
  dependency_aware: true
```

### Available Algorithms

| Algorithm | Description |
|-----------|-------------|
| `magnitude` | Prune by weight magnitude |
| `alignment` | Prune by alignment score |
| `hybrid` | Combine magnitude and alignment |
| `random` | Random baseline |
| `gradient` | Gradient-based importance |

## Output Structure

```
results/experiment_YYYYMMDD_HHMMSS/
├── experiment_config.yaml
├── experiment.log
├── results_YYYYMMDD_HHMMSS.json
├── checkpoints/
├── plots/
│   ├── training_loss.png
│   ├── pruning_accuracy.png
│   └── ...
└── analysis/           # From post_analysis
    ├── histograms/
    ├── scatter_plots/
    └── ...
```

## Workflow Examples

### Basic Vision Experiment

```bash
# Run experiment with pruning
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml

# Results in results/resnet_pruning_YYYYMMDD_HHMMSS/
```

### LLM Analysis

```bash
# Compute importance scores
python scripts/run_experiment.py --config configs/examples/llama3_scoring.yaml

# Generate analysis figures
python scripts/run_analysis.py --config configs/analysis/llm_paper_figures.yaml
```

### Regenerate Plots

```bash
# Regenerate plots from existing experiment
python scripts/run_experiment.py \
  --config configs/examples/resnet_pruning.yaml \
  --analysis-only \
  --experiment-dir results/resnet_pruning_20240101_120000
```
