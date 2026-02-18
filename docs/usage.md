# Usage Guide

## Running Experiments

Experiments are configured via YAML files:

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--config PATH` | YAML configuration file (required) |
| `--device STRING` | Override device (cuda:0, cpu) |
| `--seed INT` | Override random seed |
| `--output-dir PATH` | Override output directory |
| `--analysis-only` | Regenerate plots from existing results |
| `--experiment-dir PATH` | Existing experiment directory (with --analysis-only) |

### Example Configurations

| Config | Description |
|--------|-------------|
| `configs/examples/mnist_basic.yaml` | MLP on MNIST |
| `configs/examples/resnet_pruning.yaml` | ResNet-18 pruning on CIFAR-10 |
| `configs/examples/llm_alignment.yaml` | LLM importance scoring |

## Configuration Structure

```yaml
experiment:
  name: "my_experiment"
  type: "general_alignment"  # or "llm_alignment"
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

visualization:
  enabled: true
  format: "png"
  dpi: 300
```

See `configs/template.yaml` for all parameters.

## Pruning Configuration

### Basic Pruning

```yaml
pruning:
  enabled: true
  algorithms: ["alignment"]
  sparsity_levels: [0.3, 0.5, 0.7]
  alignment_metric: "rayleigh_quotient"
```

### Structured Pruning

Removes entire neurons/channels (maintains dense tensors):

```yaml
pruning:
  structured: true
  dependency_aware: true  # For models with skip connections
```

### Available Algorithms

| Algorithm | Description |
|-----------|-------------|
| `magnitude` | Prune by weight magnitude |
| `alignment` | Prune by alignment score |
| `hybrid` | Combine magnitude and alignment |
| `random` | Random baseline |
| `gradient` | Gradient-based importance |

### Selection Modes

- `low`: Prune low-scoring neurons (standard)
- `high`: Prune high-scoring neurons (ablation)
- `random`: Random pruning (baseline)

## Analysis

### Standalone Analysis

Generate visualizations from existing results:

```bash
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick

python scripts/run_analysis.py --config configs/analysis_template.yaml \
    --analyses histograms pruning_curves
```

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

### Available Analyses

| Analysis | Description |
|----------|-------------|
| `histograms` | Importance score distributions |
| `scatter_plots` | Metric correlations |
| `heatmaps` | Layer-metric heatmaps |
| `pruning_curves` | Sparsity vs performance |
| `scar_analysis` | SCAR metrics (LLM) |
| `supernode_analysis` | Supernode identification and cross-layer analysis |

## Output Structure

```
results/experiment_YYYYMMDD_HHMMSS/
├── experiment_config.yaml
├── experiment.log
├── results_YYYYMMDD_HHMMSS.json
├── checkpoints/
└── plots/
    ├── training_loss.png
    ├── pruning_accuracy.png
    └── ...
```

## Workflow Examples

### Vision Experiment

```bash
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
```

### LLM Analysis

```bash
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml
```

## Supernode Analysis (LLM)

Supernode analysis identifies high-importance neurons and traces their influence across layers.

### Architecture Context (LLaMA FFN)

```
input(4096) -> gate_proj/up_proj(14336) -> down_proj -> output(4096) -> next layer
              up                          up
              INTERMEDIATE neurons       OUTPUT to residual stream
              (supernodes identified)    (cross-layer analysis)
```

### Analysis Workflow

1. **Compute metrics** on intermediate neurons (14336 dim) using the selected `score_metric`
2. **Identify supernodes** as top neurons by the metric (e.g., top 1%)
3. **Trace outgoing weights** from supernodes through `down_proj`
4. **Cross-layer analysis** (optional): Analyze next layer's input neurons
   - Identify neurons with high weight connections from supernodes
   - Compare metrics (RQ, MI, redundancy) between high vs low connected neurons

### Configuration

```yaml
supernode:
  enabled: true

  # Supernode identification (in intermediate dimension)
  score_metric: "scar_activation_power"  # Options: scar_activation_power, scar_taylor,
                                         #          scar_loss_proxy, rayleigh_quotient,
                                         #          mutual_information, activation_l2_norm
  core_fraction: 0.01                    # Top 1% as supernodes

  # Cross-layer analysis
  cross_layer_analysis: true             # Enable next-layer analysis
  follower_fraction: 0.10                # Top 10% by weight from supernodes

  compute_metrics:
    - "activation"
    - "rayleigh_quotient"
    - "mutual_information"
    - "redundancy"

  compare_by_connection: true            # Compare high vs low connected neurons

  # Target layers (optional)
  # If not specified: uses tracked_layers from config
  # If empty list []: analyzes ALL layers
  # target_layers:
  #   - "model.layers.10.mlp.down_proj"
```

### Generated Plots

| Plot | Description |
|------|-------------|
| `supernode_score_dist_*.png` | Distribution of supernode scores with threshold |
| `supernode_outgoing_weights_*.png` | Histogram of weights from supernodes |
| `supernode_influence_*.png` | Influence of supernodes on output neurons |
| `next_layer_correlation_*.png` | Correlation matrix of high-connection neurons |
| `next_layer_redundancy_hist_*.png` | Redundancy distribution (next layer input) |
| `next_layer_rq_hist_*.png` | RQ distribution (next layer input) |
| `next_layer_mi_hist_*.png` | MI distribution (next layer input) |
| `next_layer_rq_vs_mi_*.png` | RQ vs MI scatter (next layer input) |
| `redundancy_comparison_*.png` | High vs low connected neuron comparison |

### Regenerate Plots

```bash
python scripts/run_experiment.py \
  --config configs/examples/resnet_pruning.yaml \
  --analysis-only \
  --experiment-dir results/previous_run
```
