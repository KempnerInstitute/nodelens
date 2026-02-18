# LLM Analysis Guide

Guide for analyzing and pruning large language models.

## Overview

The `LLMAlignmentExperiment` class provides tools for:

- Computing per-neuron importance scores
- SCAR-style second-order metrics
- Structured MLP and attention head pruning
- Supernode detection and protection
- Perplexity evaluation

## Quick Start

```bash
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml
```

## Configuration

```yaml
experiment:
  name: "llm_analysis"
  type: "llm_alignment"

model_name: "hf_causal_lm"
model_config:
  model_id: "meta-llama/Llama-3.1-8B"
  model_backend: "hf"
  torch_dtype: "bfloat16"

alignment_methods:
  - "activation_l2_norm"
  - "rayleigh_quotient"

tracked_layers:
  - "model.layers.*.mlp.up_proj"
  - "model.layers.*.mlp.down_proj"

do_scar_metrics: true
scar_num_samples: 100
scar_max_length: 512

supernode:
  enabled: true
  core_fraction: 0.12
  protect_core: true
```

## Available Metrics

### Activation Metrics

| Metric | Description |
|--------|-------------|
| `activation_l2_norm` | L2 norm of activations |
| `activation_variance` | Activation variance |
| `activation_outlier_index` | Outlier detection |

### SCAR Metrics

Computed via `compute_scar_supernode_metrics()`:

| Metric | Description |
|--------|-------------|
| `scar_activation_power` | Mean squared activation E[u_i^2] |
| `scar_taylor` | First-order Taylor saliency |
| `scar_curvature` | Rayleigh-style curvature |
| `scar_loss_proxy` | 0.5 x activation_power x curvature |

## Pruning

### MLP Pruning

Prunes gate_proj, up_proj (output dims) and down_proj (input dims) together:

```yaml
pruning:
  enabled: true
  algorithms: ["alignment"]
  sparsity_levels: [0.1, 0.2, 0.3]
  alignment_metric: "scar_loss_proxy"
  selection_mode: "low"
  structured: true
```

### Attention Head Pruning

Prunes entire attention heads by applying shared masks to Q/K/V/O projections.

### Supernode Protection

Protects high-importance neurons from pruning:

```yaml
supernode:
  enabled: true
  core_fraction: 0.12
  protect_core: true
```

## Supernode Analysis

The framework analyzes supernode connections across transformer layers.

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

### Configuration

```yaml
supernode:
  enabled: true
  
  # Supernode identification (in intermediate dimension)
  score_metric: "scar_activation_power"  # Options: scar_activation_power, scar_taylor,
                                         #          scar_loss_proxy, rayleigh_quotient,
                                         #          mutual_information, activation_l2_norm
  core_fraction: 0.01                    # Top 1% as supernodes
  protect_core: true                     # Protect during pruning
  
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
  # - If not specified: uses tracked_layers from main config
  # - If empty list []: analyzes ALL layers with SCAR scores
  # target_layers:
  #   - "model.layers.10.mlp.down_proj"
  #   - "model.layers.15"  # Pattern matching
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

### Understanding Cross-Layer Analysis

The cross-layer analysis traces how supernodes in layer N influence layer N+1:

1. **Supernodes** are identified in the intermediate dimension (14336 neurons inside the FFN)
2. **Outgoing weights** from supernodes are traced through `down_proj` to the hidden dimension (4096)
3. **High-connection neurons** are positions in the hidden dimension that receive large weights from supernodes
4. These positions become **inputs to the next transformer block**
5. Metrics (RQ, MI, redundancy) are computed for these high-connection positions

## Programmatic Usage

```python
from alignment.experiments import LLMAlignmentExperiment

experiment = LLMAlignmentExperiment(config)
experiment.setup()

# Compute importance scores
scores = experiment.compute_importance_scores(num_samples=100)

# Compute SCAR metrics
scar_scores = experiment.compute_scar_supernode_metrics()

# Apply pruning
masks = experiment.apply_pruning(sparsity=0.3, metric="scar_loss_proxy", mode="low")

# Evaluate
perplexity = experiment.evaluate_perplexity("wikitext", "test", num_samples=100)
```

## Visualization

```python
from alignment.analysis.visualization import UnifiedVisualizer

viz = UnifiedVisualizer()

# SCAR metrics
viz.plot_scar_layer_scores(scar_scores, metric_name="scar_loss_proxy")
viz.plot_scar_heatmap(scar_scores, metrics=["scar_activation_power", "scar_loss_proxy"])

# Importance histograms
viz.plot_importance_histogram(scores, layer_name, metric_name, plots_dir)
```

## Memory Considerations

- Use `batch_size: 1` for large models
- Use `device_map: "auto"` for multi-GPU
- Use `torch_dtype: "bfloat16"` to reduce memory

## Example Workflow

```bash
# 1. Compute importance scores
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml

# 2. Results saved to results/experiment_YYYYMMDD_HHMMSS/
# 3. Plots generated in results/.../plots/
```

