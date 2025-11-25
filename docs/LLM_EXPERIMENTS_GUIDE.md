# LLM Experiment and Analysis Guide

This document describes all available LLM-related experiments, metrics, pruning strategies, and visualizations in the alignment framework. It also explains how to reproduce common figures and analyses.

---

## Table of Contents

1. [Overview](#overview)
2. [Available Metrics](#available-metrics)
3. [LLM Experiment Pipelines](#llm-experiment-pipelines)
4. [Pruning Strategies](#pruning-strategies)
5. [Visualization Capabilities](#visualization-capabilities)
6. [Reproducing Common Figures](#reproducing-common-figures)
7. [Configuration Reference](#configuration-reference)
8. [Known Limitations and Future Work](#known-limitations-and-future-work)

---

## Overview

The alignment framework provides comprehensive tools for analyzing and pruning large language models (LLMs). The primary entry point for LLM experiments is `LLMAlignmentExperiment` in `src/alignment/experiments/llm_experiments.py`.

### Key Capabilities

- **Importance Score Computation**: Compute per-neuron importance using activation magnitude, Rayleigh quotient, mutual information, redundancy, synergy, and SCAR metrics.
- **Structured Pruning**: Prune MLP layers (gate/up/down projections) and attention heads (Q/K/V/O projections) while maintaining structural consistency.
- **Supernode Protection**: Identify and protect high-importance "core" neurons during pruning.
- **Perplexity Evaluation**: Evaluate model quality before and after pruning.
- **LoRA Repair**: Fine-tune pruned models using LoRA adapters.
- **Visualization**: Generate histograms, heatmaps, scatter plots, and layer-wise distributions.

---

## Available Metrics

### Activation-Based Metrics

| Metric Name | Registry Key | Description |
|-------------|--------------|-------------|
| Activation L2 Norm | `activation_l2_norm` | L2 norm of activations across samples |
| Activation Variance | `activation_variance` | Variance of neuron activations |
| Activation Outlier Index | `activation_outlier_index` | Identifies neurons with outlier activations |

### Alignment Metrics

| Metric Name | Registry Key | Description |
|-------------|--------------|-------------|
| Rayleigh Quotient | `rayleigh_quotient` | Alignment with input covariance structure |
| Delta Alignment | `delta_alignment` | Change in alignment during training |

### Information-Theoretic Metrics

| Metric Name | Registry Key | Description |
|-------------|--------------|-------------|
| Gaussian MI | `mutual_information_gaussian` | Gaussian approximation of mutual information |
| Pairwise Redundancy | `pairwise_redundancy_gaussian` | Redundancy between neuron pairs |
| Gaussian PID Synergy | `gaussian_pid_synergy_mmi` | Synergistic information (MMI-based) |

### Experimental Metrics

| Metric Name | Registry Key | Description |
|-------------|--------------|-------------|
| Language Model Alignment | `language_model_alignment` | Measures neuron contribution to next-token prediction, attention correlation, or semantic coherence. **Note**: This is experimental and uses synthetic projections when real data isn't provided. For practical LLM pruning, use SCAR metrics or activation magnitude instead. |

### SCAR Metrics (Second-Order) — Recommended for LLM Pruning

Computed via `compute_scar_supernode_metrics()`:

| Metric Name | Description |
|-------------|-------------|
| `scar_activation_power` | E[u_i²] - mean squared activation |
| `scar_taylor` | First-order Taylor saliency |
| `scar_curvature` | Rayleigh-style curvature estimate |
| `scar_loss_proxy` | 0.5 × activation_power × curvature |

---

## LLM Experiment Pipelines

### Basic Importance Scoring

```yaml
# configs/examples/llm_alignment.yaml
model_name: "hf_causal_lm"
model_config:
  model_id: "meta-llama/Llama-3.1-8B"
  model_backend: "hf"
  torch_dtype: "bfloat16"

alignment_methods:
  - "activation_l2_norm"
  - "rayleigh_quotient"

alignment_data_num_samples: 100

tracked_layers:
  - "model.layers.*.mlp.gate_proj"
  - "model.layers.*.mlp.up_proj"
```

Run:
```bash
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml
```

### SCAR Supernode Analysis

```yaml
# configs/projects/llm_supernode.yaml
do_scar_metrics: true
scar_num_samples: 100
scar_max_length: 512

supernode:
  enabled: true
  score_metric: "composite"
  core_fraction: 0.12
  protect_core: true
```

### Pruning with Perplexity Evaluation

```yaml
do_pruning_experiments: true
pruning_amounts: [0.1, 0.2, 0.3, 0.4, 0.5]
pruning_alignment_metric: "scar_loss_proxy"
pruning_selection_mode: "low"

do_perplexity_computation: true
evaluation_dataset: "wikitext"
evaluation_num_samples: 100
```

---

## Pruning Strategies

### MLP Pruning

The framework prunes MLP layers by applying consistent masks to:
- `gate_proj` output dimension (rows)
- `up_proj` output dimension (rows)
- `down_proj` input dimension (columns)

This ensures the intermediate dimension remains consistent.

### Attention Head Pruning

Attention heads are pruned as whole units by applying shared masks to:
- `q_proj`, `k_proj`, `v_proj` output dimensions
- `o_proj` input dimension

The mask is computed at the head level and expanded to individual neurons.

### Supernode Protection

When enabled, neurons in the "core" set (top-k by importance) are protected from pruning:

```yaml
supernode:
  enabled: true
  core_fraction: 0.12  # Protect top 12%
  protect_core: true
```

### Dependency-Aware Pruning

For models with structural dependencies (e.g., ResNet skip connections), use:

```yaml
pruning:
  dependency_aware: true
```

---

## Visualization Capabilities

### UnifiedVisualizer Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `plot_layer_scores()` | Violin/box/bar plots of scores across layers | Layer-wise metric distributions |
| `plot_importance_histogram()` | Histogram with top-k annotations | Per-layer importance analysis |
| `plot_neuron_outgoing_weights()` | Outgoing weight distribution for a neuron | Connection analysis |
| `plot_1d_histogram()` | General 1D histogram | Any scalar metric |
| `plot_scatter_2d()` | 2D scatter plot | Metric correlations |
| `plot_heatmap()` | Heatmap visualization | Layer × metric matrices |
| `plot_scar_layer_scores()` | SCAR metric distributions | Supernode analysis |
| `plot_scar_heatmap()` | SCAR metrics across layers | Summary view |
| `plot_pruning_performance()` | Performance vs sparsity curves | Pruning evaluation |
| `plot_radar_chart()` | Multi-metric comparison | Strategy comparison |
| `create_comprehensive_report()` | Full report with all plots | Final results |

### Example Usage

```python
from alignment.analysis.visualization import UnifiedVisualizer

viz = UnifiedVisualizer()

# Plot SCAR metrics
viz.plot_scar_layer_scores(
    scar_scores,
    metric_name="scar_loss_proxy",
    plot_type="violin",
    save_path="plots/scar_loss_proxy.png"
)

# Plot importance histogram
viz.plot_importance_histogram(
    scores=layer_scores["activation_l2_norm"],
    layer_name="model.layers.15.mlp.gate_proj",
    metric_name="activation_l2_norm",
    plots_dir="plots/",
    top_k=10
)

# Scatter plot: activation vs RQ
viz.plot_scatter_2d(
    x=scores["activation_l2_norm"],
    y=scores["rayleigh_quotient"],
    xlabel="Activation L2 Norm",
    ylabel="Rayleigh Quotient",
    title="Activation vs Alignment",
    save_path="plots/activation_vs_rq.png"
)
```

---

## Reproducing Common Figures

### Figure 1: Neuron Importance Distribution

Shows histogram of importance scores for a specific layer.

```python
from alignment.experiments import LLMAlignmentExperiment
from alignment.analysis.visualization import UnifiedVisualizer

# Run experiment
exp = LLMAlignmentExperiment(config)
exp.setup()
scores = exp.compute_importance_scores(num_samples=100)

# Plot
viz = UnifiedVisualizer()
for layer_name, layer_scores in scores.items():
    if "gate_proj" in layer_name:
        viz.plot_importance_histogram(
            scores=layer_scores["activation_l2_norm"],
            layer_name=layer_name,
            metric_name="activation_l2_norm",
            plots_dir="figures/",
            top_k=10
        )
```

### Figure 2: SCAR Metrics Heatmap

Shows activation power, curvature, and loss proxy across all FFN layers.

```yaml
# Config
do_scar_metrics: true
scar_num_samples: 500
```

```python
scar_scores = exp.compute_scar_supernode_metrics()

viz.plot_scar_heatmap(
    scar_scores,
    metrics=["scar_activation_power", "scar_curvature", "scar_loss_proxy"],
    title="SCAR Metrics per Layer",
    save_path="figures/scar_heatmap.png"
)
```

### Figure 3: Sparsity vs Perplexity Curve

Shows model quality degradation as sparsity increases.

```yaml
do_pruning_experiments: true
pruning_amounts: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
do_perplexity_computation: true
```

```python
# After running experiment
results = exp.run()

# Extract data
sparsities = [0.0] + config.pruning_amounts
perplexities = [results["evaluation"]["baseline_perplexity"]]
for sp in config.pruning_amounts:
    perplexities.append(results["pruning_results"][f"sparsity_{sp}"]["perplexity"])

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(sparsities, perplexities, 'o-', linewidth=2, markersize=8)
plt.xlabel("Sparsity")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Sparsity")
plt.grid(True, alpha=0.3)
plt.savefig("figures/sparsity_perplexity.png", dpi=300)
```

### Figure 4: Layer-wise Score Distributions

Violin plots showing score distributions across all tracked layers.

```python
# Collect scores for all layers
layer_to_scores = {}
for layer_name, layer_scores in scores.items():
    if "rayleigh_quotient" in layer_scores:
        layer_to_scores[layer_name] = layer_scores["rayleigh_quotient"]

viz.plot_layer_scores(
    scores=layer_to_scores,
    metric_name="Rayleigh Quotient",
    plot_type="violin",
    save_path="figures/rq_distributions.png"
)
```

### Figure 5: Redundancy vs Synergy Scatter

Shows relationship between neuron redundancy and synergy.

```python
# Requires both metrics computed
viz.plot_scatter_2d(
    x=layer_scores["pairwise_redundancy_gaussian"],
    y=layer_scores["gaussian_pid_synergy_mmi"],
    xlabel="Redundancy",
    ylabel="Synergy",
    title="Redundancy vs Synergy",
    save_path="figures/redundancy_synergy.png"
)
```

### Figure 6: Composite Score Breakdown

Shows contribution of each metric to the composite score.

```python
# Using NodeScoringService
from alignment.services import NodeScoringService

scorer = NodeScoringService(
    metrics={"rq": rq_metric, "mi": mi_metric, "redundancy": red_metric},
    alpha_mi=0.3,
    beta_synergy=0.2,
    gamma_redundancy=0.3,
    delta_rq=0.2
)

composite = scorer.compute_composite_scores(inputs, weights, targets)

# Plot individual components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, vals) in zip(axes.flat, [
    ("RQ", composite.rq),
    ("MI", composite.mi),
    ("Redundancy", composite.redundancy),
    ("Composite", composite.composite)
]):
    ax.hist(vals.cpu().numpy(), bins=50)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("figures/composite_breakdown.png", dpi=300)
```

---

## Configuration Reference

### Complete LLM Experiment Config

```yaml
experiment:
  name: "llm_full_analysis"
  type: "llm_supernode"
  seed: 42
  device: "cuda"

model_name: "hf_causal_lm"
model_config:
  model_id: "meta-llama/Llama-3.1-8B"
  model_backend: "hf"
  device_map: "auto"
  torch_dtype: "bfloat16"

dataset_name: "wikitext"
batch_size: 1
num_workers: 0

tracked_layers:
  - "model.layers.*.mlp.gate_proj"
  - "model.layers.*.mlp.up_proj"
  - "model.layers.*.mlp.down_proj"
  - "model.layers.*.self_attn.q_proj"

alignment_methods:
  - "activation_l2_norm"
  - "activation_outlier_index"
  - "rayleigh_quotient"
  - "pairwise_redundancy_gaussian"

alignment_data_num_samples: 100
alignment_composite_weights:
  activation_l2_norm: 0.35
  activation_outlier_index: 0.25
  rayleigh_quotient: 0.25
  pairwise_redundancy_gaussian: -0.15

supernode:
  enabled: true
  score_metric: "composite"
  core_fraction: 0.12
  min_core_neurons: 64
  protect_core: true

do_scar_metrics: true
scar_num_samples: 100
scar_max_length: 512

pruning:
  enabled: true
  strategy: "alignment"
  target_sparsity: 0.3
  sparsity_levels: [0.1, 0.2, 0.3, 0.4, 0.5]
  structured: true
  dependency_aware: true

pruning_alignment_metric: "scar_loss_proxy"
pruning_selection_mode: "low"

do_perplexity_computation: true
evaluation_dataset: "wikitext"
evaluation_num_samples: 100

log_dir: "./logs"
checkpoint_dir: "./checkpoints"

visualization:
  enabled: true
  format: "png"
  dpi: 300
```

---

## Known Limitations and Future Work

### Current Limitations

1. **Memory Usage**: Large models require significant GPU memory for activation capture. Consider using `device_map: "auto"` for multi-GPU setups.

2. **Batch Size**: LLM experiments typically require `batch_size: 1` due to memory constraints.

3. **Attention Pruning**: Currently prunes entire heads; per-head Q/K/V pruning is not yet implemented.

4. **Custom Loss Functions**: The framework uses HuggingFace's built-in LM loss. Custom objectives (e.g., LLM-U) require manual integration.

### Planned Enhancements

1. **Gradient Checkpointing**: Reduce memory usage during SCAR computation.

2. **Streaming Evaluation**: Evaluate perplexity on larger datasets without loading all data.

3. **Per-Head Analysis**: Compute importance scores per attention head.

4. **Custom LLM Losses**: Add configurable loss functions for specialized objectives.

5. **Quantization Integration**: Combine pruning with quantization for maximum compression.

---

## Quick Reference: Running Experiments

```bash
# Basic importance scoring
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml

# Full supernode analysis with SCAR
python scripts/run_experiment.py --config configs/projects/llm_supernode.yaml

# Analysis-only mode (regenerate plots from existing results)
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml \
    --analysis-only --experiment-dir ./results/previous_run
```

---

## API Reference

### LLMAlignmentExperiment

```python
class LLMAlignmentExperiment(BaseExperiment):
    def setup(self) -> None:
        """Initialize model, tokenizer, and dataset."""
    
    def compute_importance_scores(self, num_samples: int = 1) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute per-layer importance scores using configured metrics."""
    
    def compute_scar_supernode_metrics(self, num_samples: int = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute SCAR-style second-order metrics for FFN layers."""
    
    def apply_pruning(self, sparsity: float, metric: str, mode: str) -> Dict[str, torch.Tensor]:
        """Apply structured pruning to MLP and attention layers."""
    
    def evaluate_perplexity(self, dataset: str, split: str, num_samples: int) -> float:
        """Evaluate model perplexity on a dataset."""
    
    def apply_minimal_repair(self, dataset_name: str, epochs: int, lr: float) -> None:
        """Fine-tune pruned model using LoRA."""
    
    def run(self) -> Dict[str, Any]:
        """Execute full experiment pipeline."""
```

### UnifiedVisualizer

```python
class UnifiedVisualizer:
    def plot_layer_scores(self, scores, metric_name, plot_type, save_path) -> Figure
    def plot_importance_histogram(self, scores, layer_name, metric_name, plots_dir, top_k) -> Path
    def plot_1d_histogram(self, values, title, xlabel, bins, logx, save_path) -> Figure
    def plot_scatter_2d(self, x, y, xlabel, ylabel, title, save_path) -> Figure
    def plot_heatmap(self, data, title, cmap, annotate, save_path) -> Figure
    def plot_scar_layer_scores(self, scar_scores, metric_name, plot_type, save_path) -> Figure
    def plot_scar_heatmap(self, scar_scores, metrics, title, save_path) -> Figure
    def plot_pruning_performance(self, results, metrics, save_path) -> Figure
    def create_comprehensive_report(self, results, output_dir, experiment_name) -> None
```

