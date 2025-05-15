# Alignment Metrics System

This document provides an overview of the neural network alignment metrics system, its architecture, and available metrics.

## Overview

The alignment metrics system measures various properties of neural networks, focusing on the relationship between weight matrices and input activations. These metrics can be used for various purposes including pruning experiments, understanding network properties, and analyzing network training dynamics.

## Architecture

The system is built around a central registry of metrics and a dispatch mechanism for computing them:

- `ALIGNMENT_METRICS_REGISTRY`: Maps metric names to metric functions
- `get_metric()`: Returns a metric object configured with the requested parameters. Metric-specific parameters (e.g., `scale_by_norm` for RQ, `bins` for MI) are typically passed here.
- `_AlignmentMetricImpl`: Implements the metric calculation with proper dispatch logic.

When running experiments via the experiment runner, some metric configurations are also sourced from the `alignment_settings` block in your main configuration file (see [Configuration Documentation](configuration.md) for details).

## Available Metrics

### Rayleigh Quotient (RQ) Metrics

The Rayleigh Quotient measures the alignment between weight vectors and input data.

| Metric Name | Description |
|-------------|-------------|
| `rayleigh_quotient` / `rq` | Standard Rayleigh Quotient calculation |
| `rq_alt_denom` | Alternative RQ calculation with different denominator and scaling |

Usage:
```python
from alignment.metrics import get_metric

# Standard RQ
rq_metric = get_metric("rq", scale_by_norm=True)
scores = rq_metric.compute_per_node_scores(layer_inputs, layer_weights)

# Alternative RQ
rq_alt_metric = get_metric("rq_alt_denom")
scores = rq_alt_metric.compute_per_node_scores(layer_inputs, layer_weights)
```

### Mutual Information (MI) Metrics

Mutual Information metrics quantify the shared information between neurons and various reference signals.

| Metric Name | Description |
|-------------|-------------|
| `mi_gaussian` / `mi_g` | MI using Gaussian approximation |
| `mi_direct` / `mi_bin` | MI using direct binning approach |
| `mi_proj_vs_mean_input` | MI between neuron's projected input and mean input feature activity |

Usage:
```python
# MI using Gaussian approximation
mi_g_metric = get_metric("mi_gaussian")
scores = mi_g_metric.compute_per_node_scores(layer_outputs=outputs)

# MI using direct binning
mi_bin_metric = get_metric("mi_direct")
scores = mi_bin_metric.compute_per_node_scores(
    layer_outputs=outputs, 
    target_outputs=targets,  # Optional
    bins=20  # Optional, default=10
)

# MI projected vs mean input
mi_proj_metric = get_metric("mi_proj_vs_mean_input")
scores = mi_proj_metric.compute_per_node_scores(
    layer_inputs=inputs,
    layer_weights=weights,
    bins=30 # Default is 30 for this specific metric
)
```

### Redundancy Metrics

Redundancy metrics measure information overlap between neurons or input features.

| Metric Name | Description |
|-------------|-------------|
| `redundancy_gaussian` / `red_g` | Measures redundancy between neurons using Gaussian approximation |
| `node_redundancy` | Measures redundancy between input features based on correlation |

Usage:
```python
# Neuron redundancy
red_metric = get_metric("redundancy_gaussian")
scores = red_metric.compute_per_node_scores(
    layer_inputs=inputs,
    layer_weights=weights
)

# Input feature redundancy
node_red_metric = get_metric("node_redundancy")
scores = node_red_metric.compute_per_node_scores(layer_inputs=inputs)
```

### Partial Information Decomposition (PID) Metrics

PID metrics decompose the information that a pair of neurons convey about an input.

| Metric Name | Description |
|-------------|-------------|
| `pid_shared_info` / `pid_si` | Shared information component |
| `pid_unique_info_neuron` / `pid_uiy` / `pid_ui1` | Unique information in neuron 1 |
| `pid_unique_info_other` / `pid_uiz` / `pid_ui2` | Unique information in neuron 2 |
| `pid_synergy_info` / `pid_ci` | Synergistic information component |

Usage:
```python
# Shared information component
pid_si_metric = get_metric("pid_si")
scores = pid_si_metric.compute_per_node_scores(
    layer_inputs=inputs,  # Or layer_outputs depending on exact PID formulation
    layer_outputs=outputs, # Or target_outputs
    bins=20  # Optional, default=10
)
```

### Weight Similarity Metrics

These metrics measure similarity between weight vectors.

| Metric Name | Description |
|-------------|-------------|
| `weight_cosine_similarity` | Cosine similarity between weight vectors |
| `weight_dot_similarity` | Dot product similarity between weight vectors |
| `weight_euclidean_distance` | Euclidean distance between weight vectors |

Usage:
```python
# Weight cosine similarity
cosine_metric = get_metric("weight_cosine_similarity")
sim_matrix = cosine_metric.compute_per_node_scores(layer_weights=weights)
```

### NullSpace Metrics (Details TBD)

*Valid metric name: `NullSpace`*

This metric is intended to measure properties related to the null space of weight matrices.
(Further details, specific parameters, and usage examples need to be documented based on its full implementation and registration in `alignment.metrics.py`.)

| Metric Name | Description |
|-------------|-------------|
| `nullspace` (Tentative) | Measures properties related to the null space of weight matrices. |

Usage:
```python
# Usage example TBD
# nullspace_metric = get_metric("NullSpace", **specific_params)
# scores = nullspace_metric.compute_per_node_scores(...)
```

### Rank Alignment Metrics (Details TBD)

*Valid metric name: `RankAlignment`*

This metric is intended to measure alignment of singular vectors or other rank-related properties between matrices (e.g., weight matrices or activation covariance matrices).
(Further details, specific parameters, and usage examples need to be documented based on its full implementation and registration in `alignment.metrics.py`.)

| Metric Name | Description |
|-------------|-------------|
| `rank_alignment` (Tentative) | Measures rank-related alignment properties. |

Usage:
```python
# Usage example TBD
# rank_metric = get_metric("RankAlignment", **specific_params)
# scores = rank_metric.compute_per_node_scores(...)
```

## Common Parameters

When using `get_metric(...).compute_per_node_scores(...)`, many metrics accept:

- `layer_inputs: Optional[torch.Tensor]`: Input activations to a layer.
- `layer_weights: Optional[torch.Tensor]`: Weight matrix of a layer.
- `layer_outputs: Optional[torch.Tensor]`: Output activations from a layer.
- `device: Optional[Union[str, torch.device]]`: Computation device.
- `min_samples_for_cov: int = 2`: Minimum samples for covariance-based metrics.
- `target_outputs: Optional[torch.Tensor]`: Target outputs, e.g., for some MI variants.
- `bins: int = 10`: Default number of bins for histogram-based metrics (can be overridden).
- `verbose: bool = False`: Print detailed debugging information.
- `force_cpu_for_large_metric_ops: bool = False` (default in `get_metric` context, but can be `True` from `AlignmentConfig`): Offload large computations to CPU.
- Additional `**metric_specific_kwargs` can be passed.

## Experiment-Level Metric Configurations

When metrics are computed via the experiment runner (using `ExperimentConfig`), parameters from the `alignment_settings` block in your YAML configuration file are also used. These include:

- `metric: str`: The primary metric to compute (e.g., "RQ", "MI").
- `scale_by_norm: bool`: Specifically for RQ, whether to scale by norm (passed to `get_metric`).
- `cnn_mode: str`: For CNNs, how feature maps are processed ("unfold", "patchwise", etc.). This affects how `layer_inputs` are shaped for metric computation.
- `cnn_rq_aggregation_op: str`: For RQ on CNNs, how scores are aggregated ("mean", "max", etc.).
- `run_progressive: bool`: (Likely relates to experiment flow rather than a direct metric parameter).
- `run_eigenvector: bool`: (Likely relates to RQ computation, possibly for eigenvector analysis).
- `force_cpu_for_large_metric_ops: bool`: Global override for forcing CPU.

Refer to the [Configuration Documentation](configuration.md#alignment-configuration-alignmentconfig) for full details on these settings.

## High-Level Utility Functions

The system provides high-level functions for computing metrics across networks:

- `compute_metrics_for_layers()`: Computes metrics for multiple layers at once
- `compute_all_node_scores()`: Computes metrics efficiently across the whole network
- `compute_pairwise_metric()`: Computes metrics between pairs of data

## Example Usage

Computing metrics for a full network:

```python
from alignment.metrics import compute_all_node_scores, get_metric

# Define metrics configurations
metric_configs = [
    {"name": "rq", "scale_by_norm": True},
    {"name": "mi_gaussian"},
    {"name": "redundancy_gaussian"}
]

# Compute metrics
scores = compute_all_node_scores(
    model=my_model,
    metric_configs=metric_configs,
    device="cuda",
    data_loader=train_loader,
    num_batches=5
)
```

## Notes on Metric Selection

Different metrics serve different purposes:

- **RQ metrics**: Good for measuring weight-input alignment; useful for pruning
- **MI metrics**: Measure information content; useful for analyzing feature importance
- **Redundancy metrics**: Identify information overlap; useful for network compression
- **PID metrics**: Provide deeper analysis of information sharing; useful for understanding neuron cooperation
- **Weight similarity metrics**: Identify similar neurons; useful for understanding network structure

Choose metrics appropriate for your specific analysis task. 