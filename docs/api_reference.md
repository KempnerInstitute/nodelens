# API Reference

## Core Classes

### ModelWrapper

Wraps PyTorch models for activation capture and analysis.

```python
from alignment import ModelWrapper

wrapper = ModelWrapper(
    model,                    # PyTorch model
    tracked_layers=None,      # List of layer names or None (auto-detect)
    track_inputs=True,
    track_outputs=True
)

outputs, activations = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights(layers=None)
```

### BaseMetric

All metrics inherit from `BaseMetric`:

```python
metric.requires_inputs   # bool
metric.requires_weights  # bool
metric.requires_outputs  # bool
metric.compute(inputs, weights, outputs, **kwargs)  # Returns scores
```

## Metrics

### Rayleigh Quotient

```python
from alignment.metrics import get_metric

rq = get_metric('rayleigh_quotient',
    relative=True,
    regularization=1e-6
)
scores = rq.compute(inputs, weights)  # [num_neurons]

# Class-conditioned
results = rq.compute_class_conditioned(inputs, weights, targets, return_delta_rq=True)
```

### Redundancy

```python
redundancy = get_metric('pairwise_redundancy_gaussian',
    mode='output_based',
    num_pairs=10,
    aggregation='mean'
)
scores = redundancy.compute(outputs=layer_outputs)
```

### Synergy

```python
synergy = get_metric('synergy_gaussian_mmi', num_pairs=10)
scores = synergy.compute(inputs, weights, targets=labels)
```

## Services

### ActivationCaptureService

```python
from alignment.services import ActivationCaptureService

capture = ActivationCaptureService(model_wrapper)
data = capture.capture(input_batch, layers=['conv1'], include_weights=True)
```

### NodeScoringService

```python
from alignment.services import NodeScoringService

scorer = NodeScoringService(
    metrics={'rq': rq_metric, 'redundancy': redundancy_metric},
    gamma_redundancy=0.4,
    delta_rq=0.3
)
scores = scorer.compute_composite_scores(inputs, weights, targets)
```

### MaskOperations

```python
from alignment.services import MaskOperations

mask = MaskOperations.create_structured_mask(scores, amount=0.5, mode='low')
stats = MaskOperations.get_mask_statistics(mask)
```

## Pruning

### Quick Pruning

```python
from alignment.pruning.orchestrator import prune_with_all_options

result = prune_with_all_options(
    model,
    target_sparsity=0.7,
    distribution='adaptive_sensitivity',
    scoring='composite',
    direction='low',
    val_loader=val_loader,
    eval_fn=evaluate
)
```

### Dependency-Aware Pruning

```python
from alignment.pruning.dependency_aware import DependencyAwarePruning

pruner = DependencyAwarePruning(model)
result = pruner.prune(layer_scores={'conv1': scores1}, amount=0.5, mode='low')
```

## Model Wrappers

### TransformerWrapperEnhanced

```python
from alignment.models.transformer_enhanced import TransformerWrapperEnhanced

wrapper = TransformerWrapperEnhanced(
    transformer_model,
    track_qkv=True,
    track_per_head=True
)
head_repr = wrapper.extract_attention_heads(attn_output)
```

## Experiments

### GeneralAlignmentExperiment

```python
from alignment.experiments import GeneralAlignmentExperiment

experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()
```

### LLMAlignmentExperiment

```python
from alignment.experiments import LLMAlignmentExperiment

experiment = LLMAlignmentExperiment(config)
experiment.setup()

# Compute importance scores
scores = experiment.compute_importance_scores(num_samples=100)

# Compute SCAR metrics
scar_scores = experiment.compute_scar_supernode_metrics()

# Analyze supernode connections
supernode_analysis = experiment.analyze_supernode_connections(
    scar_scores=scar_scores,
    supernode_fraction=0.01,           # Top 1% as supernodes
    follower_fraction=0.10,            # Top 10% by weight from supernodes
    supernode_metric="scar_activation_power",  # Metric for supernode identification
    cross_layer_analysis=True,         # Enable next-layer analysis
    compute_metrics=["activation", "rayleigh_quotient", "mutual_information", "redundancy"],
    compare_by_connection=True,        # Compare high vs low connected neurons
    target_layers=["model.layers.10.mlp.down_proj"],  # Specific layers (None = use tracked_layers, [] = all)
    plots_dir="./plots"
)
```

## Visualization

### UnifiedVisualizer

```python
from alignment.analysis.visualization import UnifiedVisualizer

viz = UnifiedVisualizer()

# Basic plots
viz.plot_layer_scores(scores, metric_name, plot_type='violin', save_path='plot.png')
viz.plot_importance_histogram(scores, layer_name, metric_name, plots_dir)
viz.plot_scatter_2d(x, y, xlabel, ylabel, title, save_path)
viz.plot_heatmap(data, title, cmap, save_path)
viz.plot_pruning_performance(results, metrics, save_path)

# Supernode analysis plots
viz.plot_supernode_activation_distribution(
    activation_values, threshold_value, threshold_percentile, 
    layer_name, metric_name="scar_activation_power", save_path=path
)
viz.plot_outgoing_weights_distribution(weights, layer_name, save_path=path)
viz.plot_supernode_influence(influence_values, threshold_value, threshold_percentile, layer_name, save_path=path)
viz.plot_correlation_matrix(corr_matrix, title, xlabel, ylabel, save_path=path)
viz.plot_1d_histogram(values, xlabel, ylabel, title, vline, vline_label, color, save_path=path)
viz.plot_rq_vs_mi(rq_scores, mi_scores, redundancy_scores, layer_name, save_path=path)
viz.plot_redundancy_comparison(high_redundancy, low_redundancy, high_mean, low_mean, layer_name, save_dir)
```

## Configuration Parameters

### Metric Parameters

**RayleighQuotient**
- `relative` (bool): Normalize by trace
- `regularization` (float): Diagonal regularization

**PairwiseRedundancyGaussian**
- `mode` (str): 'output_based' or 'covariance_based'
- `num_pairs` (int): Partners to sample
- `aggregation` (str): 'mean', 'median', 'max', 'sum'

### Pruning Parameters

**Strategy**: 'magnitude', 'alignment', 'composite', 'movement', 'adaptive'

**Distribution**: 'uniform', 'global_threshold', 'adaptive_sensitivity', 'cascading'

**Direction**: 'low' (prune unimportant), 'high' (ablation), 'random' (baseline)
