# API Reference

## Core Classes

### ModelWrapper

Wraps PyTorch models for activation capture and analysis.

```python
from alignment import ModelWrapper

wrapper = ModelWrapper(
    model,                    # PyTorch model
    tracked_layers=None,      # Layer names or None (auto-detect)
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

---

## Metrics

### Rayleigh Quotient

```python
from alignment.metrics import get_metric

rq = get_metric('rayleigh_quotient',
    relative=True,
    regularization=1e-6
)
scores = rq.compute(inputs, weights)
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

### Synergy (Continuous Target)

```python
from alignment.metrics.information import SynergyContinuousTarget

synergy = SynergyContinuousTarget(
    target_type='logit_margin',  # or 'correct_logit', 'logit_pc1'
    num_pairs=10,
    sampling_strategy='top_k'
)
scores = synergy.compute(outputs=activations, logits=logits, labels=labels)
```

---

## Clustering Analysis

### MetricSpaceClustering

Clusters channels in (RQ, Redundancy, Synergy) space.

```python
from alignment.analysis.clustering import MetricSpaceClustering, ClusterResult

clusterer = MetricSpaceClustering(n_clusters=4, seed=42)
result = clusterer.fit(rq_scores, redundancy_scores, synergy_scores, layer_name="conv1")

# Result attributes
result.labels        # Cluster assignments [n_channels]
result.centroids     # Cluster centers [n_clusters, 3]
result.silhouette    # Silhouette score
result.type_mapping  # {cluster_id: 'critical'|'redundant'|'synergistic'|'background'}
result.type_counts   # {'critical': N, ...}
```

### CrossLayerHaloAnalysis

Analyzes downstream dependencies via halos.

```python
from alignment.analysis.clustering import CrossLayerHaloAnalysis, HaloResult

halo_analyzer = CrossLayerHaloAnalysis(percentile=90.0, use_activation_weight=True)

# Compute influence matrix
influence = halo_analyzer.compute_influence(weights, activations)

# Find halo for a cluster
halo_indices, rel_influence = halo_analyzer.find_halo(influence, cluster_indices)

# Analyze halo properties
halo_result = halo_analyzer.analyze_halo(
    halo_indices, next_layer_redundancy, next_layer_synergy,
    layer_name="layer2", cluster_name="critical"
)
```

### CascadeAnalysis

Validates importance via channel ablation.

```python
from alignment.analysis import CascadeAnalysis, DamagePrediction

cascade = CascadeAnalysis(model, test_loader, device="cuda")
baseline = cascade.baseline()

# Ablate specific channels
result = cascade.ablate(layer_name="conv1", indices=[0, 5, 10])
# result.accuracy_drop, result.loss_increase

# Test by cluster type
results = cascade.by_cluster(layer_name, labels, type_mapping, n_rm=5)
```

---

## Experiments

### ClusterAnalysisExperiment

General cluster-based analysis for any architecture.

```python
from alignment.experiments import ClusterAnalysisExperiment, ClusterAnalysisConfig

config = ClusterAnalysisConfig(
    model_name="resnet18",
    dataset_name="cifar10",
    n_clusters=4,
    synergy_target="logit_margin",
    halo_percentile=90.0,
    device="cuda"
)

experiment = ClusterAnalysisExperiment(config, model, train_loader, test_loader)
results = experiment.run()
experiment.generate_figures()
```

### LLMAlignmentExperiment

LLM-specific analysis with SCAR metrics.

```python
from alignment.experiments import LLMAlignmentExperiment

experiment = LLMAlignmentExperiment(config)
experiment.setup()

scores = experiment.compute_importance_scores(num_samples=100)
scar_scores = experiment.compute_scar_supernode_metrics()
masks = experiment.apply_pruning(sparsity=0.3, metric="scar_loss_proxy", mode="low")
perplexity = experiment.evaluate_perplexity("wikitext", "test", num_samples=100)
```

### GeneralAlignmentExperiment

Vision model alignment analysis.

```python
from alignment.experiments import GeneralAlignmentExperiment

experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()
```

---

## Visualization

### Cluster Plots

```python
from alignment.analysis.visualization import (
    plot_metric_scatter,
    plot_cluster_evolution,
    plot_influence_matrix,
    plot_cascade_test,
    plot_halo_properties
)

# Metric space scatter (RQ vs Red, RQ vs Syn, Red vs Syn)
plot_metric_scatter(rq, redundancy, synergy, labels, type_mapping, 
                   layer_name, save_path)

# Cluster composition across depth
plot_cluster_evolution(layer_results, save_path)

# Cross-cluster influence heatmap
plot_influence_matrix(flow_dict, layer_name, save_path)

# Cascade damage by cluster type
plot_cascade_test(cascade_results, save_path)
```

### UnifiedVisualizer

```python
from alignment.analysis.visualization import UnifiedVisualizer

viz = UnifiedVisualizer()
viz.plot_layer_scores(scores, metric_name, plot_type='violin', save_path='plot.png')
viz.plot_importance_histogram(scores, layer_name, metric_name, plots_dir)
viz.plot_scatter_2d(x, y, xlabel, ylabel, title, save_path)
viz.plot_heatmap(data, title, cmap, save_path)
```

---

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

---

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

---

## Configuration Parameters

### Metric Parameters

**RayleighQuotient**
- `relative` (bool): Normalize by trace
- `regularization` (float): Diagonal regularization

**PairwiseRedundancyGaussian**
- `mode` (str): 'output_based' or 'covariance_based'
- `num_pairs` (int): Partners to sample
- `aggregation` (str): 'mean', 'median', 'max', 'sum'

**SynergyContinuousTarget**
- `target_type` (str): 'logit_margin', 'correct_logit', 'logit_pc1'
- `num_pairs` (int): Partner neurons per channel
- `sampling_strategy` (str): 'random', 'top_k', 'all'

### Clustering Parameters

**MetricSpaceClustering**
- `n_clusters` (int): Number of clusters (default: 4)
- `seed` (int): Random seed

**CrossLayerHaloAnalysis**
- `percentile` (float): Halo membership threshold (default: 90.0)
- `use_activation_weight` (bool): Weight influence by activation std

### Pruning Parameters

**Strategy**: 'magnitude', 'alignment', 'composite', 'cluster_aware', 'random'

**Distribution**: 'uniform', 'global_threshold', 'adaptive_sensitivity'

**Direction**: 'low' (prune unimportant), 'high' (ablation)
