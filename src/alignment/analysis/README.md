# Analysis Module

Result analysis, visualization, clustering, and reporting.

## Components

### Core Analysis
- `AnalysisRunner` - Unified entry point for analysis tasks
- `ResultAggregator` - Result collection and summarization

### Clustering
- `MetricSpaceClustering` - K-means in (RQ, Redundancy, Synergy) space
- `CrossLayerHaloAnalysis` - Track downstream channel dependencies
- `CascadeAnalysis` - Validate importance via ablation
- `DamagePrediction` - Correlate scores with true damage

### Visualization
- `UnifiedVisualizer` - General plot generation
- `plot_metric_scatter` - Cluster scatter plots
- `plot_cluster_evolution` - Composition across depth
- `plot_influence_matrix` - Cross-layer influence heatmaps
- `plot_cascade_test` - Damage by cluster type

### Reporting
- `UnifiedReporter` - Report generation (HTML, Markdown, JSON)

## Usage

### Cluster Analysis

```python
from alignment.analysis.clustering import MetricSpaceClustering, CrossLayerHaloAnalysis

# Cluster channels
clusterer = MetricSpaceClustering(n_clusters=4)
result = clusterer.fit(rq, redundancy, synergy, "layer1")

# Analyze halos
halo_analyzer = CrossLayerHaloAnalysis(percentile=90.0)
halo_idx, influence = halo_analyzer.find_halo(weights, cluster_indices)
```

### Cascade Testing

```python
from alignment.analysis import CascadeAnalysis

cascade = CascadeAnalysis(model, test_loader, device="cuda")
cascade.baseline()
results = cascade.by_cluster(layer_name, labels, type_mapping, n_rm=5)
```

### Visualization

```python
from alignment.analysis.visualization import plot_metric_scatter, plot_cluster_evolution

plot_metric_scatter(rq, red, syn, labels, type_map, "layer1", "scatter.png")
plot_cluster_evolution(layer_results, "evolution.png")
```

### General Analysis

```python
from alignment.analysis import AnalysisRunner, AnalysisConfig

config = AnalysisConfig(
    results_dir="./results",
    output_dir="./plots",
    analyses=["histograms", "pruning_curves"],
)
runner = AnalysisRunner(config)
runner.run()
```

## Available Analyses

- `histograms` - Importance score distributions
- `scatter_plots` - Metric correlations
- `heatmaps` - Layer-metric heatmaps
- `pruning_curves` - Sparsity vs performance
- `scar_analysis` - SCAR metrics (LLM)
- `cluster_scatter` - Metric space cluster plots
- `cluster_evolution` - Cluster composition by depth
- `cascade_test` - Ablation damage analysis
