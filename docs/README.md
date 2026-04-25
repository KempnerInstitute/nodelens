# Documentation

NodeLens is the public project name. The Python package is imported as
`nodelens`.

## Guides

- [Usage Guide](usage.md) - Running experiments and configuration
- [API Reference](api_reference.md) - Core classes and functions
- [LLM Guide](llm_guide.md) - LLM-specific analysis and pruning
- [Metric Consistency](METRIC_CONSISTENCY.md) - Theory-code verification
- [Architecture](ARCHITECTURE.md) - Library layout and data flow

## Configuration

- [Template](../configs/template.yaml) - Complete parameter reference
- [Vision pruning configs](../configs/vision_prune/) - Vision pruning + clustering configs
- [LLM pruning configs](../configs/prune_llm/) - LLM pruning + analysis configs
- [Examples](../configs/examples/) - Example configurations

## Quick Reference

### Experiment Types

| Type | Description |
|------|-------------|
| `alignment_analysis` | General alignment metrics for vision models |
| `llm_alignment` | LLM channel metrics and structured FFN pruning |
| `cluster_analysis` | Metric-space clustering with halo analysis |

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `MetricSpaceClustering` | `analysis.clustering` | Cluster channels by functional type |
| `CrossLayerHaloAnalysis` | `analysis.clustering` | Track downstream dependencies |
| `CascadeAnalysis` | `analysis` | Validate importance via ablation |
| `LLMAlignmentExperiment` | `experiments` | LLM analysis runner |
| `ClusterAnalysisExperiment` | `experiments` | Cluster analysis runner |

### Running Experiments

```bash
# Vision/general analysis
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# LLM analysis
python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml

# Cluster-based analysis
python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml

# Post-hoc analysis
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots
```
