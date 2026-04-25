# API Reference

This page summarizes the main public APIs. For full experiments, the preferred
entry point is still the YAML runner:

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

## Model Wrapping

`ModelWrapper` wraps a PyTorch model and captures activations from selected
layers.

```python
from nodelens import ModelWrapper

wrapper = ModelWrapper(model, tracked_layers=["layer1.0.conv1"])
outputs, activations = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights(layers=["layer1.0.conv1"])
```

## Metrics

Metrics are created through the registry.

```python
from nodelens.metrics import get_metric, list_metrics

print(list_metrics())

rq = get_metric("rayleigh_quotient", relative=True, regularization=1e-6)
scores = rq.compute(inputs=layer_inputs, weights=layer_weights)
```

Common metric families:

| Family | Examples |
|--------|----------|
| Activation | `activation_l2_norm`, `activation_variance`, `activation_outlier_index` |
| Alignment | `rayleigh_quotient`, `delta_alignment` |
| Information | `mutual_information_gaussian`, `pairwise_redundancy_gaussian`, `average_redundancy` |
| Synergy | `gaussian_pid_synergy_mmi`, `synergy_gaussian_mmi` |
| Gradient | `taylor_saliency`, `gradient_alignment` |

LLM experiments can also write SCAR-specific score keys such as
`scar_activation_power`, `scar_taylor`, `scar_curvature`, and
`scar_loss_proxy`. Those are produced by the LLM experiment pipeline rather
than by the generic metric registry.

## Clustering And Halo Analysis

Metric-space clustering groups channels by score profile.

```python
from nodelens.analysis.clustering import MetricSpaceClustering

clusterer = MetricSpaceClustering(n_clusters=4, seed=42)
result = clusterer.fit(
    rq_scores,
    redundancy_scores,
    synergy_scores,
    layer_name="conv1",
)
```

`CrossLayerHaloAnalysis` estimates downstream influence and local dependency
structure.

```python
from nodelens.analysis.clustering import CrossLayerHaloAnalysis

halo = CrossLayerHaloAnalysis(percentile=90.0, use_activation_weight=True)
influence = halo.compute_influence(weights, activations)
halo_indices, rel_influence = halo.find_halo(influence, cluster_indices)
```

## Pruning

Use the pruning registry for direct scripts.

```python
from nodelens.pruning import PruningConfig, get_pruning_strategy, list_pruning_strategies

print(list_pruning_strategies())

config = PruningConfig(amount=0.5, pruning_mode="low")
strategy = get_pruning_strategy("magnitude", config=config)
mask = strategy.prune(layer, amount=0.5)
```

For full-model pruning, prefer config-driven experiments under
`configs/vision_prune/` and `configs/prune_llm/`, because they handle layer
selection, dependency constraints, evaluation, and logging.

## Experiments

Load configs with `load_config` and instantiate the matching experiment family
when writing custom scripts.

```python
from nodelens.configs.config_loader import load_config
from nodelens.experiments import (
    ClusterAnalysisExperiment,
    GeneralAlignmentExperiment,
    LLMAlignmentExperiment,
)

config = load_config("configs/examples/mnist_basic.yaml")

if config.experiment_type == "llm_alignment":
    experiment = LLMAlignmentExperiment(config)
elif config.experiment_type == "cluster_analysis":
    experiment = ClusterAnalysisExperiment(config)
else:
    experiment = GeneralAlignmentExperiment(config)

results = experiment.run()
```

## Output Analysis

Most experiments write:

- `experiment_config.yaml`
- `logs/`
- `results/`
- `figures/`
- `analysis/`

Use `scripts/run_analysis.py` for post-hoc analysis when an experiment has
already produced a results directory.

```bash
python scripts/run_analysis.py \
  --results-dir outputs/my_run \
  --output-dir outputs/my_run/analysis_extra
```
