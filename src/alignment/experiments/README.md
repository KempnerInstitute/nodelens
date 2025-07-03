# Experiments Module

Framework for running structured neural network alignment experiments.

## Available Experiments

### GeneralAlignmentExperiment
Complete pipeline for alignment analysis including training, metric computation, and pruning.

```python
from alignment.experiments import GeneralAlignmentExperiment

# From YAML configuration
experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()

# Programmatic configuration
from alignment.experiments import GeneralAlignmentConfig
config = GeneralAlignmentConfig(
    dataset_name="mnist",
    model_name="mlp",
    alignment_metrics=["rayleigh_quotient"],
    pruning_strategy="magnitude"
)
experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

### Specialized Experiments
- **LayerIsolatedPruningExperiment** - Analyze individual layer pruning effects
- **GlobalDropoutExperiment** - Global pruning across all layers
- **CascadingLayerPruningExperiment** - Sequential layer pruning analysis
- **EigenvectorDropoutExperiment** - Eigenvector-based pruning

## Base Classes

All experiments inherit from `BaseExperiment` which provides:
- Configuration management
- Model wrapping for metrics
- Result tracking and checkpointing
- Device management

## Creating Custom Experiments

```python
from alignment.experiments import BaseExperiment, ExperimentConfig

class MyExperiment(BaseExperiment):
    def run(self) -> Dict[str, Any]:
        # Your experiment logic
        metrics = self.compute_metrics(data)
        results = {"metrics": metrics}
        self.save_results(results)
        return results
```

## Configuration

Use YAML files for reproducible experiments:

```yaml
name: "my_experiment"
model_name: "resnet18"
dataset_name: "cifar10"
metrics: ["rayleigh_quotient", "spectral_gap"]
device: "cuda"
```

## Integration

Works seamlessly with the analysis module for automatic result processing and reporting. 