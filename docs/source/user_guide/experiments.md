# Experiments Guide

This guide provides detailed information about the experiment types available in the Neural Network Alignment framework.

## Overview

The framework includes several experiment types designed to study different aspects of neural network pruning and alignment:

1. **Progressive Dropout**: Gradually increases pruning percentage
2. **Eigenvector-based Pruning**: Uses eigenvector analysis for pruning decisions
3. **Layer-Isolated Pruning**: Analyzes the impact of pruning individual layers
4. **Cascading Layer Pruning**: Sequential pruning across layers

## Progressive Dropout Experiments

Progressive dropout experiments gradually increase the pruning percentage to study the relationship between network sparsity and performance.

### Basic Usage

```python
from alignment_refactor.experiments.progressive_dropout import ProgressiveDropoutExperiment
from alignment_refactor.experiments.base import ExperimentConfig

config = ExperimentConfig(
    name="progressive_pruning_experiment",
    model_name="mlp",
    dataset_name="mnist",
    model_config={
        "hidden_dims": [300, 200, 100],
        "dropout_rate": 0.5
    },
    metrics=["rayleigh_quotient"],
    dropout_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    batch_size=128,
    device="cuda"
)

experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()
```

### Results Structure

```python
{
    'dropout_fractions': [0.0, 0.1, 0.2, ...],
    'accuracies': {
        'strategy_name': [acc1, acc2, ...],
        ...
    },
    'metrics': {
        'layer_name': {
            'metric_name': scores,
            ...
        }
    }
}
```

### Pruning Strategies

The experiment supports multiple pruning strategies:

- **magnitude**: Prune neurons with smallest weight magnitudes
- **gradient**: Prune based on gradient information
- **random**: Random pruning (baseline)
- **metric-based**: Use alignment metrics (RQ, MI) for pruning

## Eigenvector-based Pruning

This experiment type uses eigenvector analysis of the weight covariance matrix to identify important neurons.

### Basic Usage

```python
from alignment_refactor.experiments.eigenvector import EigenvectorDropoutExperiment

config = ExperimentConfig(
    name="eigenvector_pruning",
    model_name="cnn2p2",
    dataset_name="cifar10",
    num_components=50,  # Number of eigenvectors to consider
    variance_threshold=0.95  # Cumulative variance explained
)

experiment = EigenvectorDropoutExperiment(config)
results = experiment.run()
```

### Key Parameters

- `num_components`: Number of principal components to retain
- `variance_threshold`: Minimum cumulative variance to preserve
- `layer_wise`: Whether to apply eigenvector analysis per layer

## Layer-Isolated Pruning

This experiment analyzes the sensitivity of individual layers to pruning.

### Basic Usage

```python
from alignment_refactor.experiments.layer_isolated import LayerIsolatedPruningExperiment

config = ExperimentConfig(
    name="layer_sensitivity_analysis",
    model_name="mlp",
    dataset_name="mnist",
    pruning_percentages=[0.1, 0.3, 0.5, 0.7, 0.9],
    metrics=["rayleigh_quotient", "mutual_information"]
)

experiment = LayerIsolatedPruningExperiment(config)
results = experiment.run()
```

### Results Analysis

```python
# Analyze layer sensitivity
layer_results = results['layer_results']
for layer_name, layer_data in layer_results.items():
    print(f"\nLayer: {layer_name}")
    for pct, acc in zip(layer_data['pruning_percentages'], 
                       layer_data['accuracies']):
        print(f"  {pct*100}% pruned: {acc:.2f}% accuracy")
```

## Cascading Layer Pruning

This experiment applies pruning in a cascading manner, propagating effects through the network.

### Basic Usage

```python
from alignment_refactor.experiments.cascading import CascadingLayerPruningExperiment

config = ExperimentConfig(
    name="cascading_pruning",
    model_name="cnn2p2",
    dataset_name="cifar10",
    cascade_threshold=0.1,  # Pruning threshold per layer
    propagation_mode="forward"  # or "backward"
)

experiment = CascadingLayerPruningExperiment(config)
results = experiment.run()
```

## Running Multiple Experiments

The framework provides an `ExperimentRunner` for managing multiple experiments:

```python
from alignment_refactor.experiments.runner import ExperimentRunner
from alignment_refactor.experiments.base import ExperimentConfig

# Base configuration
base_config = ExperimentConfig(
    model_name="mlp",
    dataset_name="mnist",
    batch_size=128,
    device="cuda"
)

runner = ExperimentRunner(base_config=base_config)

# Add experiments with different parameters
for dropout_rate in [0.1, 0.3, 0.5]:
    runner.add_experiment(
        "progressive_dropout",
        config_overrides={"dropout_rate": dropout_rate},
        name_suffix=f"dropout_{dropout_rate}"
    )

# Run all experiments
results = runner.run_all()
```

## Configuration Options

### Essential Parameters

- `name`: Experiment identifier
- `model_name`: Model architecture to use
- `dataset_name`: Dataset for evaluation
- `device`: Computing device (cuda/cpu)
- `seed`: Random seed for reproducibility

### Model Configuration

```python
model_config = {
    "hidden_dims": [300, 200, 100],  # For MLP
    "conv_channels": [32, 64],       # For CNN
    "dropout_rate": 0.5,
    "activation": "relu"
}
```

### Training Configuration

```python
training_config = {
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "train_before_dropout": True
}
```

### Metric Configuration

```python
metric_configs = {
    "rayleigh_quotient": {
        "scale_by_norm": False,
        "aggregation_op": "mean"
    },
    "mutual_information": {
        "estimation_method": "gaussian"
    }
}
```

## Advanced Features

### Custom Pruning Functions

```python
def custom_pruning_fn(weights, scores, pruning_fraction):
    """Custom pruning logic"""
    threshold = torch.quantile(scores, pruning_fraction)
    mask = scores > threshold
    return weights * mask.unsqueeze(1)

config.custom_pruning_fn = custom_pruning_fn
```

### Callbacks

```python
def on_pruning_step(experiment, step, metrics):
    """Called after each pruning step"""
    print(f"Step {step}: {metrics}")

config.callbacks = [on_pruning_step]
```

### Checkpointing

```python
config.checkpoint_interval = 1000  # Save every 1000 steps
config.checkpoint_dir = "./checkpoints"
config.save_best = True  # Save best performing model
```

## Best Practices

1. **Start with Small Models**: Test configurations on small models first
2. **Use Appropriate Batch Sizes**: Balance memory usage and training stability
3. **Set Random Seeds**: Ensure reproducibility across runs
4. **Monitor Metrics**: Track multiple metrics for comprehensive analysis
5. **Save Intermediate Results**: Enable checkpointing for long experiments

## Troubleshooting

### Out of Memory Errors

- Reduce batch size
- Use gradient accumulation
- Enable CPU offloading for large models

### Slow Experiments

- Use GPU acceleration
- Reduce number of pruning steps
- Enable parallel data loading

### Inconsistent Results

- Set fixed random seeds
- Disable non-deterministic operations
- Verify data loading consistency 