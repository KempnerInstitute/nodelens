# API Reference

This document provides a comprehensive reference of the public API for the Network Alignment Analysis library.

## Table of Contents

1. [Metrics API](#metrics-api)
2. [Experiment API](#experiment-api)
3. [Networks API](#networks-api)
4. [Pruning API](#pruning-api)
5. [Data API](#data-api)
6. [Utilities API](#utilities-api)

## Metrics API

The Metrics API provides tools for computing various neural network alignment metrics.

### Core Functions

#### `get_metric(name, **kwargs)`

Returns a configured metric object.

**Parameters:**
- `name` (str): Name of the metric to get
- `**kwargs`: Additional parameters for the metric

**Returns:**
- `_AlignmentMetricImpl`: An object that computes the requested metric

**Example:**
```python
from alignment.metrics import get_metric

rq_metric = get_metric("rq", scale_by_norm=True)
scores = rq_metric.compute_per_node_scores(layer_inputs, layer_weights)
```

#### `compute_metrics_for_layers(layers, metric_configs, inputs, outputs=None, device=None, verbose=False)`

Computes metrics for multiple layers.

**Parameters:**
- `layers` (list): List of layer objects
- `metric_configs` (list): List of metric configurations
- `inputs` (torch.Tensor): Input data
- `outputs` (torch.Tensor, optional): Output data
- `device` (str, optional): Device to compute metrics on
- `verbose` (bool): Whether to print verbose output

**Returns:**
- `dict`: Dictionary of metric results for each layer

**Example:**
```python
from alignment.metrics import compute_metrics_for_layers

metric_configs = [
    {"name": "rq", "scale_by_norm": True},
    {"name": "mi_gaussian"}
]

layer_metrics = compute_metrics_for_layers(
    layers=model.layers,
    metric_configs=metric_configs,
    inputs=inputs
)
```

#### `compute_all_node_scores(model, metric_configs, data_loader, device=None, num_batches=5, verbose=False)`

Computes metrics for all nodes in a model.

**Parameters:**
- `model` (torch.nn.Module): Model to compute metrics for
- `metric_configs` (list): List of metric configurations
- `data_loader` (torch.utils.data.DataLoader): Data loader for inputs
- `device` (str, optional): Device to compute metrics on
- `num_batches` (int): Number of batches to use for computation
- `verbose` (bool): Whether to print verbose output

**Returns:**
- `dict`: Dictionary of metric results for each layer and node

**Example:**
```python
from alignment.metrics import compute_all_node_scores

metric_configs = [
    {"name": "rq", "scale_by_norm": True},
    {"name": "mi_gaussian"}
]

scores = compute_all_node_scores(
    model=model,
    metric_configs=metric_configs,
    data_loader=train_loader,
    device="cuda",
    num_batches=5
)
```

### Available Metrics

See the [Metrics Documentation](metrics/README) for a full list of available metrics.

## Experiment API

The Experiment API provides tools for running alignment experiments.

### Core Classes

#### `ExperimentRunner`

Main class for running experiments.

**Methods:**
- `__init__(config_path=None, config=None)`: Initialize with configuration file or dict
- `run()`: Run the experiment
- `load_checkpoint(path)`: Load experiment checkpoint
- `save_checkpoint(path)`: Save experiment checkpoint

**Example:**
```python
from alignment.experiment import ExperimentRunner

runner = ExperimentRunner(config_path="configs/my_experiment.yaml")
results = runner.run()
```

#### `ExperimentConfig`

Class for managing experiment configuration.

**Methods:**
- `__init__(config=None, config_path=None)`: Initialize with configuration dict or file
- `save(path)`: Save configuration to file
- `load(path)`: Load configuration from file
- `update(updates)`: Update configuration with new values

**Example:**
```python
from alignment.experiment import ExperimentConfig

config = ExperimentConfig(config_path="configs/base_config.yaml")
config.update({"model.num_networks": 10, "training.epochs": 20})
config.save("configs/updated_config.yaml")
```

## Networks API

The Networks API provides tools for creating and managing neural networks.

### Core Functions

#### `create_model(config, input_dim, output_dim)`

Creates a model according to the configuration.

**Parameters:**
- `config` (dict): Model configuration
- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension

**Returns:**
- `torch.nn.Module`: Created model

**Example:**
```python
from alignment.networks import create_model

model_config = {
    "type": "mlp",
    "hidden_dims": [100, 100],
    "activation": "relu"
}

model = create_model(model_config, input_dim=784, output_dim=10)
```

#### `create_multiple_models(config, input_dim, output_dim, num_networks=None)`

Creates multiple models with the same architecture.

**Parameters:**
- `config` (dict): Model configuration
- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension
- `num_networks` (int, optional): Number of networks to create

**Returns:**
- `list`: List of created models

**Example:**
```python
from alignment.networks import create_multiple_models

model_config = {
    "type": "mlp",
    "hidden_dims": [100, 100],
    "activation": "relu"
}

models = create_multiple_models(model_config, input_dim=784, output_dim=10, num_networks=5)
```

#### `create_optimizer(model, config)`

Creates an optimizer for the model.

**Parameters:**
- `model` (torch.nn.Module): Model to optimize
- `config` (dict): Optimizer configuration

**Returns:**
- `torch.optim.Optimizer`: Created optimizer

**Example:**
```python
from alignment.networks import create_optimizer

optimizer_config = {
    "type": "adam",
    "learning_rate": 0.001,
    "weight_decay": 1e-5
}

optimizer = create_optimizer(model, optimizer_config)
```

### Network Training Functions

#### `train_model(model, train_loader, optimizer, config, device=None)`

Trains a model for one epoch.

**Parameters:**
- `model` (torch.nn.Module): Model to train
- `train_loader` (torch.utils.data.DataLoader): Training data loader
- `optimizer` (torch.optim.Optimizer): Optimizer
- `config` (dict): Training configuration
- `device` (str, optional): Device to train on

**Returns:**
- `dict`: Dictionary of training metrics

**Example:**
```python
from alignment.networks import train_model

metrics = train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    config=training_config,
    device="cuda"
)
```

#### `train_multiple_models(models, train_loader, optimizers, config, method="auto", device=None)`

Trains multiple models.

**Parameters:**
- `models` (list): List of models to train
- `train_loader` (torch.utils.data.DataLoader): Training data loader
- `optimizers` (list): List of optimizers
- `config` (dict): Training configuration
- `method` (str): Training method ("auto", "sequential", "tensorized", "fully_tensorized")
- `device` (str, optional): Device to train on

**Returns:**
- `dict`: Dictionary of training metrics for each model

**Example:**
```python
from alignment.networks import train_multiple_models

metrics = train_multiple_models(
    models=models,
    train_loader=train_loader,
    optimizers=optimizers,
    config=training_config,
    method="fully_tensorized",
    device="cuda"
)
```

## Pruning API

The Pruning API provides tools for analyzing network pruning.

### Core Classes

#### `ProgressiveDropout`

Class for running progressive dropout experiments.

**Methods:**
- `__init__(model, strategy="high_rq", use_multi_strategy=False)`: Initialize
- `run_dropout_experiment(data_loader, steps=10)`: Run dropout experiment
- `compute_scores(data_loader)`: Compute neuron scores
- `apply_dropout_mask(mask)`: Apply dropout mask to model

**Example:**
```python
from alignment.pruning import ProgressiveDropout

dropout = ProgressiveDropout(
    model=model,
    strategy="high_rq",
    use_multi_strategy=True
)

results = dropout.run_dropout_experiment(
    data_loader=test_loader,
    steps=10
)
```

### Pruning Strategies

The library includes several pruning strategies:

- `"high_rq"`: Prune neurons with highest Rayleigh Quotient
- `"low_rq"`: Prune neurons with lowest Rayleigh Quotient
- `"random"`: Randomly prune neurons
- `"high_mi"`: Prune neurons with highest Mutual Information
- `"low_mi"`: Prune neurons with lowest Mutual Information

## Data API

The Data API provides tools for working with datasets.

### Core Functions

#### `get_dataset(name, path, **kwargs)`

Gets a dataset by name.

**Parameters:**
- `name` (str): Name of the dataset
- `path` (str): Path to the dataset
- `**kwargs`: Additional dataset parameters

**Returns:**
- `tuple`: Tuple of (train_dataset, test_dataset)

**Example:**
```python
from alignment.data import get_dataset

train_dataset, test_dataset = get_dataset(
    name="mnist",
    path="/path/to/datasets",
    transform=transform
)
```

#### `create_data_loaders(train_dataset, test_dataset, batch_size=128, num_workers=4)`

Creates data loaders for training and testing.

**Parameters:**
- `train_dataset` (torch.utils.data.Dataset): Training dataset
- `test_dataset` (torch.utils.data.Dataset): Testing dataset
- `batch_size` (int): Batch size
- `num_workers` (int): Number of workers

**Returns:**
- `tuple`: Tuple of (train_loader, test_loader)

**Example:**
```python
from alignment.data import create_data_loaders

train_loader, test_loader = create_data_loaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=128,
    num_workers=4
)
```

## Utilities API

The Utilities API provides various utility functions.

### Visualization Functions

#### `plot_metric_heatmap(metric_scores, title=None, figsize=(10, 8))`

Plots a heatmap of metric scores.

**Parameters:**
- `metric_scores` (numpy.ndarray): Metric scores
- `title` (str, optional): Plot title
- `figsize` (tuple): Figure size

**Returns:**
- `matplotlib.figure.Figure`: Figure object

**Example:**
```python
from alignment.utils.visualization import plot_metric_heatmap

fig = plot_metric_heatmap(
    metric_scores=layer_metrics["layer_0_rq"],
    title="Layer 0 RQ Scores"
)
fig.savefig("plots/rq_heatmap.png")
```

#### `plot_accuracy_vs_pruning(results_dir, strategies=None, figsize=(10, 6))`

Plots accuracy vs pruning curve.

**Parameters:**
- `results_dir` (str): Directory containing results
- `strategies` (list, optional): List of strategies to plot
- `figsize` (tuple): Figure size

**Returns:**
- `matplotlib.figure.Figure`: Figure object

**Example:**
```python
from alignment.utils.visualization import plot_accuracy_vs_pruning

fig = plot_accuracy_vs_pruning(
    results_dir="results/experiment_name/pruning",
    strategies=["high_rq", "low_rq", "random"]
)
fig.savefig("plots/accuracy_vs_pruning.png")
```

### IO Functions

#### `save_results(results, path)`

Saves results to disk.

**Parameters:**
- `results` (dict): Dictionary of results
- `path` (str): Path to save results to

**Example:**
```python
from alignment.utils.io import save_results

save_results(results, "results/experiment_name/metrics.json")
```

#### `load_results(path)`

Loads results from disk.

**Parameters:**
- `path` (str): Path to load results from

**Returns:**
- `dict`: Dictionary of results

**Example:**
```python
from alignment.utils.io import load_results

results = load_results("results/experiment_name/metrics.json")
``` 