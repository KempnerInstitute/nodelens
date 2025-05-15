# Experiment Framework

This document provides a comprehensive guide to the experiment framework in the alignment codebase, including configuration options, running experiments, and analyzing results.

## Table of Contents

1. [Overview](#overview)
2. [Experiment Configuration](#experiment-configuration)
3. [Running Experiments](#running-experiments)
4. [Experiment Types](#experiment-types)
5. [Analyzing Results](#analyzing-results)
6. [Advanced Usage](#advanced-usage)
7. [Integration with External Tools](#integration-with-external-tools)

## Overview

The experiment framework provides a flexible and configurable way to run alignment experiments. It handles:

- Setting up the experiment environment
- Loading and preprocessing data
- Creating and initializing models
- Training models
- Collecting and analyzing metrics
- Running progressive dropout experiments
- Logging results and visualizations

The framework is designed to be extensible, allowing you to add new experiment types and metrics.

## Experiment Configuration

Experiments are configured using YAML files in the `configs/` directory. These configuration files specify all aspects of the experiment, from the dataset to model architectures and pruning strategies.

### Basic Configuration Structure

```yaml
# Experiment metadata
name: "sample_experiment"
description: "A sample alignment experiment"
seed: 42

# Dataset configuration
dataset:
  name: "mnist"
  path: "/path/to/datasets"
  batch_size: 128
  num_workers: 4

# Model configuration
model:
  type: "mlp"
  num_networks: 5
  hidden_dims: [100, 100]
  activation: "relu"
  dropout: 0.0

# Training configuration
training:
  optimizer: "adam"
  learning_rate: 0.001
  epochs: 10
  scheduler: "cosine"
  save_checkpoints: true

# Metrics configuration
metrics:
  - name: "rq"
    scale_by_norm: true
  - name: "mi_gaussian"
  - name: "redundancy_gaussian"

# Pruning configuration
pruning:
  enabled: true
  strategies: ["high_rq", "low_rq", "random"]
  steps: 10
  step_size: 0.1

# Extra options
extra:
  training_method: "fully_tensorized"
  use_multi_strategy_dropout: true
  wandb_enabled: true
```

### Configuration Options

#### Dataset Options

- `name`: Dataset name (e.g., "mnist", "cifar10", "custom")
- `path`: Path to the dataset directory
- `batch_size`: Batch size for training and evaluation
- `num_workers`: Number of workers for data loading
- `transforms`: Data transformations to apply

#### Model Options

- `type`: Model architecture ("mlp", "cnn", "resnet", etc.)
- `num_networks`: Number of networks to train
- `hidden_dims`: List of hidden dimensions for each layer
- `activation`: Activation function ("relu", "tanh", etc.)
- `dropout`: Dropout probability (0.0 for no dropout)

#### Training Options

- `optimizer`: Optimizer to use ("sgd", "adam", etc.)
- `learning_rate`: Learning rate for the optimizer
- `epochs`: Number of training epochs
- `scheduler`: Learning rate scheduler ("step", "cosine", etc.)
- `save_checkpoints`: Whether to save model checkpoints

#### Metrics Options

Each metric is specified as a dictionary with:
- `name`: Name of the metric
- Additional parameters specific to the metric

#### Pruning Options

- `enabled`: Whether to run pruning experiments
- `strategies`: List of pruning strategies to use
- `steps`: Number of pruning steps
- `step_size`: Fraction of neurons to prune at each step

#### Extra Options

- `training_method`: Method for training multiple networks
- `use_multi_strategy_dropout`: Whether to use multi-strategy dropout
- `wandb_enabled`: Whether to log results to Weights & Biases

## Running Experiments

### Basic Usage

To run an experiment from a configuration file:

```bash
python experiment.py configs/my_experiment.yaml
```

### Command-Line Overrides

You can override configuration options from the command line:

```bash
python experiment.py configs/my_experiment.yaml --seed 123 --model.num_networks 10
```

### Distributed Training

For distributed training on a cluster:

```bash
sbatch cluster/submit_job.sh configs/my_experiment.yaml
```

## Experiment Types

The framework supports several types of experiments:

### Basic Training Experiment

Trains multiple networks and collects metrics during training.

```yaml
experiment_type: "training"
```

### Progressive Dropout Experiment

Trains networks, then progressively drops neurons based on different metrics.

```yaml
experiment_type: "progressive_dropout"
```

### Feature Importance Experiment

Analyzes the importance of features using various metrics.

```yaml
experiment_type: "feature_importance"
```

### Custom Experiments

Custom experiments can be implemented by extending the `ExperimentRunner` class.

## Analyzing Results

Experiment results are saved to the configured output directory. The framework provides utilities for analyzing and visualizing these results.

### Result Structure

Results are saved in the following structure:

```
results/
└── experiment_name/
    ├── config.yaml              # Experiment configuration
    ├── metrics/                 # Computed metrics
    │   ├── layer_0_rq.npy       # RQ scores for layer 0
    │   ├── layer_1_rq.npy       # RQ scores for layer 1
    │   └── ...
    ├── models/                  # Saved model checkpoints
    │   ├── net_0.pt             # Checkpoint for network 0
    │   ├── net_1.pt             # Checkpoint for network 1
    │   └── ...
    ├── pruning/                 # Pruning experiment results
    │   ├── high_rq/             # Results for high_rq strategy
    │   ├── low_rq/              # Results for low_rq strategy
    │   └── random/              # Results for random strategy
    ├── plots/                   # Generated plots
    │   ├── rq_heatmap.png       # Heatmap of RQ scores
    │   ├── accuracy_vs_pruning.png  # Accuracy vs pruning curve
    │   └── ...
    └── summary.json             # Summary of experiment results
```

### Visualization Tools

The framework provides several visualization tools:

```python
from alignment.utils.visualization import (
    plot_metric_heatmap,
    plot_accuracy_vs_pruning,
    plot_metric_distribution
)

# Plot a heatmap of RQ scores
plot_metric_heatmap("results/experiment_name/metrics/layer_0_rq.npy", title="Layer 0 RQ Scores")

# Plot accuracy vs pruning curve
plot_accuracy_vs_pruning("results/experiment_name/pruning/", strategies=["high_rq", "low_rq", "random"])

# Plot distribution of a metric
plot_metric_distribution("results/experiment_name/metrics/layer_0_rq.npy", title="Layer 0 RQ Distribution")
```

## Advanced Usage

### Custom Datasets

To use a custom dataset:

```python
from alignment.data import register_dataset

def load_custom_dataset(path, **kwargs):
    # Load and return your dataset
    return train_dataset, test_dataset

# Register your dataset
register_dataset("custom_dataset", load_custom_dataset)
```

Then in your configuration:

```yaml
dataset:
  name: "custom_dataset"
  path: "/path/to/custom_data"
```

### Custom Models

To use a custom model architecture:

```python
from alignment.networks import register_model

def create_custom_model(input_dim, output_dim, **kwargs):
    # Create and return your model
    return model

# Register your model
register_model("custom_model", create_custom_model)
```

Then in your configuration:

```yaml
model:
  type: "custom_model"
  custom_param1: value1
  custom_param2: value2
```

### Custom Metrics

To use a custom metric:

```python
from alignment.metrics import register_metric

def compute_custom_metric(layer_inputs, layer_weights, **kwargs):
    # Compute and return your metric
    return scores

# Register your metric
register_metric("custom_metric", compute_custom_metric)
```

Then in your configuration:

```yaml
metrics:
  - name: "custom_metric"
    custom_param: value
```

## Integration with External Tools

### Weights & Biases Integration

The framework integrates with Weights & Biases for experiment tracking:

```yaml
extra:
  wandb_enabled: true
  wandb_project: "alignment_experiments"
  wandb_entity: "your_username"
```

### TensorBoard Integration

Results can also be logged to TensorBoard:

```yaml
extra:
  tensorboard_enabled: true
  tensorboard_log_dir: "logs/tensorboard"
```

### Cluster Integration

For running on a cluster, the framework provides integration with SLURM:

```bash
sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:4 cluster/submit_job.sh configs/my_experiment.yaml
```

The framework automatically detects the distributed environment and configures distributed training accordingly. 