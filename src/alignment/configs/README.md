# Configuration System Guide

This guide explains how to use the configuration system for running neural network alignment experiments.

## Overview

The configuration system allows you to define all experiment parameters in YAML files, making experiments reproducible and easy to manage. The system supports:

- Model configuration (MLP, CNN, ResNet, etc.)
- Dataset configuration (MNIST, CIFAR, ImageNet)
- Training parameters
- Metric selection and configuration
- Experiment-specific settings
- Checkpointing and logging
- Distributed training
- Weights & Biases integration

## Quick Start

1. **Choose a template configuration:**
   ```bash
   # For MNIST with MLP
   cp templates/mnist_mlp.yaml my_experiment.yaml
   
   # For CIFAR-10 with CNN
   cp templates/cifar10_cnn.yaml my_experiment.yaml
   
   # For comprehensive example with all options
   cp templates/comprehensive_example.yaml my_experiment.yaml
   ```

2. **Edit the configuration file:**
   ```yaml
   name: "my_experiment"
   model_name: "mlp"
   dataset_name: "mnist"
   dropout_fractions: [0.0, 0.2, 0.4, 0.6, 0.8]
   metrics: ["rayleigh_quotient", "mutual_information"]
   ```

3. **Run the experiment:**
   ```bash
   python examples/run_experiment_from_config.py my_experiment.yaml
   ```

## Configuration Structure

### Basic Structure

```yaml
# Experiment identification
name: "experiment_name"
description: "What this experiment does"
tags: ["tag1", "tag2"]

# Model configuration
model_name: "mlp"  # or "cnn2p2", "resnet18", etc.
model_config:
  # Model-specific parameters

# Dataset configuration  
dataset_name: "mnist"  # or "cifar10", "cifar100", etc.
data_path: "./data"

# Training configuration
batch_size: 128
device: "cuda"
training_epochs: 10

# Metrics configuration
metrics: ["rayleigh_quotient"]
```

### Model Configuration

#### MLP
```yaml
model_name: "mlp"
model_config:
  input_dim: 784
  hidden_dims: [300, 200, 100]
  output_dim: 10
  dropout_rate: 0.5
  activation: "relu"  # relu, tanh, sigmoid, gelu
```

#### CNN2P2
```yaml
model_name: "cnn2p2"
model_config:
  in_channels: 3
  conv_channels: [32, 64]
  kernel_sizes: [5, 5]
  hidden_fc_dim: 256
  output_dim: 10
  example_input_hw: [32, 32]
```

#### Pre-trained Models
```yaml
model_name: "resnet18"  # Any torchvision model
pretrained: true
model_config:
  num_classes: 10  # Override default
```

### Metrics Configuration

```yaml
metrics:
  - "rayleigh_quotient"
  - "mutual_information"
  - "cka"

metric_configs:
  rayleigh_quotient:
    scale_by_norm: false
    aggregation_op: "mean"  # For CNNs: mean, max, sum, var
    force_cpu: true
    
  mutual_information:
    estimation_method: "gaussian"
    num_samples: 1000
    
  cka:
    kernel: "linear"  # or "rbf"
    threshold: 0.01
```

### Experiment Types

#### Progressive Dropout
```yaml
dropout_fractions: [0.0, 0.2, 0.4, 0.6, 0.8]
dropout_mode: "scaled"  # or "unscaled"
pruning_strategy: "magnitude"  # or "random"
```

#### Eigenvector Alignment
```yaml
num_components: 10
compute_full_eigenspectrum: false
```

#### Layer-Isolated Pruning
```yaml
pruning_percentages: [0.1, 0.3, 0.5, 0.7, 0.9]
layer_pruning_order: "sequential"  # or "reverse", "random"
```

## Environment Variables

The configuration system supports environment variables with defaults:

```yaml
data_path: ${DATA_PATH:./data}  # Uses $DATA_PATH or ./data if not set
device: ${DEVICE:cuda}
checkpoint_dir: ${CHECKPOINT_DIR:./checkpoints}
```

## Command-Line Overrides

Override configuration values from the command line:

```bash
# Override device
python run_experiment_from_config.py config.yaml --device cpu

# Override batch size and epochs
python run_experiment_from_config.py config.yaml --batch-size 256 --epochs 20

# Skip training
python run_experiment_from_config.py config.yaml --no-train
```

## Advanced Features

### Distributed Training
```yaml
distributed: true
world_size: 4
dist_backend: "nccl"
```

### Weights & Biases Integration
```yaml
wandb_project: "my_project"
wandb_entity: "my_team"
wandb_config:
  log_model: true
  log_gradients: true
  gradient_log_freq: 100
```

### Memory Optimization
```yaml
gradient_accumulation_steps: 4
mixed_precision: true
force_cpu_for_large_metric_ops: true
```

### Plotting Configuration
```yaml
plotting:
  save_plots: true
  plot_format: "png"  # or "pdf", "svg"
  dpi: 300
  plot_types: ["metric_vs_dropout", "layer_comparison"]
```

## Complete Example

Here's a complete example for a progressive dropout experiment on CIFAR-10:

```yaml
name: "cifar10_progressive_dropout"
description: "Study alignment during progressive dropout on CIFAR-10"
tags: ["cifar10", "dropout", "alignment"]

# Model
model_name: "cnn2p2"
model_config:
  in_channels: 3
  conv_channels: [64, 128]
  kernel_sizes: [3, 3]
  hidden_fc_dim: 512
  output_dim: 10
  dropout_rate: 0.2
  example_input_hw: [32, 32]

# Dataset
dataset_name: "cifar10"
data_path: ${DATA_PATH:./data}
dataset_config:
  normalize: true
  augmentation: true

# Training
batch_size: 128
device: ${DEVICE:cuda}
seed: 42
train_before_dropout: true
training_epochs: 20
learning_rate: 0.001
optimizer: "adamw"

# Metrics
metrics: ["rayleigh_quotient", "mutual_information", "cka"]
metric_configs:
  rayleigh_quotient:
    scale_by_norm: false
    force_cpu: true
  cka:
    kernel: "linear"

# Experiment specific
dropout_fractions: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_mode: "scaled"

# Outputs
checkpoint_dir: "./checkpoints/cifar10_dropout"
log_dir: "./logs/cifar10_dropout"
wandb_project: "alignment-studies"

# Plotting
plotting:
  save_plots: true
  plot_types: ["metric_vs_dropout", "layer_comparison"]
```

## Tips

1. **Start with templates**: Use the provided templates as starting points
2. **Use environment variables**: For paths and device settings
3. **Version control configs**: Track your experiment configurations in git
4. **Use descriptive names**: Make experiment names self-documenting
5. **Add tags**: Use tags to organize related experiments
6. **Document changes**: Use the description field to note what you're testing

## Troubleshooting

- **Validation errors**: Run `validate_config()` to check for issues
- **Missing models**: Ensure model names match registered models
- **Memory issues**: Enable `force_cpu_for_large_metric_ops`
- **Slow training**: Reduce batch size or use gradient accumulation 